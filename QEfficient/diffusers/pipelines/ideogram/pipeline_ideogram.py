# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers import Ideogram4Pipeline, Ideogram4PromptEnhancerHead
from diffusers.pipelines.ideogram4.pipeline_ideogram4 import (
    QWEN3_VL_ACTIVATION_LAYERS,
    _expand_tensor_to_effective_batch,
    _logit_normal_sigmas,
    _resolution_aware_mu,
)
from diffusers.pipelines.ideogram4.pipeline_output import Ideogram4PipelineOutput
from tqdm import tqdm

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.diffusers.models.pytorch_transforms import (
    AttentionTransform,
    Bnb4BitLinearToLinearTransform,
    CustomOpsTransform,
    NormalizationTransform,
)
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ModulePerf,
    QEffPipelineOutput,
    compile_modules_parallel,
    compile_modules_sequential,
    config_manager,
    set_execute_params,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.pytorch_transforms import (
    CustomOpsTransform as TransformersCustomOpsTransform,
    KVCacheTransform,
)
from QEfficient.transformers.models.qwen3_vl.modeling_qwen3_vl import QEffQwen3VLTextDecoderLayer
from QEfficient.utils import constants
from QEfficient.utils.logging_utils import logger


IDEOGRAM_QEFF_TEXT_SEQ_LEN = 512


@contextmanager
def _allow_cpu_bnb_4bit_loading(enabled: bool):
    """Temporarily bypass upstream bitsandbytes GPU checks for CPU-side NF4 dequantization.

    Diffusers/Transformers normally reject 4-bit bitsandbytes checkpoints when no CUDA GPU is present.
    QEfficient only needs to instantiate those modules long enough to replace them with regular
    ``torch.nn.Linear`` modules, so this context allows loading to continue on CPU-only hosts.
    """
    if not enabled:
        yield
        return

    patches = []
    try:
        from diffusers.quantizers.bitsandbytes.bnb_quantizer import BnB4BitDiffusersQuantizer

        patches.append(
            (BnB4BitDiffusersQuantizer, "validate_environment", BnB4BitDiffusersQuantizer.validate_environment)
        )
        patches.append((BnB4BitDiffusersQuantizer, "update_device_map", BnB4BitDiffusersQuantizer.update_device_map))
        BnB4BitDiffusersQuantizer.validate_environment = lambda self, *args, **kwargs: None
        BnB4BitDiffusersQuantizer.update_device_map = lambda self, device_map=None: _cpu_device_map(device_map)
    except Exception as exc:
        logger.debug("Could not patch Diffusers BnB4Bit quantizer validation: %s", exc)

    try:
        from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer

        patches.append((Bnb4BitHfQuantizer, "validate_environment", Bnb4BitHfQuantizer.validate_environment))
        patches.append((Bnb4BitHfQuantizer, "update_device_map", Bnb4BitHfQuantizer.update_device_map))
        Bnb4BitHfQuantizer.validate_environment = lambda self, *args, **kwargs: None
        Bnb4BitHfQuantizer.update_device_map = lambda self, device_map=None: _cpu_device_map(device_map)
    except Exception as exc:
        logger.debug("Could not patch Transformers BnB4Bit quantizer validation: %s", exc)

    try:
        yield
    finally:
        for obj, attr, original in patches:
            setattr(obj, attr, original)


def _cpu_device_map(device_map=None):
    if device_map is None:
        return {"": "cpu"}
    if isinstance(device_map, str):
        return "cpu" if device_map.startswith("cuda") else device_map
    if isinstance(device_map, dict):
        return {
            key: "cpu" if isinstance(value, str) and value.startswith("cuda") else value
            for key, value in device_map.items()
        }
    return device_map


def _clear_quantization_config(module: torch.nn.Module) -> None:
    config = getattr(module, "config", None)
    if config is None:
        return

    if hasattr(config, "quantization_config"):
        try:
            delattr(config, "quantization_config")
        except Exception:
            config.quantization_config = None

    if hasattr(config, "to_dict") and isinstance(getattr(config, "_pre_quantization_dtype", None), torch.dtype):
        try:
            delattr(config, "_pre_quantization_dtype")
        except Exception:
            pass


def _dequantize_loaded_bnb_modules(model: Ideogram4Pipeline) -> None:
    """Convert loaded bitsandbytes 4-bit modules in all pipeline components to regular Linear modules."""
    for component_name, component in getattr(model, "components", {}).items():
        if isinstance(component, torch.nn.Module):
            component.cpu()
            component, transformed = Bnb4BitLinearToLinearTransform.apply(component)
            if transformed:
                logger.info("Dequantized bitsandbytes 4-bit modules in Ideogram component: %s", component_name)
            _clear_quantization_config(component)
            setattr(model, component_name, component)


def _qaic_runtime_devices(module: QEFFBaseModel) -> Optional[set[int]]:
    """Return the explicit device IDs pinned to ``module``, or ``None`` when auto-allocated.

    Ideogram intentionally never pins ``device_ids`` on its modules: each ``QAICInferenceSession``
    is created without ``device_ids`` (see ``_ensure_qaic_session``), so the QAIC runtime picks its
    own free devices for every session from whatever is available on the host. ``None`` here means
    "runtime-managed pool", not "device 0" -- treating it as device 0 previously caused every module
    to be considered as colliding with every other module, forcing needless deactivate/activate churn
    every denoising step even on hosts with plenty of spare devices.
    """
    device_ids = getattr(module, "device_ids", None)
    return set(device_ids) if device_ids else None


def _modules_share_qaic_devices(first_module: QEFFBaseModel, second_module: QEFFBaseModel) -> bool:
    """Return True only when both modules have *explicit*, overlapping ``device_ids``.

    Auto-allocated modules (``device_ids is None``) are assumed independent -- the QAIC runtime is
    responsible for giving each such session its own free devices, so they can safely stay resident
    and run concurrently as long as the host has enough total devices for both QPCs at once.
    """
    devices_first = _qaic_runtime_devices(first_module)
    devices_second = _qaic_runtime_devices(second_module)
    if devices_first is None or devices_second is None:
        return False
    return bool(devices_first & devices_second)


def _ensure_qaic_session(module: QEFFBaseModel) -> QAICInferenceSession:
    if module.qpc_session is None:
        module.qpc_session = QAICInferenceSession(
            str(module.qpc_path),
            device_ids=module.device_ids,
            data_path_timeout_ms=module.data_path_timeout_ms,
        )
    elif not getattr(module.qpc_session, "is_active", True):
        module.qpc_session.activate()
    return module.qpc_session


def _deactivate_qaic_session(module: QEFFBaseModel) -> None:
    if module.qpc_session is not None and getattr(module.qpc_session, "is_active", False):
        module.qpc_session.deactivate()


def _release_qaic_session(module: QEFFBaseModel) -> None:
    if module.qpc_session is None:
        return
    _deactivate_qaic_session(module)
    module.qpc_session = None
    gc.collect()


class QEffIdeogram4TransformerModel(QEFFBaseModel):
    """QEfficient wrapper for Ideogram4 conditional/unconditional transformers."""

    _pytorch_transforms = [
        Bnb4BitLinearToLinearTransform,
        AttentionTransform,
        CustomOpsTransform,
        NormalizationTransform,
    ]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: torch.nn.Module, module_name: str = "transformer") -> None:
        super().__init__(model, module_name=module_name)
        self.model = model
        self.hash_params["module_name"] = module_name
        self.hash_params["encoder_hidden_states_projected"] = True
        self.hash_params["encoder_hidden_dim"] = self.model.llm_cond_proj.out_features
        self.hash_params["use_attention_mask"] = getattr(self.model, "qeff_use_attention_mask", True)
        self.module_name = module_name

    def set_attention_mask_enabled(self, enabled: bool) -> None:
        previous = getattr(self.model, "qeff_use_attention_mask", True)
        if hasattr(self.model, "qeff_set_attention_mask_enabled"):
            self.model.qeff_set_attention_mask_enabled(enabled)
        else:
            self.model.qeff_use_attention_mask = enabled
        self.hash_params["use_attention_mask"] = enabled
        if previous != enabled:
            _release_qaic_session(self)
            self.onnx_path = None
            self.qpc_path = None

    @property
    def get_model_config(self) -> Dict:
        return self.model.config.__dict__

    def get_onnx_params(
        self,
        batch_size: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
        seq_len: int = 4096,
        encoder_seq_len: Optional[int] = None,
    ):
        # Rotary cos/sin are precomputed on host in fp32 (see `QEffIdeogram4Pipeline.__call__`) and passed in
        # directly instead of raw `position_ids`. Ideogram4's image position ids start at 65536
        # (`IMAGE_POSITION_OFFSET`), which already exceeds the fp16 max representable value (~65504); computing
        # `inv_freq @ position_ids` inside a graph compiled with `convert_to_fp16`/`mxfp6_matmul` silently
        # overflows to `inf`/`NaN` on-device and corrupts every image token's attention (a blank/garbage image).
        # cos/sin are bounded to [-1, 1] and therefore safe under fp16/mxfp6 compilation.
        head_dim = self.model.config.attention_head_dim
        encoder_hidden_dim = self.model.llm_cond_proj.out_features
        example_inputs = {
            "hidden_states": torch.randn(batch_size, seq_len, self.model.config.in_channels, dtype=torch.float32),
            "timestep": torch.ones(batch_size, dtype=torch.float32),
            "rotary_emb_cos": torch.ones(batch_size, seq_len, head_dim, dtype=torch.float32),
            "rotary_emb_sin": torch.zeros(batch_size, seq_len, head_dim, dtype=torch.float32),
            "indicator": torch.full((batch_size, seq_len), 2, dtype=torch.int64),
            "return_dict": False,
        }
        if encoder_seq_len is not None:
            example_inputs["encoder_hidden_states"] = torch.randn(
                batch_size, encoder_seq_len, encoder_hidden_dim, dtype=torch.float32
            )
        output_names = ["output"]
        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "seq_len"},
            "timestep": {0: "batch_size"},
            "rotary_emb_cos": {0: "batch_size", 1: "seq_len"},
            "rotary_emb_sin": {0: "batch_size", 1: "seq_len"},
            "indicator": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size", 1: "seq_len"},
        }
        if getattr(self.model, "qeff_use_attention_mask", True):
            example_inputs["segment_ids"] = torch.ones(batch_size, seq_len, dtype=torch.int64)
            dynamic_axes["segment_ids"] = {0: "batch_size", 1: "seq_len"}
        if encoder_seq_len is not None:
            dynamic_axes["encoder_hidden_states"] = {0: "batch_size", 1: "encoder_seq_len"}
        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        if hasattr(self.model.config, "_use_default_values"):
            self.model.config["_use_default_values"].sort()
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            offload_pt_weights=False,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        # Ideogram exports one specialization per transformer module. Avoid
        # `-network-specialization-config` here: qaic-compile reports "No node under name"
        # for the single-graph Diffusers ONNX even when the config name matches `main_graph`.
        # Fixed symbols are equivalent for this use case and are accepted as regular compiler flags.
        spec = specializations[0]
        compiler_options["onnx_define_symbol"] = [
            f"batch_size,{spec['batch_size']}",
            f"seq_len,{spec['seq_len']}",
        ]
        if "encoder_seq_len" in spec:
            compiler_options["onnx_define_symbol"].append(f"encoder_seq_len,{spec['encoder_seq_len']}")
        self._compile(
            specializations=None,
            **compiler_options,
        )


class QEffIdeogram4PromptEnhancerHead(QEFFBaseModel):
    """QEfficient wrapper for the Ideogram4 prompt-enhancer LM head.

    The ``diffusers/qwen3-vl-8b-instruct-lm-head`` repository contains only the
    lightweight ``Ideogram4PromptEnhancerHead`` linear projection. The Qwen3-VL
    body that consumes this head is already onboarded in QEfficient; this wrapper
    makes the Diffusers head exportable/compilable and can patch the original
    head's ``forward``/``lm_head.forward`` so Diffusers prompt upsampling invokes
    the QAIC QPC at runtime.
    """

    _pytorch_transforms = [Bnb4BitLinearToLinearTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model, module_name="prompt_enhancer_head")
        self.model = model
        self.hash_params["module_name"] = "prompt_enhancer_head"
        self.runtime_perf = []
        self._runtime_patched = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        allow_cpu_nf4_dequant: Optional[bool] = None,
        **kwargs,
    ):
        """Load and wrap the Diffusers Ideogram4 prompt-enhancer LM head.

        This is the QEff-facing loader for
        ``diffusers/qwen3-vl-8b-instruct-lm-head``. It mirrors the NF4 CPU-loading
        path used by the Ideogram pipeline so a quantized head can be materialized
        and dequantized before ONNX export when necessary.
        """
        if allow_cpu_nf4_dequant is None:
            allow_cpu_nf4_dequant = not torch.cuda.is_available()

        dtype = kwargs.pop("dtype", None) or kwargs.pop("torch_dtype", None)
        with _allow_cpu_bnb_4bit_loading(allow_cpu_nf4_dequant):
            model = Ideogram4PromptEnhancerHead.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if dtype is not None:
            model = model.to(dtype=dtype)
        model, _ = Bnb4BitLinearToLinearTransform.apply(model)
        _clear_quantization_config(model)
        return cls(model)

    @property
    def get_model_config(self) -> Dict:
        return self.model.config.__dict__

    def get_onnx_params(
        self,
        batch_size: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
        seq_len: int = 1,
    ):
        hidden_size = getattr(self.model.config, "hidden_size", self.model.lm_head.in_features)
        example_inputs = {
            "hidden_states": torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32),
        }
        output_names = ["logits"]
        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        }
        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        if hasattr(self.model.config, "_use_default_values"):
            self.model.config["_use_default_values"].sort()
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            offload_pt_weights=False,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

    def compile(self, specializations: Optional[List[Dict]] = None, **compiler_options) -> str:
        if specializations is None:
            specializations = [{"batch_size": 1, "seq_len": 1, "_graph_name": "prompt_enhancer_head"}]
        if self.onnx_path is None and compiler_options.get("onnx_path") is None:
            seq_len = max(int(spec.get("seq_len", 1)) for spec in specializations)
            example_inputs, dynamic_axes, output_names = self.get_onnx_params(seq_len=seq_len)
            self.export(
                inputs=example_inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_dir=compiler_options.get("compile_dir"),
                use_onnx_subfunctions=compiler_options.get("use_onnx_subfunctions", False),
            )
        if compiler_options.get("onnx_path") is None:
            compiler_options["onnx_path"] = self.onnx_path
        return self._compile(specializations=specializations, **compiler_options)

    def patch_for_qaic_runtime(self) -> None:
        """Route Diffusers prompt-enhancer head calls through the compiled QAIC session."""
        if self._runtime_patched or self.qpc_path is None:
            return

        def qaic_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            squeeze_seq = hidden_states.dim() == 2
            head_inputs = hidden_states.unsqueeze(1) if squeeze_seq else hidden_states
            head_inputs = head_inputs.to(torch.float32)

            head_session = _ensure_qaic_session(self)

            logits_shape = (*head_inputs.shape[:-1], self.model.config.vocab_size)
            head_session.set_buffers({"logits": np.empty(logits_shape, dtype=np.float32)})
            start = time.perf_counter()
            logits = head_session.run({"hidden_states": head_inputs.detach().cpu().numpy()})["logits"]
            self.runtime_perf.append(time.perf_counter() - start)
            output = torch.from_numpy(logits).to(device=hidden_states.device, dtype=hidden_states.dtype)
            return output.squeeze(1) if squeeze_seq else output

        self.model.forward = qaic_forward
        self.model.lm_head.forward = qaic_forward
        self._runtime_patched = True


class _IdeogramQwenTextFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        text_encoder: torch.nn.Module,
        activation_layers: tuple[int, ...],
        conditioning_transformer: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        num_layers = len(text_encoder.language_model.layers)
        self.activation_layers = tuple(layer_idx for layer_idx in activation_layers if layer_idx < num_layers)
        self.config = text_encoder.config
        self.conditioning_norm = None
        self.conditioning_proj = None
        if conditioning_transformer is not None:
            self.conditioning_norm = conditioning_transformer.llm_cond_norm
            self.conditioning_proj = conditioning_transformer.llm_cond_proj

        raw_features_dim = text_encoder.config.text_config.hidden_size * len(self.activation_layers)
        self.output_features_dim = (
            self.conditioning_proj.out_features if self.conditioning_proj is not None else raw_features_dim
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
            return_ideogram_text_features=True,
            ideogram_activation_layers=self.activation_layers,
        )
        feature_mask = attention_mask.to(text_features.dtype).unsqueeze(-1)
        text_features = text_features * feature_mask
        if self.conditioning_proj is None:
            return text_features

        text_features = self.conditioning_norm(text_features)
        text_features = self.conditioning_proj(text_features)
        return text_features * feature_mask


class QEffIdeogram4TextEncoder(QEFFBaseModel):
    """QEfficient wrapper for Ideogram4's Qwen3-VL text feature extractor."""

    _pytorch_transforms = [Bnb4BitLinearToLinearTransform, TransformersCustomOpsTransform, KVCacheTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: torch.nn.Module, conditioning_transformer: Optional[torch.nn.Module] = None) -> None:
        feature_extractor = _IdeogramQwenTextFeatureExtractor(
            model,
            QWEN3_VL_ACTIVATION_LAYERS,
            conditioning_transformer=conditioning_transformer,
        )
        super().__init__(feature_extractor, module_name="text_encoder")
        self.model = feature_extractor
        self.hash_params["module_name"] = "text_encoder"
        self.hash_params["ideogram_activation_layers"] = feature_extractor.activation_layers
        self.hash_params["output_features_dim"] = feature_extractor.output_features_dim
        self.hash_params["projects_conditioning"] = feature_extractor.conditioning_proj is not None

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffQwen3VLTextDecoderLayer}

    @property
    def get_model_config(self) -> Dict:
        return self.model.config.__dict__

    def get_onnx_params(
        self,
        batch_size: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
        seq_len: int = 2048,
    ):
        position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, 1, seq_len).expand(4, batch_size, seq_len)
        example_inputs = {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.int64),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.int64),
            "position_ids": position_ids,
        }
        output_names = ["text_features"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "position_ids": {1: "batch_size", 2: "seq_len"},
            "text_features": {0: "batch_size", 1: "seq_len"},
        }
        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        if hasattr(self.model.config, "_use_default_values"):
            self.model.config["_use_default_values"].sort()
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            offload_pt_weights=False,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        spec = specializations[0]
        compiler_options["onnx_define_symbol"] = [
            f"batch_size,{spec['batch_size']}",
            f"seq_len,{spec['seq_len']}",
        ]
        self._compile(
            specializations=None,
            **compiler_options,
        )


class QEffIdeogram4VAE(QEFFBaseModel):
    """QEfficient wrapper for the Ideogram4 AutoencoderKLFlux2 decoder."""

    _pytorch_transforms = [Bnb4BitLinearToLinearTransform, CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        self.model = model
        self.model.forward = lambda latent_sample, return_dict=False: self.model.decode(latent_sample, return_dict)

    @property
    def get_model_config(self) -> Dict:
        return self.model.config.__dict__

    def get_onnx_params(self, latent_height: int = 64, latent_width: int = 64):
        latent_channels = getattr(self.model.config, "latent_channels", 32)
        example_inputs = {
            "latent_sample": torch.randn(
                constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
                latent_channels,
                latent_height,
                latent_width,
                dtype=torch.float32,
            ),
            "return_dict": False,
        }
        output_names = ["sample"]
        dynamic_axes = {
            "latent_sample": {0: "batch_size", 2: "latent_height", 3: "latent_width"},
            "sample": {0: "batch_size", 2: "height", 3: "width"},
        }
        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = {},
    ) -> str:
        if hasattr(self.model.config, "_use_default_values"):
            self.model.config["_use_default_values"].sort()
        # `QEffIdeogram4Pipeline.__call__` reads `self.vae_decode.model.bn.running_mean`/`running_var`
        # (real PyTorch buffers) at runtime to denormalize latents before VAE decode. Do not offload
        # these buffers to meta tensors after ONNX export, or that lookup fails with
        # `NotImplementedError: Cannot copy out of meta tensor; no data!`. Matches the
        # `offload_pt_weights=False` used by the other Ideogram4 wrappers (transformer, prompt enhancer head).
        export_kwargs = {"offload_pt_weights": False, **export_kwargs}
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            **export_kwargs,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        self._compile(specializations=specializations, **compiler_options)


class QEffIdeogram4Pipeline:
    """QEfficient Ideogram4 pipeline.

    This wrapper keeps Ideogram4 prompt encoding/scheduler logic from Diffusers and
    runs the conditional transformer, unconditional transformer, and VAE decoder on
    QAIC. NF4/bitsandbytes 4-bit Linear modules in these exported modules are
    dequantized to regular floating-point ``nn.Linear`` by
    ``Bnb4BitLinearToLinearTransform`` before ONNX export.
    """

    _hf_auto_class = Ideogram4Pipeline

    def __init__(self, model: Ideogram4Pipeline, *args, **kwargs):
        self.model = model
        self.scheduler = model.scheduler
        self.vae_scale_factor = model.vae_scale_factor
        self.patch_size = model.patch_size
        self.image_processor = model.image_processor
        self.tokenizer = model.tokenizer
        self.text_encoder = model.text_encoder
        self.qeff_text_encoder = None
        self.prompt_enhancer_head = getattr(model, "prompt_enhancer_head", None)
        self.qeff_prompt_enhancer_head = (
            QEffIdeogram4PromptEnhancerHead(self.prompt_enhancer_head)
            if isinstance(self.prompt_enhancer_head, torch.nn.Module)
            else None
        )
        self._prompt_enhancer = None
        self._caption_logits_processor = None

        self.transformer = QEffIdeogram4TransformerModel(model.transformer, module_name="transformer")
        self.unconditional_transformer = QEffIdeogram4TransformerModel(
            model.unconditional_transformer, module_name="unconditional_transformer"
        )
        self.vae_decode = QEffIdeogram4VAE(model.vae)

        self.modules = {
            "transformer": self.transformer,
            "unconditional_transformer": self.unconditional_transformer,
            "vae_decoder": self.vae_decode,
        }
        if self.qeff_prompt_enhancer_head is not None:
            self.modules["prompt_enhancer_head"] = self.qeff_prompt_enhancer_head

    def _ensure_qeff_text_encoder(self) -> None:
        if self.qeff_text_encoder is None:
            self.qeff_text_encoder = QEffIdeogram4TextEncoder(self.text_encoder, self.transformer.model)
            self.modules = {"text_encoder": self.qeff_text_encoder, **self.modules}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        allow_cpu_nf4_dequant: Optional[bool] = None,
        **kwargs,
    ):
        """Load an Ideogram4 pipeline and dequantize NF4/bitsandbytes modules for QEff export.

        ``diffusers`` and ``transformers`` normally require CUDA for bitsandbytes 4-bit loading. On CPU-only
        QEfficient hosts, set ``allow_cpu_nf4_dequant=True`` (or leave it as ``None``) to bypass that validation,
        instantiate the NF4 modules on CPU, and immediately replace them with regular ``torch.nn.Linear`` modules.
        """
        if allow_cpu_nf4_dequant is None:
            allow_cpu_nf4_dequant = not torch.cuda.is_available()

        qeff_prompt_enhancer_head = kwargs.get("prompt_enhancer_head")
        if isinstance(qeff_prompt_enhancer_head, QEffIdeogram4PromptEnhancerHead):
            kwargs["prompt_enhancer_head"] = qeff_prompt_enhancer_head.model
        else:
            qeff_prompt_enhancer_head = None

        with _allow_cpu_bnb_4bit_loading(allow_cpu_nf4_dequant):
            model = cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        _dequantize_loaded_bnb_modules(model)
        qeff_model = cls(model=model, pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)
        if qeff_prompt_enhancer_head is not None:
            qeff_model.prompt_enhancer_head = qeff_prompt_enhancer_head.model
            qeff_model.qeff_prompt_enhancer_head = qeff_prompt_enhancer_head
            qeff_model.model.prompt_enhancer_head = qeff_prompt_enhancer_head.model
            qeff_model.modules["prompt_enhancer_head"] = qeff_prompt_enhancer_head
        return qeff_model

    def to(self, *args, **kwargs):
        """Mirror Diffusers ``.to`` for the non-exported prompt encoder/scheduler path."""
        self.model.to(*args, **kwargs)
        return self

    @staticmethod
    def get_default_config_path() -> str:
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/ideogram_config.json")

    def export(
        self,
        export_dir: Optional[str] = None,
        max_sequence_length: int = 2048,
        height: int = 1024,
        width: int = 1024,
        transformer_attention_mask: bool = True,
        use_onnx_subfunctions: bool = False,
    ) -> None:
        self._ensure_qeff_text_encoder()
        self.transformer.set_attention_mask_enabled(transformer_attention_mask)
        self.unconditional_transformer.set_attention_mask_enabled(False)
        grid_h, grid_w = (
            height // (self.vae_scale_factor * self.patch_size),
            width // (self.vae_scale_factor * self.patch_size),
        )
        num_image_tokens = grid_h * grid_w
        module_seq_lens = {
            "transformer": max_sequence_length + num_image_tokens,
            "unconditional_transformer": num_image_tokens,
        }

        for module_name, module_obj in tqdm(self.modules.items(), desc="Exporting modules", unit="module"):
            if module_name == "vae_decoder":
                example_inputs, dynamic_axes, output_names = module_obj.get_onnx_params(
                    latent_height=grid_h * self.patch_size,
                    latent_width=grid_w * self.patch_size,
                )
            elif module_name == "text_encoder":
                example_inputs, dynamic_axes, output_names = module_obj.get_onnx_params(seq_len=max_sequence_length)
            elif module_name == "prompt_enhancer_head":
                example_inputs, dynamic_axes, output_names = module_obj.get_onnx_params(seq_len=max_sequence_length)
            elif module_name == "transformer":
                example_inputs, dynamic_axes, output_names = module_obj.get_onnx_params(
                    seq_len=module_seq_lens[module_name],
                    encoder_seq_len=max_sequence_length,
                )
            else:
                example_inputs, dynamic_axes, output_names = module_obj.get_onnx_params(
                    seq_len=module_seq_lens[module_name]
                )

            export_params = {
                "inputs": example_inputs,
                "output_names": output_names,
                "dynamic_axes": dynamic_axes,
                "export_dir": export_dir,
            }
            if use_onnx_subfunctions:
                export_params["use_onnx_subfunctions"] = True
            if module_obj.qpc_path is None:
                module_obj.export(**export_params)

    def compile(
        self,
        compile_config: Optional[str] = None,
        parallel: bool = False,
        height: int = 1024,
        width: int = 1024,
        max_sequence_length: int = 2048,
        transformer_attention_mask: bool = True,
        use_onnx_subfunctions: bool = False,
    ) -> None:
        self._ensure_qeff_text_encoder()
        self.transformer.set_attention_mask_enabled(transformer_attention_mask)
        self.unconditional_transformer.set_attention_mask_enabled(False)
        config_manager(self, config_source=compile_config, use_onnx_subfunctions=use_onnx_subfunctions)
        set_execute_params(self)

        if any(module.onnx_path is None for module in self.modules.values()):
            self.export(
                max_sequence_length=max_sequence_length,
                height=height,
                width=width,
                transformer_attention_mask=transformer_attention_mask,
                use_onnx_subfunctions=use_onnx_subfunctions,
            )

        grid_h, grid_w = (
            height // (self.vae_scale_factor * self.patch_size),
            width // (self.vae_scale_factor * self.patch_size),
        )
        num_image_tokens = grid_h * grid_w
        specialization_updates = {
            "text_encoder": {"seq_len": max_sequence_length},
            "transformer": {"seq_len": max_sequence_length + num_image_tokens, "encoder_seq_len": max_sequence_length},
            "unconditional_transformer": {"seq_len": num_image_tokens},
            "vae_decoder": {"latent_height": grid_h * self.patch_size, "latent_width": grid_w * self.patch_size},
        }
        if self.qeff_prompt_enhancer_head is not None:
            specialization_updates["prompt_enhancer_head"] = [
                {"seq_len": 1},
                {"seq_len": max_sequence_length},
            ]

        if parallel:
            compile_modules_parallel(self.modules, self.custom_config, specialization_updates)
        else:
            compile_modules_sequential(self.modules, self.custom_config, specialization_updates)

        if self.qeff_prompt_enhancer_head is not None:
            self.qeff_prompt_enhancer_head.patch_for_qaic_runtime()

    def _get_prompt_text_lengths(self, prompt: str | list[str], max_sequence_length: int) -> list[int]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        text_lengths = []
        for text_prompt in prompts:
            messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            text_len = int(tokens.shape[0])
            if text_len > max_sequence_length:
                raise ValueError(f"prompt has {text_len} tokens, exceeds max_sequence_length={max_sequence_length}")
            text_lengths.append(text_len)
        return text_lengths

    def _prepare_prompt_enhancer_head_for_qaic(
        self,
        compile_config: Optional[str],
        max_sequence_length: int,
        use_onnx_subfunctions: bool,
    ) -> None:
        if self.qeff_prompt_enhancer_head is None:
            return

        config_manager(self, config_source=compile_config, use_onnx_subfunctions=use_onnx_subfunctions)
        set_execute_params(self)

        module_name = "prompt_enhancer_head"
        module_obj = self.qeff_prompt_enhancer_head
        module_config = self.custom_config["modules"][module_name]
        specializations = module_config["specializations"].copy()
        compile_kwargs = module_config["compilation"].copy()
        if compile_kwargs.get("onnx_path") is None:
            compile_kwargs["onnx_path"] = module_obj.onnx_path

        if isinstance(specializations, list):
            for spec in specializations:
                if int(spec.get("seq_len", 1)) != 1:
                    spec["seq_len"] = max_sequence_length
        else:
            specializations.update({"seq_len": max_sequence_length})
            specializations = [specializations]
        specializations = [{**spec, "_graph_name": module_name} for spec in specializations]

        if module_obj.qpc_path is None:
            module_obj.compile(specializations=specializations, **compile_kwargs)
        module_obj.patch_for_qaic_runtime()

    def _encode_prompt_on_qaic(
        self,
        prompt: str | list[str],
        grid_h: int,
        grid_w: int,
        max_sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        batch_size = len(prompts)
        num_image_tokens = grid_h * grid_w

        token_ids = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
        text_position_ids = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
        text_lengths = []
        for batch_idx, text_prompt in enumerate(prompts):
            messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            text_len = int(tokens.shape[0])
            if text_len > max_sequence_length:
                raise ValueError(f"prompt has {text_len} tokens, exceeds max_sequence_length={max_sequence_length}")
            text_lengths.append(text_len)
            offset = max_sequence_length - text_len
            token_ids[batch_idx, offset:] = tokens
            attention_mask[batch_idx, offset:] = 1
            text_position_ids[batch_idx, offset:] = torch.arange(text_len)

        position_ids_4d = text_position_ids[None, ...].expand(4, text_position_ids.shape[0], -1)

        session = _ensure_qaic_session(self.qeff_text_encoder)
        feature_dim = self.qeff_text_encoder.model.output_features_dim
        session.set_buffers(
            {"text_features": np.empty((batch_size, max_sequence_length, feature_dim), dtype=np.float32)}
        )
        encode_start = time.perf_counter()
        text_features_np = session.run(
            {
                "input_ids": token_ids.numpy().astype(np.int64),
                "attention_mask": attention_mask.numpy().astype(np.int64),
                "position_ids": position_ids_4d.numpy().astype(np.int64),
            }
        )["text_features"]
        encode_perf = time.perf_counter() - encode_start
        _release_qaic_session(self.qeff_text_encoder)

        text_features = torch.from_numpy(text_features_np).to(device=device, dtype=torch.float32)

        position_ids, segment_ids, indicator = self.model._prepare_ids(
            text_lengths, grid_h, grid_w, max_sequence_length, device
        )

        return text_features, position_ids, segment_ids, indicator, encode_perf

    def __getattr__(self, name: str):
        return getattr(self.model, name)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        height: int = 1024,
        width: int = 1024,

        num_inference_steps: int = 48,
        guidance_scale: float | None = None,
        guidance_schedule: list[float] | torch.Tensor | None = (7.0,) * 45 + (3.0,) * 3,
        mu: float = 0.0,
        std: float = 1.5,
        prompt_upsampling: bool = False,
        prompt_upsampling_temperature: float = 1.0,
        max_sequence_length: int = 2048,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Optional[
            Callable[["QEffIdeogram4Pipeline", int, int, dict[str, Any]], dict[str, Any]]
        ] = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        custom_config_path: Optional[str] = None,
        parallel_compile: bool = False,
        use_onnx_subfunctions: bool = False,
    ) -> Ideogram4PipelineOutput | QEffPipelineOutput | tuple[Any]:
        self.model.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_schedule=guidance_schedule,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("`prompt` must be provided.")

        device = self.model._execution_device
        self.model._guidance_scale = guidance_scale
        self.model._attention_kwargs = attention_kwargs
        self.model._interrupt = False
        prompt_enhancer_perf = []

        if prompt_upsampling:
            self._prepare_prompt_enhancer_head_for_qaic(
                compile_config=custom_config_path,
                max_sequence_length=IDEOGRAM_QEFF_TEXT_SEQ_LEN,
                use_onnx_subfunctions=use_onnx_subfunctions,
            )
            if self.qeff_prompt_enhancer_head is not None:
                self.qeff_prompt_enhancer_head.runtime_perf = []
            prompt = self.model.upsample_prompt(
                prompt,
                height=height,
                width=width,
                temperature=prompt_upsampling_temperature,
                max_new_tokens=IDEOGRAM_QEFF_TEXT_SEQ_LEN,
                generator=generator,
                device=device,
            )
            if self.qeff_prompt_enhancer_head is not None:
                prompt_enhancer_perf = self.qeff_prompt_enhancer_head.runtime_perf.copy()
                _release_qaic_session(self.qeff_prompt_enhancer_head)

        text_lengths = self._get_prompt_text_lengths(prompt, IDEOGRAM_QEFF_TEXT_SEQ_LEN)
        transformer_attention_mask = any(text_len < IDEOGRAM_QEFF_TEXT_SEQ_LEN for text_len in text_lengths)

        self.compile(
            compile_config=custom_config_path,
            parallel=parallel_compile,
            height=height,
            width=width,
            max_sequence_length=IDEOGRAM_QEFF_TEXT_SEQ_LEN,
            transformer_attention_mask=transformer_attention_mask,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

        grid_h, grid_w = (
            height // (self.vae_scale_factor * self.patch_size),
            width // (self.vae_scale_factor * self.patch_size),
        )
        num_image_tokens = grid_h * grid_w

        llm_features, position_ids, segment_ids, indicator, encode_perf = self._encode_prompt_on_qaic(
            prompt=prompt,
            grid_h=grid_h,
            grid_w=grid_w,
            max_sequence_length=IDEOGRAM_QEFF_TEXT_SEQ_LEN,
            device=device,
        )

        llm_features = _expand_tensor_to_effective_batch(llm_features, batch_size, num_images_per_prompt)
        position_ids = _expand_tensor_to_effective_batch(position_ids, batch_size, num_images_per_prompt)
        segment_ids = _expand_tensor_to_effective_batch(segment_ids, batch_size, num_images_per_prompt)
        indicator = _expand_tensor_to_effective_batch(indicator, batch_size, num_images_per_prompt)

        effective_batch = batch_size * num_images_per_prompt
        neg_position_ids = position_ids[:, IDEOGRAM_QEFF_TEXT_SEQ_LEN:]
        neg_segment_ids = segment_ids[:, IDEOGRAM_QEFF_TEXT_SEQ_LEN:]
        neg_indicator = indicator[:, IDEOGRAM_QEFF_TEXT_SEQ_LEN:]

        # Precompute MRoPE cos/sin on host in plain fp32 PyTorch (`position_ids` is constant across the
        # denoising loop, so this only needs to happen once per generation, per transformer). This keeps the
        # wide-range `inv_freq @ position_ids` matmul -- Ideogram4's image position ids start at 65536
        # (`IMAGE_POSITION_OFFSET`), already beyond the fp16 max representable value (~65504) -- entirely off
        # the compiled AIC graph. Feeding raw `position_ids` into a graph compiled with
        # `convert_to_fp16`/`mxfp6_matmul` would silently overflow to `inf`/`NaN` on-device and corrupt every
        # image token's attention, producing a blank/garbage image. cos/sin are bounded to [-1, 1] and are
        # therefore safe to cast into the transformer's fp16/mxfp6 compute dtype. Each transformer uses its own
        # `rotary_emb` submodule (rather than sharing one computation) since the conditional/unconditional
        # transformers are independent modules, even though they are expected to share the same MRoPE config.
        rotary_emb_cos, rotary_emb_sin = self.transformer.model.rotary_emb(position_ids)
        rotary_emb_cos = rotary_emb_cos.to(torch.float32)
        rotary_emb_sin = rotary_emb_sin.to(torch.float32)
        neg_rotary_emb_cos, neg_rotary_emb_sin = self.unconditional_transformer.model.rotary_emb(neg_position_ids)
        neg_rotary_emb_cos = neg_rotary_emb_cos.to(torch.float32)
        neg_rotary_emb_sin = neg_rotary_emb_sin.to(torch.float32)

        schedule_mu = _resolution_aware_mu(height=height, width=width, base_mu=mu)
        sigmas = _logit_normal_sigmas(num_inference_steps, schedule_mu, std=std, device=device)
        self.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
        timesteps = self.scheduler.timesteps
        self.model._num_timesteps = len(timesteps)

        if guidance_scale is not None:
            guidance_schedule = [float(guidance_scale)] * num_inference_steps
        gw = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)

        latent_dim = self.transformer.model.config.in_channels
        latents = self.model.prepare_latents(
            batch_size=effective_batch,
            num_image_tokens=num_image_tokens,
            latent_dim=latent_dim,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        max_text_tokens = IDEOGRAM_QEFF_TEXT_SEQ_LEN
        llm_features = llm_features[:, :max_text_tokens].to(torch.float32)
        # These tensors are constant across every denoising step (only `latents`/`timestep` change each
        # iteration), so convert them to numpy once here instead of re-converting on every one of the
        # `num_inference_steps` iterations below. The text encoder QPC also pre-projects the huge raw Ideogram text
        # features once, so the conditional transformer no longer receives or re-projects a 2048 x 53248 tensor on
        # every denoising step.
        llm_features_np = llm_features.detach().cpu().numpy()
        rotary_emb_cos_np = rotary_emb_cos.detach().cpu().numpy().astype(np.float32)
        rotary_emb_sin_np = rotary_emb_sin.detach().cpu().numpy().astype(np.float32)
        neg_rotary_emb_cos_np = neg_rotary_emb_cos.detach().cpu().numpy().astype(np.float32)
        neg_rotary_emb_sin_np = neg_rotary_emb_sin.detach().cpu().numpy().astype(np.float32)
        segment_ids_np = segment_ids.detach().cpu().numpy().astype(np.int64)
        indicator_np = indicator.detach().cpu().numpy().astype(np.int64)
        neg_segment_ids_np = neg_segment_ids.detach().cpu().numpy().astype(np.int64)
        neg_indicator_np = neg_indicator.detach().cpu().numpy().astype(np.int64)

        # Auto-allocated QAIC sessions (the default here -- Ideogram never pins explicit `device_ids`) are
        # assumed to live on independent, runtime-managed device pools, so `transformers_share_devices` is
        # False whenever the host has enough free devices for both 16-device QPCs to be resident at once.
        # Only deactivate/activate one transformer around the other's call when they actually contend for the
        # same explicit device IDs; otherwise both QPCs stay resident and run concurrently below.
        transformers_share_devices = _modules_share_qaic_devices(self.transformer, self.unconditional_transformer)
        transformer_output_buffer = {
            "output": np.empty((effective_batch, max_text_tokens + num_image_tokens, latent_dim), dtype=np.float32)
        }
        unconditional_output_buffer = {
            "output": np.empty((effective_batch, num_image_tokens, latent_dim), dtype=np.float32)
        }
        transformer_static_inputs = {
            "encoder_hidden_states": llm_features_np,
            "rotary_emb_cos": rotary_emb_cos_np,
            "rotary_emb_sin": rotary_emb_sin_np,
            "indicator": indicator_np,
        }
        if getattr(self.transformer.model, "qeff_use_attention_mask", True):
            transformer_static_inputs["segment_ids"] = segment_ids_np
        unconditional_static_inputs = {
            "rotary_emb_cos": neg_rotary_emb_cos_np,
            "rotary_emb_sin": neg_rotary_emb_sin_np,
            "indicator": neg_indicator_np,
        }
        if getattr(self.unconditional_transformer.model, "qeff_use_attention_mask", True):
            unconditional_static_inputs["segment_ids"] = neg_segment_ids_np
        prepared_sessions = {}
        pos_hidden_states_np = np.empty(
            (effective_batch, max_text_tokens + num_image_tokens, latent_dim), dtype=np.float32
        )
        pos_hidden_states_np[:, :max_text_tokens, :] = 0.0
        neg_hidden_states_np = np.empty((effective_batch, num_image_tokens, latent_dim), dtype=np.float32)

        def _prepare_transformer_session(
            module: QEFFBaseModel,
            static_inputs: Dict[str, np.ndarray],
            output_buffer: Dict[str, np.ndarray],
        ) -> QAICInferenceSession:
            session = _ensure_qaic_session(module)
            if prepared_sessions.get(module.module_name) is not session:
                session.set_buffers({**static_inputs, **output_buffer})
                prepared_sessions[module.module_name] = session
            return session

        def _run_transformer_step(
            module: QEFFBaseModel,
            other_module: QEFFBaseModel,
            inputs: Dict[str, np.ndarray],
            static_inputs: Dict[str, np.ndarray],
            output_buffer: Dict[str, np.ndarray],
        ):
            """Run one transformer forward pass; only pin/unpin `other_module` when devices are shared."""
            if transformers_share_devices:
                _deactivate_qaic_session(other_module)
            session = _prepare_transformer_session(module, static_inputs, output_buffer)
            start = time.perf_counter()
            output = session.run(inputs)["output"]
            elapsed = time.perf_counter() - start
            if transformers_share_devices:
                _deactivate_qaic_session(module)
            return output, elapsed

        transformer_perf = []
        unconditional_perf = []
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        # Only used when the two transformers are on independent device pools, so their per-step forward
        # passes can be dispatched concurrently instead of running strictly one after the other.
        step_executor = None if transformers_share_devices else ThreadPoolExecutor(max_workers=2)
        if step_executor is not None:
            # Construct (and thus device-auto-allocate + activate) each `QAICInferenceSession` one at a time,
            # *before* any concurrent `.run()` calls are dispatched below. `QAICInferenceSession.__init__` claims
            # free devices and issues `Program.load()`/`activate()` against the QAIC runtime; doing that from two
            # threads at once (e.g. both transformers lazily initializing their sessions on the first loop
            # iteration) is a race -- both threads can simultaneously grab overlapping devices and fail with
            # "Failed to allocate GSM semaphore resource" / "Failed to create ExecObj", even when the host has
            # plenty of free devices in total. Sequential construction here avoids that race; only the already
            # -initialized sessions' `.run()` calls are executed concurrently inside the loop.
            _prepare_transformer_session(self.transformer, transformer_static_inputs, transformer_output_buffer)
            _prepare_transformer_session(
                self.unconditional_transformer,
                unconditional_static_inputs,
                unconditional_output_buffer,
            )
        try:
            with self.model.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.model.interrupt:
                        continue

                    t_model = 1.0 - (t.float() / num_train_timesteps)
                    t_model = t_model.expand(effective_batch).to(torch.float32)
                    t_model_np = t_model.detach().cpu().numpy()

                    latents_np = latents.detach().cpu().numpy()
                    np.copyto(pos_hidden_states_np[:, max_text_tokens:, :], latents_np)
                    np.copyto(neg_hidden_states_np, latents_np)
                    pos_inputs = {
                        "hidden_states": pos_hidden_states_np,
                        "timestep": t_model_np,
                    }
                    neg_inputs = {
                        "hidden_states": neg_hidden_states_np,
                        "timestep": t_model_np,
                    }

                    if step_executor is not None:
                        pos_future = step_executor.submit(
                            _run_transformer_step,
                            self.transformer,
                            self.unconditional_transformer,
                            pos_inputs,
                            transformer_static_inputs,
                            transformer_output_buffer,
                        )
                        neg_future = step_executor.submit(
                            _run_transformer_step,
                            self.unconditional_transformer,
                            self.transformer,
                            neg_inputs,
                            unconditional_static_inputs,
                            unconditional_output_buffer,
                        )
                        pos_out, pos_elapsed = pos_future.result()
                        neg_out, neg_elapsed = neg_future.result()
                    else:
                        pos_out, pos_elapsed = _run_transformer_step(
                            self.transformer,
                            self.unconditional_transformer,
                            pos_inputs,
                            transformer_static_inputs,
                            transformer_output_buffer,
                        )
                        neg_out, neg_elapsed = _run_transformer_step(
                            self.unconditional_transformer,
                            self.transformer,
                            neg_inputs,
                            unconditional_static_inputs,
                            unconditional_output_buffer,
                        )

                    transformer_perf.append(pos_elapsed)
                    unconditional_perf.append(neg_elapsed)
                    pos_v = torch.from_numpy(pos_out[:, max_text_tokens:]).to(device=device, dtype=torch.float32)
                    neg_v = torch.from_numpy(neg_out).to(device=device, dtype=torch.float32)

                    self.model._guidance_scale = guidance_schedule[i]
                    gw_i = gw[i]
                    v = gw_i * pos_v + (1.0 - gw_i) * neg_v
                    latents = self.scheduler.step(-v, t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)

                    progress_bar.update()
        finally:
            if step_executor is not None:
                step_executor.shutdown()

        if output_type == "latent":
            image = latents
            vae_perf = 0.0
        else:
            z = latents
            bn_mean = self.vae_decode.model.bn.running_mean.view(1, 1, -1).to(device=z.device, dtype=z.dtype)
            bn_std = torch.sqrt(
                self.vae_decode.model.bn.running_var + self.vae_decode.model.config.batch_norm_eps
            ).view(1, 1, -1)
            bn_std = bn_std.to(device=z.device, dtype=z.dtype)
            z = z * bn_std + bn_mean

            patch = self.patch_size
            ae_channels = z.shape[-1] // (patch * patch)
            z = z.view(effective_batch, grid_h, grid_w, patch, patch, ae_channels)
            z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
            z = z.view(effective_batch, ae_channels, grid_h * patch, grid_w * patch)

            _release_qaic_session(self.transformer)
            _release_qaic_session(self.unconditional_transformer)
            gc.collect()

            vae_session = _ensure_qaic_session(self.vae_decode)
            vae_session.set_buffers({"sample": np.empty((effective_batch, 3, height, width), dtype=np.float32)})
            start = time.perf_counter()
            decoded = vae_session.run({"latent_sample": z.detach().cpu().numpy().astype(np.float32)})["sample"]
            vae_perf = time.perf_counter() - start
            image = self.image_processor.postprocess(torch.from_numpy(decoded).float(), output_type=output_type)

        self.model.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        perf_metrics = []
        if prompt_enhancer_perf:
            perf_metrics.append(ModulePerf(module_name="prompt_enhancer_head", perf=prompt_enhancer_perf))
        perf_metrics.extend(
            [
                ModulePerf(module_name="text_encoder", perf=encode_perf),
                ModulePerf(module_name="transformer", perf=transformer_perf),
                ModulePerf(module_name="unconditional_transformer", perf=unconditional_perf),
                ModulePerf(module_name="vae_decoder", perf=vae_perf),
            ]
        )
        return QEffPipelineOutput(pipeline_module=perf_metrics, images=image)
