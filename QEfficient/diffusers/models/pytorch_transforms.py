# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from diffusers.models.autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWan,
    WanDecoder3d,
    WanEncoder3d,
    WanResample,
    WanResidualBlock,
)
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle, RMSNorm
import torch
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxAttnProcessor,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    FluxTransformerBlock,
)
from diffusers.models.transformers.transformer_ideogram4 import (
    Ideogram4Attention,
    Ideogram4AttnProcessor,
    Ideogram4MRoPE,
    Ideogram4Transformer2DModel,
    Ideogram4TransformerBlock,
)
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor, WanTransformer3DModel
from torch import nn
from transformers.models.clip.modeling_clip import CLIPTextTransformer

from QEfficient.base.pytorch_transforms import ModuleMappingTransform, PytorchTransform
from QEfficient.customop.rms_norm import CustomRMSNormAIC
from QEfficient.diffusers.models.autoencoders.autoencoder_kl_wan import (
    QEffAutoencoderKLWan,
    QEffWanDecoder3d,
    QEffWanEncoder3d,
    QEffWanResample,
    QEffWanResidualBlock,
)
from QEfficient.diffusers.models.normalization import (
    QEffAdaLayerNormContinuous,
    QEffAdaLayerNormZero,
    QEffAdaLayerNormZeroSingle,
)
from QEfficient.diffusers.models.transformers.transformer_flux import (
    QEffFluxAttention,
    QEffFluxAttnProcessor,
    QEffFluxSingleTransformerBlock,
    QEffFluxTransformer2DModel,
    QEffFluxTransformerBlock,
)
from QEfficient.diffusers.models.transformers.transformer_ideogram import (
    QEffIdeogram4Attention,
    QEffIdeogram4AttnProcessor,
    QEffIdeogram4MRoPE,
    QEffIdeogram4Transformer2DModel,
    QEffIdeogram4TransformerBlock,
)
from QEfficient.diffusers.models.transformers.transformer_wan import (
    QEffWanAttention,
    QEffWanAttnProcessor,
    QEffWanTransformer3DModel,
)
from QEfficient.transformers.models.clip.modeling_clip import QEffCLIPTextTransformer
from QEfficient.utils.logging_utils import logger


class Bnb4BitLinearToLinearTransform(PytorchTransform):
    """
    Dequantize bitsandbytes 4-bit Linear modules (including NF4) to regular ``nn.Linear``.

    Diffusers can load models such as NF4 checkpoints with bitsandbytes modules
    (for example ``bitsandbytes.nn.Linear4bit``). QAIC does not consume these
    modules directly, so this transform materializes their weights as floating
    point tensors before ONNX export. The exported ONNX then contains normal
    Linear/MatMul weights and can be compiled in fp32/fp16 like other QEfficient
    Diffusers modules.
    """

    _bnb_4bit_class_names = {"Linear4bit", "LinearNF4", "LinearFP4"}

    @classmethod
    def _is_bnb_4bit_linear(cls, module: nn.Module) -> bool:
        return module.__class__.__name__ in cls._bnb_4bit_class_names and hasattr(module, "weight")

    @classmethod
    def _dequantize_weight(cls, original_module: nn.Module) -> torch.Tensor:
        weight = original_module.weight
        quant_state = getattr(weight, "quant_state", None)

        if quant_state is not None:
            try:
                import bitsandbytes.functional as bnb_functional
            except ImportError as exc:
                raise ImportError("bitsandbytes is required to dequantize 4-bit Diffusers modules") from exc

            dequant_weight = (
                bnb_functional.dequantize_4bit(weight.data, quant_state=quant_state).detach().to(torch.float32).cpu()
            )
        elif hasattr(weight, "dequantize"):
            dequant_weight = weight.dequantize().detach().to(torch.float32).cpu()
        else:
            raise TypeError(f"Cannot dequantize bitsandbytes weight of type {type(weight)}: missing quant_state")

        out_features = getattr(original_module, "out_features", None)
        in_features = getattr(original_module, "in_features", None)
        if out_features is not None and in_features is not None and dequant_weight.shape != (out_features, in_features):
            if dequant_weight.numel() != out_features * in_features:
                raise ValueError(
                    "Dequantized bitsandbytes weight has unexpected size: "
                    f"got {tuple(dequant_weight.shape)} with {dequant_weight.numel()} elements, "
                    f"expected ({out_features}, {in_features})"
                )
            dequant_weight = dequant_weight.reshape(out_features, in_features).contiguous()

        return dequant_weight

    @classmethod
    def mutate(cls, original_module: nn.Module) -> nn.Linear:
        dequant_weight = cls._dequantize_weight(original_module)
        out_features = getattr(original_module, "out_features", dequant_weight.shape[0])
        in_features = getattr(original_module, "in_features", dequant_weight.shape[1])
        bias = getattr(original_module, "bias", None)

        linear = nn.Linear(in_features, out_features, bias=bias is not None, device="cpu", dtype=torch.float32)
        linear.weight = nn.Parameter(dequant_weight, requires_grad=False)
        if bias is not None:
            linear.bias = nn.Parameter(bias.detach().to(torch.float32).cpu(), requires_grad=False)
        return linear

    @classmethod
    def apply(cls, model: nn.Module):
        transformed = False

        for name, module in model.named_children():
            if cls._is_bnb_4bit_linear(module):
                setattr(model, name, cls.mutate(module))
                transformed = True
            else:
                _, child_transformed = cls.apply(module)
                transformed = transformed or child_transformed

        if cls._is_bnb_4bit_linear(model):
            model = cls.mutate(model)
            transformed = True

        if transformed:
            logger.info("Dequantized bitsandbytes 4-bit Linear modules to torch.nn.Linear")
        return model, transformed


class CustomOpsTransform(ModuleMappingTransform):
    _module_mapping = {
        RMSNorm: CustomRMSNormAIC,
        nn.RMSNorm: CustomRMSNormAIC,  #  for torch.nn.RMSNorm
    }


class CLIPTextTransform(ModuleMappingTransform):
    _module_mapping = {
        CLIPTextTransformer: QEffCLIPTextTransformer,
    }


class AttentionTransform(ModuleMappingTransform):
    _module_mapping = {
        FluxSingleTransformerBlock: QEffFluxSingleTransformerBlock,
        FluxTransformerBlock: QEffFluxTransformerBlock,
        FluxTransformer2DModel: QEffFluxTransformer2DModel,
        FluxAttention: QEffFluxAttention,
        FluxAttnProcessor: QEffFluxAttnProcessor,
        WanAttnProcessor: QEffWanAttnProcessor,
        WanAttention: QEffWanAttention,
        WanTransformer3DModel: QEffWanTransformer3DModel,
        Ideogram4TransformerBlock: QEffIdeogram4TransformerBlock,
        Ideogram4Transformer2DModel: QEffIdeogram4Transformer2DModel,
        Ideogram4MRoPE: QEffIdeogram4MRoPE,
        Ideogram4Attention: QEffIdeogram4Attention,
        Ideogram4AttnProcessor: QEffIdeogram4AttnProcessor,
        AutoencoderKLWan: QEffAutoencoderKLWan,
        WanDecoder3d: QEffWanDecoder3d,
        WanEncoder3d: QEffWanEncoder3d,
        WanResidualBlock: QEffWanResidualBlock,
        WanResample: QEffWanResample,
    }


class NormalizationTransform(ModuleMappingTransform):
    _module_mapping = {
        AdaLayerNormZero: QEffAdaLayerNormZero,
        AdaLayerNormZeroSingle: QEffAdaLayerNormZeroSingle,
        AdaLayerNormContinuous: QEffAdaLayerNormContinuous,
    }
