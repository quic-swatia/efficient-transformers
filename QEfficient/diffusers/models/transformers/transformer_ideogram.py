# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
import math
import os
from typing import Type, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_ideogram4 import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    Ideogram4Attention,
    Ideogram4AttnProcessor,
    Ideogram4MRoPE,
    Ideogram4Transformer2DModel,
    Ideogram4TransformerBlock,
)
from diffusers.utils import apply_lora_scale

from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention, get_attention_blocking_config


_IDEOGRAM_DEFAULT_ATTENTION_BLOCK_SIZE = 1024
_IDEOGRAM_DEFAULT_KV_BLOCK_SIZE = 256


def _get_ideogram_attention_blocking_config(
    seq_len: int,
    num_heads: int,
    has_attention_mask: bool,
) -> tuple[str, int | None, int | None, int | None]:
    blocking_mode, head_block_size, num_kv_blocks, num_q_blocks = get_attention_blocking_config()
    del has_attention_mask
    if os.environ.get("ATTENTION_BLOCKING_MODE") is not None or seq_len < 1024:
        return blocking_mode, head_block_size, num_kv_blocks, num_q_blocks

    return (
        "qkv",
        head_block_size or num_heads,
        num_kv_blocks or math.ceil(seq_len / _IDEOGRAM_DEFAULT_KV_BLOCK_SIZE),
        num_q_blocks or math.ceil(seq_len / _IDEOGRAM_DEFAULT_ATTENTION_BLOCK_SIZE),
    )


def qeff_apply_rotary_emb(x: torch.Tensor, freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    Ideogram4 applies MRoPE in (B, L, H, D) layout, where cos/sin are shaped (B, L, D) and broadcast over
    the head axis. This QEff-friendly implementation avoids the cat-based rotate-half path and mirrors the
    Flux replacement style.
    """
    cos, sin = freqs_cis  # [B, S, D]
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    cos, sin = cos.to(x.device), sin.to(x.device)
    B, S, H, D = x.shape
    x_first_half, x_second_half = x.reshape(B, S, H, 2, D // 2).unbind(-2)
    x_rotated = torch.stack([-x_second_half, x_first_half], dim=-2).flatten(3)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


class QEffIdeogram4AttnProcessor(Ideogram4AttnProcessor):
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "QEffIdeogram4Attention",
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = qeff_apply_rotary_emb(query, image_rotary_emb)
        key = qeff_apply_rotary_emb(key, image_rotary_emb)

        has_attention_mask = attention_mask is not None
        blocking_mode, head_block_size, num_kv_blocks, num_q_blocks = _get_ideogram_attention_blocking_config(
            seq_len=query.shape[1],
            num_heads=attn.num_heads,
            has_attention_mask=has_attention_mask,
        )
        hidden_states = compute_blocked_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            blocking_mode=blocking_mode,
            head_block_size=head_block_size,
            num_kv_blocks=num_kv_blocks,
            num_q_blocks=num_q_blocks,
            attention_mask=attention_mask,
            # Ideogram4 uses a block-diagonal segment mask for self-attention. Keep it in the blocked
            # self-attention kernels instead of forcing the generic full-matrix cross-attention fallback;
            # otherwise the conditional transformer ignores ATTENTION_BLOCKING_MODE whenever text padding
            # makes a mask necessary.
            is_cross_attention=False,
        )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)
        return attn.to_out[0](hidden_states)


class QEffIdeogram4Attention(Ideogram4Attention):
    def __qeff_init__(self):
        self.processor = QEffIdeogram4AttnProcessor()


class QEffIdeogram4MRoPE(Ideogram4MRoPE):
    """ONNX-friendly Ideogram4 MRoPE.

    Upstream Ideogram4MRoPE builds the final multi-axis frequency tensor by cloning the temporal
    frequencies and then performing advanced-index in-place assignment for the H/W axes. That exports
    to a Scatter/Expand-heavy ONNX subgraph that can contain malformed empty inputs for qaic-compile.

    This replacement keeps the exact same half-duplicated Ideogram RoPE layout, but selects H/W lanes
    with static boolean masks and torch.where, avoiding in-place indexed assignment.
    """

    def __qeff_init__(self):
        inv_freq_size = self.inv_freq.shape[0]
        h_mask = torch.zeros(inv_freq_size, dtype=torch.bool)
        w_mask = torch.zeros(inv_freq_size, dtype=torch.bool)

        h_length = self.mrope_section[1] * 3
        w_length = self.mrope_section[2] * 3
        h_mask[1:h_length:3] = True
        w_mask[2:w_length:3] = True

        self.register_buffer("qeff_h_axis_mask", h_mask.view(1, 1, inv_freq_size), persistent=False)
        self.register_buffer("qeff_w_axis_mask", w_mask.view(1, 1, inv_freq_size), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (B, L, 3), with axes ordered as (t, h, w).
        pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(3, position_ids.shape[0], -1, 1)

        # Keep the high-offset image positions in fp32, matching upstream Ideogram4MRoPE.
        with torch.autocast(device_type=position_ids.device.type, enabled=False):
            freqs = inv_freq @ pos.unsqueeze(2)
        freqs = freqs.transpose(2, 3)  # (3, B, L, inv_freq_size)

        freqs_t = freqs[0]
        freqs_t = torch.where(self.qeff_h_axis_mask, freqs[1], freqs_t)
        freqs_t = torch.where(self.qeff_w_axis_mask, freqs[2], freqs_t)

        # Ideogram's rotate-half implementation expects duplicated halves: [freqs, freqs].
        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos().float(), emb.sin().float()


class QEffIdeogram4Transformer2DModel(Ideogram4Transformer2DModel):
    """QEff Ideogram4 transformer with MRoPE hoisted out of the exported/compiled graph.

    Upstream ``Ideogram4Transformer2DModel.forward`` calls ``self.rotary_emb(position_ids)`` internally.
    Ideogram4's image position ids start at ``IMAGE_POSITION_OFFSET`` (65536), which already exceeds the
    fp16 max representable value (~65504) before any grid offset is even added. When this graph is compiled
    with ``convert_to_fp16``/``mxfp6_matmul`` (as the default Ideogram compiler config does, for the matmul
    performance win elsewhere in the transformer), the ``inv_freq @ position_ids`` matmul silently overflows
    to `inf`/`NaN` on-device, poisoning every image token's Q/K after RoPE and producing blank/garbage images
    -- with no PyTorch-side ``torch.autocast(enabled=False)`` guard able to help, since that only protects
    the ONNX export/trace step, not the compiler's post-export fp16/mxfp6 conversion.

    This override instead takes precomputed ``rotary_emb_cos``/``rotary_emb_sin`` tensors (bounded to
    ``[-1, 1]``, safe for fp16/mxfp6) as direct forward inputs. The caller (see
    ``QEffIdeogram4Pipeline.__call__``) computes them once on host in plain fp32 PyTorch via
    ``QEffIdeogram4MRoPE`` (since ``position_ids`` is constant across the denoising loop, this is also a
    perf win: MRoPE is computed once per generation instead of once per step per transformer).
    """

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffIdeogram4TransformerBlock}

    def __qeff_init__(self):
        self.qeff_encoder_hidden_states_projected = True
        self.qeff_use_attention_mask = True

    def qeff_set_attention_mask_enabled(self, enabled: bool) -> None:
        self.qeff_use_attention_mask = enabled

    def qeff_project_encoder_hidden_states(
        self,
        encoder_hidden_states: torch.Tensor,
        indicator: torch.Tensor,
    ) -> torch.Tensor:
        indicator = indicator[:, : encoder_hidden_states.shape[1]]
        encoder_mask = (indicator == LLM_TOKEN_INDICATOR).to(encoder_hidden_states.dtype).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states * encoder_mask
        encoder_hidden_states = self.llm_cond_norm(encoder_hidden_states)
        encoder_hidden_states = self.llm_cond_proj(encoder_hidden_states)
        return encoder_hidden_states * encoder_mask

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        rotary_emb_cos: torch.Tensor,
        rotary_emb_sin: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
        indicator: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        r"""
        Predict the flow-matching velocity for the image-token positions of the packed sequence.

        Identical to upstream ``Ideogram4Transformer2DModel.forward``, except ``position_ids`` is replaced
        by precomputed ``rotary_emb_cos``/``rotary_emb_sin`` (each of shape
        ``(batch_size, sequence_length, head_dim)``), so this graph never computes MRoPE from raw position
        ids and stays numerically safe under fp16/mxfp6 compilation. When
        ``qeff_encoder_hidden_states_projected`` is enabled, ``encoder_hidden_states`` is expected to already be
        normalized/projected by ``llm_cond_norm`` and ``llm_cond_proj``.
        """
        batch_size, seq_len, in_channels = hidden_states.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected last dim {self.in_channels}, got {in_channels}.")
        if indicator is None:
            raise ValueError("`indicator` must be provided for Ideogram transformer forward.")

        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(hidden_states.dtype).unsqueeze(-1)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(hidden_states.dtype).unsqueeze(-1)

        hidden_states = hidden_states * output_image_mask

        hidden_states = self.input_proj(hidden_states) * output_image_mask

        # Keep shape (B, 1, ...) when t is per-sample so downstream adaln projections do not pay for L
        # identical copies.
        t_cond = self.t_embedding(timestep)
        if timestep.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))

        if encoder_hidden_states is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_mask = llm_token_mask[:, :encoder_seq_len]
            if getattr(self, "qeff_encoder_hidden_states_projected", False):
                expected_dim = self.llm_cond_proj.out_features
                if encoder_hidden_states.shape[-1] != expected_dim:
                    raise ValueError(
                        f"Expected pre-projected encoder hidden size {expected_dim}, "
                        f"got {encoder_hidden_states.shape[-1]}."
                    )
                encoder_hidden_states = encoder_hidden_states * encoder_mask
            else:
                expected_dim = self.llm_cond_proj.in_features
                if encoder_hidden_states.shape[-1] != expected_dim:
                    raise ValueError(
                        f"Expected raw encoder hidden size {expected_dim}, got {encoder_hidden_states.shape[-1]}."
                    )
                encoder_hidden_states = self.qeff_project_encoder_hidden_states(encoder_hidden_states, indicator)
            if encoder_hidden_states.shape[1] != seq_len:
                image_padding = torch.zeros(
                    batch_size,
                    seq_len - encoder_hidden_states.shape[1],
                    encoder_hidden_states.shape[-1],
                    dtype=encoder_hidden_states.dtype,
                    device=encoder_hidden_states.device,
                )
                encoder_hidden_states = torch.cat([encoder_hidden_states, image_padding], dim=1)
            hidden_states = hidden_states + (encoder_hidden_states * llm_token_mask)

        image_indicator_embedding = self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))
        hidden_states = hidden_states + image_indicator_embedding

        # Precomputed on host (see `QEffIdeogram4Pipeline.__call__`); bounded to [-1, 1] and therefore
        # safe to cast into the transformer's fp16/mxfp6 compute dtype, unlike raw `position_ids`.
        cos = rotary_emb_cos.to(hidden_states.dtype)
        sin = rotary_emb_sin.to(hidden_states.dtype)
        image_rotary_emb = (cos, sin)

        # Block-diagonal mask from segment ids: tokens only attend within their segment. For image-only and
        # no-padding single-prompt exports this mask is all-true, so omit it to avoid a large per-layer `Where` over
        # the attention scores.
        if self.qeff_use_attention_mask:
            if segment_ids is None:
                raise ValueError("`segment_ids` must be provided when Ideogram attention masking is enabled.")
            attention_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)
        else:
            attention_mask = None

        for block in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, attention_mask, image_rotary_emb, adaln_input
                )
            else:
                hidden_states = block(hidden_states, attention_mask, image_rotary_emb, adaln_input)

        output = self.final_layer(hidden_states, conditioning=adaln_input)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


class QEffIdeogram4TransformerBlock(Ideogram4TransformerBlock):
    pass


__all__ = [
    "QEffIdeogram4Attention",
    "QEffIdeogram4AttnProcessor",
    "QEffIdeogram4MRoPE",
    "QEffIdeogram4Transformer2DModel",
    "QEffIdeogram4TransformerBlock",
    "qeff_apply_rotary_emb",
]
