# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Utility functions for creating device maps for pipeline parallelism.
"""

from typing import Dict, Optional

import numpy as np
import torch
from transformers import AutoConfig

from QEfficient.finetune.experimental.core.utils.dist_utils import get_local_rank
from QEfficient.utils._utils import get_num_layers_from_config


def get_device_map(
    model_name: str,
    device: str,
    enable_pp: bool = False,
    num_pp_stages: int = 1,
) -> Optional[Dict[str, int]]:
    """
    Returns device map for the given model based on PP and DDP configuration.

    Args:
        model_name: Name of the model to load configuration from.
        device: Device type (e.g., 'cuda', 'qaic').
        enable_pp: Whether pipeline parallelism is enabled.
        num_pp_stages: Number of pipeline stages.
    Returns:
        Dict: A dictionary mapping layer names to device IDs, or None if no PP.
    """
    torch_device = torch.device(device)
    num_available_devices = getattr(torch, torch_device.type).device_count()

    if enable_pp:
        if num_pp_stages < num_available_devices:
            device_map = custom_device_map(model_name, device, num_pp_stages)
        elif num_pp_stages == num_available_devices:
            device_map = "auto"
    else:
        device_map = None

    return device_map


def custom_device_map(model_name: str, device: str, num_pp_stages: int) -> Dict[str, int]:
    """
    Returns custom device map for model layers based on number of pipeline stages and process rank.

    Args:
        model_name: Name of the model to load configuration from.
        device: Device type (e.g., 'cuda', 'qaic').
        num_pp_stages: Number of pipeline stages.

    Returns:
        Dict: A dictionary mapping layer names to device IDs.

    Notes:
        - This device map structure is verified for llama models primarily.
        - For other architectures, you may need to adjust the layer naming conventions.

    Example:
        Example config for PP + DDP is provided below as it works for only PP as well.
        Configuration for meta-llama/Llama-3.2-1B
        Total devices: 4 (2x PP x 2x DDP)

        PP (Pipeline Parallelism): Each copy of the model is split into 2 stages
        DDP (Distributed Data Parallel): 2 model copies run in parallel

        |--------------------------------------------------------------------------|
        | Process Rank |   Assigned Device IDs  | Model Component                  |
        |--------------------------------------------------------------------------|
        | Rank 0       | 0                      | model.embed_tokens               |
        |              |                        | model.lm_head                    |
        |              |                        | model.layers.0 - model.layers.7  |
        |--------------------------------------------------------------------------|
        | Rank 0       | 1                      | model.norm                       |
        |              |                        | model.rotary_emb                 |
        |              |                        | model.layers.8 - model.layers.15 |
        |--------------------------------------------------------------------------|
        | Rank 1       | 2                      | model.embed_tokens               |
        |              |                        | model.lm_head                    |
        |              |                        | model.layers.0 - model.layers.7  |
        |--------------------------------------------------------------------------|
        | Rank 1       | 3                      | model.norm                       |
        |              |                        | model.rotary_emb                 |
        |              |                        | model.layers.8 - model.layers.15 |
        |--------------------------------------------------------------------------|
    """

    model_config = AutoConfig.from_pretrained(model_name)
    num_layers = get_num_layers_from_config(model_config)
    local_rank = get_local_rank()
    first_device = local_rank * num_pp_stages
    last_device = local_rank * num_pp_stages + (num_pp_stages - 1)

    # Handle tied embeddings
    if model_config.tie_word_embeddings:
        lm_head_device = first_device
    else:
        lm_head_device = last_device

    device_map = {
        "model.embed_tokens": first_device,
        "lm_head": lm_head_device,
        "model.norm": last_device,
        "model.rotary_emb": last_device,
    }

    # Calculate layers per stage
    n_layer_per_stage = np.ceil(num_layers / num_pp_stages)

    # Create device mapping for each stage
    pp_stage_ids = np.arange(num_pp_stages)
    pp_device_map = np.repeat(pp_stage_ids, n_layer_per_stage)

    # Assign each layer to a device
    for i in range(num_layers):
        device_map[f"model.layers.{i}"] = int(pp_device_map[i] + local_rank * num_pp_stages)

    return device_map


def validate_pp_config(
    enable_pp: bool,
    num_pp_stages: int,
    device: str,
    local_world_size: int = 1,
) -> None:
    """
    Validate pipeline parallelism configuration.

    Args:
        enable_pp: Whether pipeline parallelism is enabled.
        num_pp_stages: Number of pipeline stages.
        device: Device type (e.g., 'cuda', 'qaic').
        local_world_size: Number of processes per node for DDP.

    Raises:
        AssertionError: If configuration is invalid.
    """
    if enable_pp:
        assert num_pp_stages > 1, (
            f"For pipeline parallelism, num_pp_stages should be greater than 1. Got {num_pp_stages}"
        )

        # Validate device availability
        torch_device = torch.device(device)
        num_available_devices = getattr(torch, torch_device.type).device_count()

        assert local_world_size * num_pp_stages <= num_available_devices, (
            f"Number of devices required per node (LOCAL_WORLD_SIZE * num_pp_stages = "
            f"{local_world_size} * {num_pp_stages} = {local_world_size * num_pp_stages}) "
            f"should be <= locally available devices ({num_available_devices})."
        )
