# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Translate verl's QAT JSON configs into the HF-style quantization_config dict
that TRT-LLM's _torch model config loader (`load_hf_quant_config`) understands.

TRT-LLM dispatches on the top-level ``quant_method`` string. Recognized values:
``fp8``, ``mxfp4``, ``compressed-tensors``, ``nvfp4``. Anything else (including
verl's ``modelopt`` from the Megatron path, or ``compressed-tensors`` from the
FSDP path with ``format == "nvfp4-pack-quantized"`` that TRT-LLM's
``compressed-tensors`` branch only supports for 8-bit) falls through silently
and the engine builds bf16 linears -- the subsequent NVFP4 weight sync then
crashes with a shape mismatch.

This module normalizes both verl QAT JSON shapes into the TRT-LLM
``quant_method == "nvfp4"`` shape:

    {"quant_method": "nvfp4", "group_size": 16, "modules_to_not_convert": [...]}

See ``tensorrt_llm/_torch/model_config.py::load_hf_quant_config`` for the
authoritative TRT-LLM-side parser.
"""

from __future__ import annotations

import json
from typing import Any


def verl_qat_json_to_trtllm_nvfp4_config(qat_json_path: str) -> dict[str, Any]:
    """Read a verl QAT JSON and return a TRT-LLM-compatible NVFP4 config dict.

    Supports three input shapes:
      1. Megatron variant (recipe/qat/config/nvfp4_w4a16_megatron.json):
         ``quant_method == "modelopt"`` with top-level ``quant_algo == "NVFP4"``.
      2. FSDP variant (recipe/qat/config/nvfp4_w4a16.json):
         ``quant_method == "compressed-tensors"`` with
         ``format == "nvfp4-pack-quantized"``.
      3. Already TRT-LLM-shaped: ``quant_method == "nvfp4"`` -> passthrough.

    Group size is pulled from ``config_groups.group_0.weights.group_size``
    when present (both verl variants encode it there); falls back to 16.
    Excluded modules come from the top-level ``ignore`` list.
    """
    with open(qat_json_path) as f:
        raw = json.load(f)

    quant_method = (raw.get("quant_method") or "").lower()

    # Passthrough if already in TRT-LLM shape.
    if quant_method == "nvfp4":
        return raw

    # Detect NVFP4 intent across the two verl shapes.
    is_modelopt_nvfp4 = (
        quant_method == "modelopt" and str(raw.get("quant_algo", "")).upper() == "NVFP4"
    )
    is_compressed_nvfp4 = (
        quant_method == "compressed-tensors"
        and "nvfp4" in str(raw.get("format", "")).lower()
    )
    if not (is_modelopt_nvfp4 or is_compressed_nvfp4):
        raise ValueError(
            f"QAT JSON at {qat_json_path} is not a recognized NVFP4 verl config. "
            f"Got quant_method={raw.get('quant_method')!r}, "
            f"quant_algo={raw.get('quant_algo')!r}, "
            f"format={raw.get('format')!r}. "
            "Expected one of: modelopt+NVFP4 (Megatron), "
            "compressed-tensors+nvfp4-pack-quantized (FSDP), or quant_method=nvfp4."
        )

    # Pull group_size from config_groups.group_0.weights if present.
    weights_cfg = (
        raw.get("config_groups", {}).get("group_0", {}).get("weights", {}) or {}
    )
    group_size = weights_cfg.get("group_size", 16)
    if group_size != 16:
        raise ValueError(
            f"TRT-LLM NVFP4 requires group_size=16, got {group_size} in {qat_json_path}"
        )

    ignore = list(raw.get("ignore", []) or [])

    return {
        "quant_method": "nvfp4",
        "group_size": group_size,
        "modules_to_not_convert": ignore,
    }
