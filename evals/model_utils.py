"""Shared model-loading utilities for CortexGPT eval scripts."""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch

# Allow importing from the project root (train.py, model.py, data.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import build_cortex_gpt, CortexConfig


def load_checkpoint(
    checkpoint_path: str,
    model_name: str,
    memory_slots: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    Load a CortexGPT checkpoint.

    Returns (model, cfg).  cfg.mean_recurrence reflects the value saved in
    the checkpoint, so callers can use it as the default eval T.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config", {})
    model, cfg = build_cortex_gpt(
        model_name         = model_name,
        memory_slots       = memory_slots,
        memory_slots_iter  = saved_cfg.get("memory_slots_iter", 0),
        torch_dtype        = dtype,
        device_map         = "cpu",
        scalable_init      = saved_cfg.get("scalable_init", True),
        h0_init            = saved_cfg.get("h0_init", "random"),
        prelude_norm       = saved_cfg.get("prelude_norm", True),
    )
    if "mean_recurrence" in saved_cfg:
        cfg.mean_recurrence = saved_cfg["mean_recurrence"]
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, cfg
