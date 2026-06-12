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
    memory_slots: Optional[int],
    dtype: torch.dtype,
    device: torch.device,
):
    """
    Load a CortexGPT checkpoint.

    memory_slots=None (the default for eval CLIs) reads the value from the
    config saved in the checkpoint — passing an int overrides it (a mismatch
    will fail the strict state-dict load, which is the desired loud error).

    Returns (model, cfg).  cfg.mean_recurrence reflects the value saved in
    the checkpoint, so callers can use it as the default eval T.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config", {})
    if memory_slots is None:
        memory_slots = saved_cfg.get("memory_slots", 0)
    model, cfg = build_cortex_gpt(
        model_name         = model_name,
        memory_slots       = memory_slots,
        memory_slots_iter  = saved_cfg.get("memory_slots_iter", 0),
        torch_dtype        = dtype,
        device_map         = "cpu",
        scalable_init      = saved_cfg.get("scalable_init", True),
        h0_init            = saved_cfg.get("h0_init", "random"),
        prelude_norm       = saved_cfg.get("prelude_norm", True),
        # Old checkpoints predate ccot_direct and default to False, so they
        # build without the module and still load with strict=True.
        ccot_direct        = saved_cfg.get("ccot_direct", False),
    )
    if "mean_recurrence" in saved_cfg:
        cfg.mean_recurrence = saved_cfg["mean_recurrence"]
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, cfg
