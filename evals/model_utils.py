"""Shared model-loading utilities for CortexGPT eval scripts."""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch

# Allow importing from the project root (train.py, model.py, data.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import (
    build_cortex_gpt, build_pythia, build_pythia_ccot, CortexConfig,
)


def load_checkpoint(
    checkpoint_path: str,
    model_name: str,
    memory_slots: Optional[int],
    dtype: torch.dtype,
    device: torch.device,
):
    """
    Load a CortexGPT / PythiaVanilla / PythiaCCoT checkpoint.

    The saved config's `arch` field selects which model class to build:
      "cortex" — CortexGPT (the default; also for checkpoints predating the
                 arch field, which are all CortexGPT variants).
      "pythia" — PythiaVanilla, the true non-recurrent transformer baseline.
      "ccot"   — PythiaCCoT, full-network continuous chain-of-thought.

    memory_slots=None (the default for eval CLIs) reads the value from the
    config saved in the checkpoint — passing an int overrides it (a mismatch
    will fail the strict state-dict load, which is the desired loud error).
    Ignored for the pythia / ccot archs (they carry no memory buffer).

    Returns (model, cfg).  cfg.mean_recurrence reflects the value saved in
    the checkpoint, so callers can use it as the default eval T.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config", {})
    arch = saved_cfg.get("arch", "cortex")

    if arch == "pythia":
        # from_scratch=True → build the architecture from config (no pretrained
        # download), then overlay the trained weights below.
        model, cfg = build_pythia(
            model_name = model_name, from_scratch = True,
            torch_dtype = dtype, device_map = "cpu",
        )
    elif arch == "ccot":
        model, cfg = build_pythia_ccot(
            model_name = model_name, from_scratch = True,
            torch_dtype = dtype, device_map = "cpu",
        )
    else:
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


# ---------------------------------------------------------------------------
# Shared generation / memory-priming helpers (mirrors cortex-finetune's
# evals/model_utils.py; CortexGPT interface — model.has_cross_state property,
# num_steps = [(0, T)]).
# ---------------------------------------------------------------------------

def _has_cross_state(model) -> bool:
    return bool(getattr(model, "has_cross_state", False))


@torch.no_grad()
def prime_cross_state(model, chunks, num_steps, passes_per_chunk=1):
    """Run priming chunks through the model, carrying M_cross across them.
    passes_per_chunk > 1 runs each chunk through the FULL model that many
    times (M_cross carried pass-to-pass) so the buffer gets multiple writes
    per chunk.  Returns the final buffer, or None for models without cross
    state — those see only the final prediction chunk (no-memory control)."""
    if not _has_cross_state(model) or not chunks:
        return None
    device = next(model.parameters()).device
    m_cross = None
    for chunk in chunks:
        chunk = chunk.to(device)
        for _ in range(max(passes_per_chunk, 1)):
            out = model(input_ids=chunk, num_steps=num_steps,
                        m_cross_in=m_cross, return_m_cross=True)
            m_cross = out.get("m_cross")
    return m_cross


@torch.no_grad()
def ccot_prime(model, input_ids, num_steps, passes, m_cross_init=None):
    """Mixed CCoT: `passes` silent full forward passes over the SAME tokens,
    each pass's M_cross write feeding the next pass's read — latent
    multi-pass 'thinking' before any token is generated.  m_cross_init seeds
    the first pass (e.g. a buffer primed on earlier context chunks)."""
    if passes <= 0 or not _has_cross_state(model):
        return m_cross_init
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    m_cross = m_cross_init
    for _ in range(passes):
        out = model(input_ids=input_ids, num_steps=num_steps,
                    m_cross_in=m_cross, return_m_cross=True)
        m_cross = out.get("m_cross")
    return m_cross


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, max_new_tokens, num_steps,
                    m_cross=None, stop_on_newline=False):
    """Greedy decoding by full re-forward each step (no KV cache).  An
    optional primed m_cross buffer is held fixed as read-only context for
    every step.  Returns the generated text."""
    device = next(model.parameters()).device
    generated = input_ids.to(device)
    prompt_len = generated.shape[1]
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        out = model(input_ids=generated, num_steps=num_steps,
                    m_cross_in=m_cross, return_m_cross=False)
        next_tok = out["logits"][0, -1].argmax(dim=-1).view(1, 1)
        generated = torch.cat([generated, next_tok], dim=1)
        if eos_id is not None and next_tok.item() == eos_id:
            break
        if stop_on_newline and "\n" in tokenizer.decode(generated[0, prompt_len:]):
            break
    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)

