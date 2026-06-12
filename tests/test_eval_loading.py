"""
Eval checkpoint loading (evals/model_utils.load_checkpoint).

The eval CLIs default --memory_slots to None; load_checkpoint must then read
the architecture (memory_slots, ccot_direct, mean_recurrence, ...) from the
config dict saved in the checkpoint instead of trusting the command line.
An explicit CLI override still wins — and a mismatched override must fail the
strict state-dict load loudly rather than silently evaluating a different
architecture.

build_cortex_gpt is monkeypatched onto the fake-base constructor so no Pythia
download is needed; what matters here is which kwargs load_checkpoint passes.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "evals"))
sys.path.insert(0, os.path.dirname(__file__))

import model_utils
from model import CortexGPT, CortexConfig
from test_components import _make_fake_base, H, K


def _build_tiny(**kw) -> tuple[CortexGPT, CortexConfig]:
    """Stand-in for build_cortex_gpt: same kwargs surface, fake 4-layer base."""
    base = _make_fake_base(H, 4)
    cfg  = CortexConfig(
        n_pre=1, n_loop=2, n_coda=1,
        hidden_size=H,
        memory_slots=kw["memory_slots"],
        memory_slots_iter=kw.get("memory_slots_iter", 0),
        h0_init=kw.get("h0_init", "random"),
        prelude_norm=kw.get("prelude_norm", True),
        ccot_direct=kw.get("ccot_direct", False),
    )
    return CortexGPT(base, cfg), cfg


def _save_ckpt(tmp_path, config: dict, **build_kw) -> str:
    model, _ = _build_tiny(memory_slots=config.get("memory_slots", 0), **build_kw)
    path = str(tmp_path / "ckpt.pt")
    torch.save({"model": model.state_dict(), "config": config}, path)
    return path


@pytest.fixture
def patched_build(monkeypatch):
    captured: dict = {}

    def fake_build(**kw):
        captured.update(kw)
        return _build_tiny(**kw)

    monkeypatch.setattr(model_utils, "build_cortex_gpt", fake_build)
    return captured


class TestLoadCheckpoint:

    def test_k_read_from_checkpoint_when_cli_none(self, tmp_path, patched_build):
        """memory_slots=None (eval CLI default) → K comes from the saved config."""
        path = _save_ckpt(tmp_path, {"memory_slots": K, "mean_recurrence": 7})
        model, cfg = model_utils.load_checkpoint(
            path, "fake-model", None, torch.float32, torch.device("cpu"))
        assert patched_build["memory_slots"] == K, \
            "load_checkpoint did not read memory_slots from the saved config"
        assert model.m_cross is not None
        assert cfg.mean_recurrence == 7, \
            "mean_recurrence not restored from the saved config"

    def test_ccot_direct_restored_from_checkpoint(self, tmp_path, patched_build):
        """K=0 Cortex checkpoints restore ccot_direct from the saved config."""
        path = _save_ckpt(tmp_path, {"memory_slots": 0, "ccot_direct": True},
                          ccot_direct=True)
        model, _ = model_utils.load_checkpoint(
            path, "fake-model", None, torch.float32, torch.device("cpu"))
        assert patched_build["ccot_direct"] is True
        assert model.ccot_direct is not None

    def test_old_checkpoint_without_config_defaults_to_k0(self, tmp_path, patched_build):
        """Checkpoints predating the saved config dict load as K=0."""
        model, _ = _build_tiny(memory_slots=0)
        path = str(tmp_path / "old.pt")
        torch.save({"model": model.state_dict()}, path)
        loaded, _ = model_utils.load_checkpoint(
            path, "fake-model", None, torch.float32, torch.device("cpu"))
        assert patched_build["memory_slots"] == 0
        assert loaded.m_cross is None

    def test_explicit_cli_override_wins(self, tmp_path, patched_build):
        """An explicit --memory_slots value overrides the saved config."""
        path = _save_ckpt(tmp_path, {"memory_slots": K})
        with pytest.raises(RuntimeError):
            # K=0 build vs K=4 weights → strict load must fail loudly,
            # not silently evaluate a different architecture.
            model_utils.load_checkpoint(
                path, "fake-model", 0, torch.float32, torch.device("cpu"))
        assert patched_build["memory_slots"] == 0, \
            "explicit CLI memory_slots was ignored"

    def test_matching_cli_value_loads(self, tmp_path, patched_build):
        """Passing the same K as the checkpoint loads cleanly."""
        path = _save_ckpt(tmp_path, {"memory_slots": K})
        model, _ = model_utils.load_checkpoint(
            path, "fake-model", K, torch.float32, torch.device("cpu"))
        assert model.m_cross is not None
