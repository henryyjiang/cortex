"""
Tests for the two non-CortexGPT architectures:

  PythiaVanilla — the TRUE non-recurrent GPTNeoX baseline (replaces the old
                  recurrent-arch-at-T=1 "vanilla" mode).
  PythiaCCoT    — Coconut-style continuous chain-of-thought over the FULL
                  transformer (last hidden state added directly to the next
                  pass's input embeddings).

Forward-correctness tests use a real tiny GPTNeoXForCausalLM (no network: built
from a hand-written GPTNeoXConfig) so they exercise the actual HF layer API
(position_embeddings tuple, plain-tensor layer return) that the manual stack
relies on.  The eval-loader dispatch test monkeypatches the builders.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "evals"))
sys.path.insert(0, os.path.dirname(__file__))

from model import (
    PythiaVanilla, PythiaCCoT, CortexConfig,
)

V, H, B, S = 200, 32, 2, 12


def _tiny_base():
    """A real (random-init) tiny GPTNeoXForCausalLM — no download."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    cfg = GPTNeoXConfig(
        vocab_size=V, hidden_size=H, num_hidden_layers=4,
        num_attention_heads=4, intermediate_size=4 * H,
        max_position_embeddings=64,
    )
    return GPTNeoXForCausalLM(cfg)


def _ids():
    g = torch.Generator().manual_seed(0)
    ids    = torch.randint(0, V, (B, S), generator=g)
    labels = torch.randint(0, V, (B, S), generator=g)
    return ids, labels


# ---------------------------------------------------------------------------
# PythiaVanilla
# ---------------------------------------------------------------------------

class TestPythiaVanilla:

    def test_forward_shapes_and_loss(self):
        m = PythiaVanilla(_tiny_base(), CortexConfig(arch="pythia", hidden_size=H))
        ids, labels = _ids()
        out = m(input_ids=ids, labels=labels)
        assert tuple(out["logits"].shape) == (B, S, V)
        assert torch.isfinite(out["loss"])
        assert out["loss"].dtype == torch.float32

    def test_has_no_cross_state(self):
        m = PythiaVanilla(_tiny_base(), CortexConfig(arch="pythia", hidden_size=H))
        assert m.has_cross_state is False
        # return_m_cross is honoured (None) so the train loop's optional read is safe.
        out = m(input_ids=_ids()[0], return_m_cross=True)
        assert out["m_cross"] is None

    def test_ignores_recurrence_kwargs(self):
        """num_steps / m_cross_in / eos_mask must not change the output."""
        m = PythiaVanilla(_tiny_base(), CortexConfig(arch="pythia", hidden_size=H))
        ids, _ = _ids()
        a = m(input_ids=ids)["logits"]
        b = m(input_ids=ids, num_steps=torch.tensor([3, 5]),
              m_cross_in=torch.randn(B, 4, H),
              eos_mask=torch.ones(B, S, dtype=torch.bool))["logits"]
        assert torch.allclose(a, b)

    def test_gradients_flow(self):
        m = PythiaVanilla(_tiny_base(), CortexConfig(arch="pythia", hidden_size=H))
        ids, labels = _ids()
        m(input_ids=ids, labels=labels)["loss"].backward()
        grads = [p.grad for p in m.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)

    def test_no_internal_label_shift(self):
        """Loss must match an explicit no-shift cross-entropy (data.py already
        shifts labels), NOT the HF-internal shifted loss."""
        import torch.nn.functional as F
        m = PythiaVanilla(_tiny_base(), CortexConfig(arch="pythia", hidden_size=H))
        ids, labels = _ids()
        out = m(input_ids=ids, labels=labels)
        manual = F.cross_entropy(
            out["logits"].reshape(-1, V), labels.reshape(-1), ignore_index=-100)
        assert torch.allclose(out["loss"], manual)


# ---------------------------------------------------------------------------
# PythiaCCoT
# ---------------------------------------------------------------------------

class TestPythiaCCoT:

    def _model(self, base=None, mean_recurrence=4):
        base = base or _tiny_base()
        cfg  = CortexConfig(arch="ccot", hidden_size=H, mean_recurrence=mean_recurrence)
        return PythiaCCoT(base, cfg)

    def test_forward_shapes_and_loss(self):
        m = self._model()
        ids, labels = _ids()
        out = m(input_ids=ids, labels=labels, num_steps=torch.tensor([0, 3]))
        assert tuple(out["logits"].shape) == (B, S, V)
        assert torch.isfinite(out["loss"])

    def test_has_no_cross_state(self):
        assert self._model().has_cross_state is False

    def test_t1_reduces_to_plain_transformer(self):
        """At T=1 the carry is zero, so CCoT == a single plain pass.  Built from
        the SAME base, PythiaCCoT(T=1) and PythiaVanilla must agree."""
        base = _tiny_base().eval()
        ccot = PythiaCCoT(base, CortexConfig(arch="ccot", hidden_size=H)).eval()
        van  = PythiaVanilla(base, CortexConfig(arch="pythia", hidden_size=H)).eval()
        ids, _ = _ids()
        with torch.no_grad():
            a = ccot(input_ids=ids, num_steps=torch.tensor([0, 1]))["logits"]
            b = van(input_ids=ids)["logits"]
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)

    def test_recurrence_changes_output(self):
        """T=2 must differ from T=1 — the carry actually feeds back."""
        m = self._model().eval()
        ids, _ = _ids()
        with torch.no_grad():
            t1 = m(input_ids=ids, num_steps=torch.tensor([0, 1]))["logits"]
            t2 = m(input_ids=ids, num_steps=torch.tensor([0, 2]))["logits"]
        assert not torch.allclose(t1, t2)

    def test_none_num_steps_uses_mean_recurrence(self):
        """num_steps=None (eval default) runs mean_recurrence passes without error."""
        m = self._model(mean_recurrence=3).eval()
        ids, _ = _ids()
        with torch.no_grad():
            out = m(input_ids=ids, num_steps=None)
        assert torch.isfinite(out["logits"].float().sum())

    def test_per_sequence_depths(self):
        """A per-sequence list of (n,k) with differing T runs the per-lane path."""
        m = self._model()
        ids, labels = _ids()
        out = m(input_ids=ids, labels=labels, num_steps=[(0, 1), (1, 2)])
        assert torch.isfinite(out["loss"])
        out["loss"].backward()  # per-lane in-place writes must stay differentiable
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.parameters())

    def test_tbptt_nograd_prefix(self):
        """With n no-grad + k grad passes, gradients still reach the parameters
        (the k grad passes are on the loss path)."""
        m = self._model()
        ids, labels = _ids()
        m(input_ids=ids, labels=labels, num_steps=torch.tensor([2, 2]))["loss"].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.parameters())

    def test_state_dict_round_trip(self):
        base = _tiny_base()
        m1 = PythiaCCoT(base, CortexConfig(arch="ccot", hidden_size=H))
        m2 = PythiaCCoT(_tiny_base(), CortexConfig(arch="ccot", hidden_size=H))
        m2.load_state_dict(m1.state_dict(), strict=True)
        ids, _ = _ids()
        with torch.no_grad():
            a = m1.eval()(input_ids=ids, num_steps=torch.tensor([0, 2]))["logits"]
            b = m2.eval()(input_ids=ids, num_steps=torch.tensor([0, 2]))["logits"]
        assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Eval-loader arch dispatch (evals/model_utils.load_checkpoint)
# ---------------------------------------------------------------------------

class TestEvalArchDispatch:

    def _patch(self, monkeypatch):
        import model_utils
        calls = {"pythia": 0, "ccot": 0, "cortex": 0}

        def fake_pythia(**kw):
            calls["pythia"] += 1
            return PythiaVanilla(_tiny_base(), CortexConfig(arch="pythia", hidden_size=H)), \
                   CortexConfig(arch="pythia", hidden_size=H)

        def fake_ccot(**kw):
            calls["ccot"] += 1
            return PythiaCCoT(_tiny_base(), CortexConfig(arch="ccot", hidden_size=H)), \
                   CortexConfig(arch="ccot", hidden_size=H)

        monkeypatch.setattr(model_utils, "build_pythia", fake_pythia)
        monkeypatch.setattr(model_utils, "build_pythia_ccot", fake_ccot)
        return model_utils, calls

    def _save(self, tmp_path, model, arch, **extra):
        cfg = {"arch": arch, "mean_recurrence": 5, **extra}
        path = str(tmp_path / f"{arch}.pt")
        torch.save({"model": model.state_dict(), "config": cfg}, path)
        return path

    def test_pythia_arch_builds_pythia(self, tmp_path, monkeypatch):
        mu, calls = self._patch(monkeypatch)
        model = PythiaVanilla(_tiny_base(), CortexConfig(arch="pythia", hidden_size=H))
        path = self._save(tmp_path, model, "pythia")
        loaded, cfg = mu.load_checkpoint(path, "fake", None, torch.float32,
                                         torch.device("cpu"))
        assert calls["pythia"] == 1 and calls["ccot"] == 0
        assert isinstance(loaded, PythiaVanilla)
        assert cfg.mean_recurrence == 5

    def test_ccot_arch_builds_ccot(self, tmp_path, monkeypatch):
        mu, calls = self._patch(monkeypatch)
        model = PythiaCCoT(_tiny_base(), CortexConfig(arch="ccot", hidden_size=H))
        path = self._save(tmp_path, model, "ccot")
        loaded, _ = mu.load_checkpoint(path, "fake", None, torch.float32,
                                       torch.device("cpu"))
        assert calls["ccot"] == 1 and calls["pythia"] == 0
        assert isinstance(loaded, PythiaCCoT)
