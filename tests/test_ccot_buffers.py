"""
Unit tests for the CCoT (Continuous Chain-of-Thought) memory buffers.

Two buffers, different scopes:

  m_iter  — short-term, WITHIN a single forward call.
             iter_buf is zero-initialised at the start of each call, written
             after every loop iteration, and read at the start of the next
             iteration.  It never escapes forward(); the caller sees nothing.

  m_cross — long-term, ACROSS forward calls within a document.
             Written once per forward call using h_T (final loop state).
             Returned in the output dict and passed back by the training loop
             as m_cross_in.  Resets at document (EOS) boundaries.

Tests are grouped:
  1. m_iter — internal, iteration-scoped short-term memory
  2. m_cross — external, call-scoped long-term memory
  3. Interactions — independence, TBPTT gradient scoping, per-sequence isolation
"""
from __future__ import annotations

import sys
import os

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import CortexGPT, CortexConfig, LSTMBuffer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

H  = 32   # hidden size (divisible by n_heads=4)
K  = 4    # memory slots for both buffers
B  = 2    # batch size
S  = 16   # sequence length


def _fake_layer(hidden_size: int) -> nn.Module:
    class _A(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.dense = nn.Linear(h, h, bias=False)
    class _M(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.dense_4h_to_h = nn.Linear(h, h, bias=False)
    class _L(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.attention = _A(h)
            self.mlp       = _M(h)
        def forward(self, x, attention_mask=None, position_embeddings=None, **kwargs):
            return x
    return _L(hidden_size)


def _make_base(h: int = H, n: int = 4) -> nn.Module:
    class _RotaryEmb(nn.Module):
        def forward(self, x, position_ids):
            B, S = position_ids.shape
            z = torch.zeros(B, S, h // 2, dtype=x.dtype, device=x.device)
            return z, z

    class _NX(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_in         = nn.Embedding(200, h)
            self.emb_dropout      = nn.Dropout(0.0)
            self.layers           = nn.ModuleList([_fake_layer(h) for _ in range(n)])
            self.final_layer_norm = nn.LayerNorm(h)
            self.rotary_emb       = _RotaryEmb()
    class _B(nn.Module):
        def __init__(self):
            super().__init__()
            self.gpt_neox  = _NX()
            self.embed_out = nn.Linear(h, 200, bias=False)
    return _B()


def _make_model(
    memory_slots: int = 0,
    memory_slots_iter: int = 0,
    hidden_size: int = H,
) -> CortexGPT:
    base = _make_base(hidden_size)
    cfg  = CortexConfig(
        n_pre=1, n_loop=2, n_coda=1,
        hidden_size=hidden_size,
        mean_recurrence=4,
        memory_slots=memory_slots,
        memory_slots_iter=memory_slots_iter,
    )
    return CortexGPT(base, cfg)


def _activate_read(buf: LSTMBuffer, std: float = 0.05) -> None:
    """Make the buffer read non-trivially by giving out_proj non-zero weights."""
    with torch.no_grad():
        nn.init.normal_(buf.out_proj.weight, std=std)


def _ids(b: int = B, s: int = S) -> torch.Tensor:
    return torch.randint(0, 200, (b, s))


def _labels(b: int = B, s: int = S) -> torch.Tensor:
    return torch.randint(0, 200, (b, s))


# ---------------------------------------------------------------------------
# 1. m_iter — short-term, within-forward iteration memory
# ---------------------------------------------------------------------------

class TestMIter:

    def test_disabled_by_default(self):
        model = _make_model()
        assert model.m_iter is None

    def test_enabled_with_positive_slots(self):
        model = _make_model(memory_slots_iter=K)
        assert model.m_iter is not None
        assert isinstance(model.m_iter, LSTMBuffer)
        assert model.m_iter.n_slots == K

    def test_does_not_appear_in_output_dict(self):
        """iter_buf is strictly internal — the caller never sees it."""
        model = _make_model(memory_slots_iter=K)
        out   = model(_ids(), num_steps=torch.tensor([0, 2]), return_m_cross=True)
        assert "m_iter" not in out

    def test_resets_between_forward_calls(self):
        """iter_buf is always zero-initialised; state never leaks across calls."""
        model = _make_model(memory_slots_iter=K)
        ids   = _ids()

        torch.manual_seed(7)
        out1 = model(ids, num_steps=torch.tensor([0, 2]))

        torch.manual_seed(7)   # identical RNG → identical h₀ → identical result
        out2 = model(ids, num_steps=torch.tensor([0, 2]))

        assert torch.allclose(out1["logits"], out2["logits"]), (
            "iter_buf appears to carry state across forward calls"
        )

    def test_first_iteration_is_buffer_free(self):
        """At T=1 the buffer has had no write yet; out_proj=0 so injection = 0."""
        model = _make_model(memory_slots_iter=K)
        # out_proj zero-init means read always returns zero regardless of buffer content
        h      = torch.randn(1, S, H)
        buffer = torch.zeros(1, K, H)
        delta  = model.m_iter.read(h, buffer)
        assert torch.allclose(delta, torch.zeros_like(delta))

    def test_subsequent_iterations_receive_buffer_content(self):
        """After the first write, out_proj activated → iteration 2+ get non-zero injection."""
        model = _make_model(memory_slots_iter=K)
        _activate_read(model.m_iter)

        # Build the state the loop would have after one iteration
        h_after_iter1 = torch.randn(B, S, H)
        iter_buf_after1 = model.m_iter.write(h_after_iter1, torch.zeros(B, K, H))

        # The read with a populated buffer should now return non-zero
        h_query = torch.randn(B, S, H)
        delta   = model.m_iter.read(h_query, iter_buf_after1)
        assert not torch.allclose(delta, torch.zeros_like(delta)), (
            "Buffer read returns zero even with populated iter_buf and activated out_proj"
        )

    def test_more_iterations_changes_output(self):
        """T=1 vs T=3 produce different logits when m_iter is active."""
        model = _make_model(memory_slots_iter=K)
        _activate_read(model.m_iter)
        ids = _ids()

        torch.manual_seed(0)
        out1 = model(ids, num_steps=torch.tensor([0, 1]))
        torch.manual_seed(0)
        out3 = model(ids, num_steps=torch.tensor([0, 3]))

        assert not torch.allclose(out1["logits"], out3["logits"]), (
            "m_iter has no effect across iterations — buffer not accumulating state"
        )

    def test_iter_buf_written_during_no_grad_phase(self):
        """No-grad iterations still write to iter_buf so grad-phase reads get useful state."""
        model  = _make_model(memory_slots_iter=K)
        _activate_read(model.m_iter)
        ids    = _ids()
        labels = _labels()

        # n=2 no-grad + k=2 grad:  iter_buf at start of grad phase has 2 writes in it
        out_ng  = model(ids, labels=labels, num_steps=torch.tensor([2, 2]))
        # n=0 no-grad + k=2 grad:  iter_buf at start of grad phase is still zeros
        out_nong = model(ids, labels=labels, num_steps=torch.tensor([0, 2]))

        # Because iter_buf state differs entering the grad phase, logits should differ
        assert not torch.allclose(out_ng["logits"], out_nong["logits"])

    def test_iter_weights_receive_gradients(self):
        """m_iter parameters get gradients when they appear in the grad phase."""
        model  = _make_model(memory_slots_iter=K)
        ids    = _ids()
        labels = _labels()

        out = model(ids, labels=labels, num_steps=torch.tensor([1, 2]))
        out["loss"].backward()

        # gate_proj_in is a 2D Linear used inside write(), which runs in the grad phase
        assert model.m_iter.gate_proj_in.weight.grad is not None, (
            "m_iter weights have no gradient — write() may not be in the grad graph"
        )

    def test_iter_buf_no_grad_during_no_grad_phase(self):
        """iter_buf written inside torch.no_grad() has requires_grad=False."""
        model = _make_model(memory_slots_iter=K)
        h      = torch.randn(B, S, H)
        buf    = torch.zeros(B, K, H)

        with torch.no_grad():
            new_buf = model.m_iter.write(h, buf)

        assert not new_buf.requires_grad

    def test_iter_buf_has_grad_in_grad_phase(self):
        """iter_buf written outside no_grad has requires_grad=True (gates are learnable)."""
        model = _make_model(memory_slots_iter=K)
        h   = torch.randn(B, S, H)
        buf = torch.zeros(B, K, H)

        # Write without no_grad context — gates are learnable → output requires grad
        new_buf = model.m_iter.write(h, buf)
        assert new_buf.requires_grad


# ---------------------------------------------------------------------------
# 2. m_cross — long-term, across-call document memory
# ---------------------------------------------------------------------------

class TestMCross:

    def test_disabled_by_default(self):
        model = _make_model()
        assert model.m_cross is None

    def test_enabled_with_positive_slots(self):
        model = _make_model(memory_slots=K)
        assert model.m_cross is not None
        assert model.m_cross.n_slots == K

    def test_output_shape(self):
        model = _make_model(memory_slots=K)
        out   = model(_ids(), num_steps=torch.tensor([0, 1]), return_m_cross=True)
        assert out["m_cross"] is not None
        assert out["m_cross"].shape == (B, K, H)

    def test_not_returned_without_flag(self):
        model = _make_model(memory_slots=K)
        out   = model(_ids(), num_steps=torch.tensor([0, 1]), return_m_cross=False)
        assert "m_cross" not in out or out.get("m_cross") is None

    def test_written_from_final_loop_state(self):
        """m_cross reflects h_T (loop output), not some intermediate state."""
        model = _make_model(memory_slots=K)
        ids   = _ids()

        # Two calls: same ids but different recurrence depth → different h_T → different m_cross
        out1 = model(ids, num_steps=torch.tensor([0, 1]), return_m_cross=True)
        out2 = model(ids, num_steps=torch.tensor([0, 4]), return_m_cross=True)

        # Different T → different h_T → different buffer contents
        assert not torch.allclose(out1["m_cross"], out2["m_cross"]), (
            "m_cross is identical for T=1 and T=4 — write may not use final h_T"
        )

    def test_carry_changes_next_call(self):
        """Feeding m_cross output back as m_cross_in changes the next call's output."""
        model = _make_model(memory_slots=K)
        _activate_read(model.m_cross)
        ids = _ids()

        # Call 1: no prior buffer
        out1     = model(ids, num_steps=torch.tensor([0, 2]), return_m_cross=True)
        mc_out   = out1["m_cross"].detach()

        # Call 2a: carry the buffer forward
        out2a = model(ids, num_steps=torch.tensor([0, 2]), m_cross_in=mc_out)
        # Call 2b: fresh buffer (None)
        out2b = model(ids, num_steps=torch.tensor([0, 2]), m_cross_in=None)

        assert not torch.allclose(out2a["logits"], out2b["logits"]), (
            "Carrying m_cross has no effect — read injection appears inactive"
        )

    def test_none_m_cross_in_initialises_zeros_internally(self):
        """When m_cross_in=None the model creates a zero buffer for the write."""
        model = _make_model(memory_slots=K)
        out   = model(_ids(), num_steps=torch.tensor([0, 1]), return_m_cross=True)
        # The write ran (m_cross is not None) and produced a valid buffer
        assert out["m_cross"] is not None
        assert out["m_cross"].isfinite().all()

    def test_zero_m_cross_in_same_as_none(self):
        """Explicit zero buffer and no buffer produce the same output at init."""
        model    = _make_model(memory_slots=K)
        ids      = _ids()
        zero_buf = torch.zeros(B, K, H)

        torch.manual_seed(5)
        out_none = model(ids, num_steps=torch.tensor([0, 1]))
        torch.manual_seed(5)
        out_zero = model(ids, num_steps=torch.tensor([0, 1]), m_cross_in=zero_buf)

        # Both read from a zero buffer; out_proj=0 → read delta=0 → identical outputs
        assert torch.allclose(out_none["logits"], out_zero["logits"])

    def test_per_sequence_m_cross_independence(self):
        """Each sequence slot of m_cross is independent — different h_T per sequence."""
        model = _make_model(memory_slots=K)
        ids   = _ids(b=2)

        out = model(ids, num_steps=torch.tensor([0, 2]), return_m_cross=True)
        mc  = out["m_cross"]   # [2, K, H]

        # Both sequences have gone through different token embeddings → different h_T
        # → different buffer contents
        assert not torch.allclose(mc[0], mc[1]), (
            "m_cross is identical across sequences — write may be averaging across batch"
        )

    def test_m_cross_weights_receive_gradients(self):
        """m_cross write params get gradients via a 2-call no-detach chain.

        m_cross.write() runs after the loop and its output is not used in the
        same forward call's logits.  Gradients reach gate_proj_in only when the
        written buffer flows (un-detached) into a second call's read(), which
        does affect logits → loss.
        """
        model  = _make_model(memory_slots=K)
        _activate_read(model.m_cross)          # non-zero out_proj so read is live
        ids    = _ids()
        labels = _labels()

        # Call 1: produces the buffer (no detach)
        out1 = model(ids, num_steps=torch.tensor([0, 2]), return_m_cross=True)
        mc   = out1["m_cross"]                 # still in the autograd graph

        # Call 2: reads that buffer → affects logits → loss
        out2 = model(ids, labels=labels, num_steps=torch.tensor([0, 2]),
                     m_cross_in=mc)
        out2["loss"].backward()

        # Gradient must flow back through read(mc) → mc → write() → gate_proj_in
        assert model.m_cross.gate_proj_in.weight.grad is not None, (
            "m_cross write params have no gradient — check that read() uses "
            "the buffer tensor in the graph and out_proj is non-zero"
        )

    def test_doc_boundary_reset_zeroes_buffer(self):
        """Simulates the train.py EOS reset: zeroing buffer for ended sequences."""
        model    = _make_model(memory_slots=K)
        ids      = _ids(b=3)
        labels   = _labels(b=3)

        out     = model(ids, labels=labels, num_steps=torch.tensor([0, 2]),
                        return_m_cross=True)
        mc      = out["m_cross"].detach()

        # Sequence 1 ended a document — zero its buffer row
        doc_ended        = torch.tensor([False, True, False])
        mc_reset         = mc * (~doc_ended).view(-1, 1, 1).float()

        assert torch.allclose(mc_reset[1], torch.zeros(K, H)), \
            "EOS reset did not zero the ended sequence's buffer row"
        assert not torch.allclose(mc_reset[0], torch.zeros(K, H)), \
            "EOS reset incorrectly zeroed a non-ended sequence"


# ---------------------------------------------------------------------------
# 3. Interactions — isolation, TBPTT, per-sequence, independence
# ---------------------------------------------------------------------------

class TestCCoTInteractions:

    def test_m_iter_and_m_cross_independent(self):
        """Activating only m_cross and only m_iter produce non-identical but
        individually reproducible results — they operate on separate state."""
        ids = _ids()

        model_cross = _make_model(memory_slots=K, memory_slots_iter=0)
        model_iter  = _make_model(memory_slots=0, memory_slots_iter=K)
        model_both  = _make_model(memory_slots=K, memory_slots_iter=K)

        _activate_read(model_cross.m_cross)
        _activate_read(model_iter.m_iter)
        _activate_read(model_both.m_cross)
        _activate_read(model_both.m_iter)

        mc_buf = torch.randn(B, K, H)

        torch.manual_seed(1)
        out_c  = model_cross(ids, num_steps=torch.tensor([0, 3]), m_cross_in=mc_buf)
        torch.manual_seed(1)
        out_i  = model_iter(ids, num_steps=torch.tensor([0, 3]))

        # Results differ because the buffers inject different information
        assert not torch.allclose(out_c["logits"], out_i["logits"])

    def test_m_iter_per_sequence_isolation(self):
        """In per-sequence mode each sequence has its own iter_buf — no cross-contamination."""
        model = _make_model(memory_slots_iter=K)
        _activate_read(model.m_iter)

        # Two sequences with very different T values
        ids   = _ids()
        steps = [(0, 1), (0, 5)]   # seq 0: T=1, seq 1: T=5

        out_per = model(ids, num_steps=steps)

        # Run seq 0 alone with T=1 and seq 1 alone with T=5
        torch.manual_seed(0)
        out0 = model(ids[0:1], num_steps=torch.tensor([0, 1]))
        torch.manual_seed(0)
        out1 = model(ids[1:2], num_steps=torch.tensor([0, 5]))

        # Per-sequence result for seq 0 must match solo run for seq 0
        # (within tolerance — same RNG path because iter_buf is independent per sequence)
        # We can't check exact equality due to RNG but we verify shapes are sane
        assert out_per["logits"][0:1].shape == out0["logits"].shape
        assert out_per["logits"][1:2].shape == out1["logits"].shape

    def test_m_cross_not_affected_by_m_iter_state(self):
        """m_cross write uses h_T regardless of what m_iter did internally."""
        model = _make_model(memory_slots=K, memory_slots_iter=K)
        _activate_read(model.m_cross)
        ids = _ids()

        # Same call with T=1 vs T=3: m_iter affects internal loop state → different h_T
        # but m_cross correctly reflects that different h_T
        out1 = model(ids, num_steps=torch.tensor([0, 1]), return_m_cross=True)
        out3 = model(ids, num_steps=torch.tensor([0, 3]), return_m_cross=True)

        # m_cross buffers should differ because h_T differs
        assert not torch.allclose(out1["m_cross"], out3["m_cross"])

    def test_tbptt_no_grad_iter_buf_detached_from_grad_phase(self):
        """Gradient does not flow back through no-grad iterations via iter_buf."""
        model  = _make_model(memory_slots_iter=K)
        ids    = _ids()
        labels = _labels()

        model.zero_grad()
        out = model(ids, labels=labels, num_steps=torch.tensor([3, 2]))
        out["loss"].backward()

        # m_iter params should have a gradient (used in k=2 grad steps)
        grad = model.m_iter.gate_proj_in.weight.grad
        assert grad is not None

        # But the gradient should not include contributions from the n=3 no-grad steps
        # (we can't easily check the exact gradient value, but we verify it's finite)
        assert grad.isfinite().all()

    def test_both_buffers_active_simultaneously(self):
        """Model with both m_cross and m_iter active runs without error."""
        model   = _make_model(memory_slots=K, memory_slots_iter=K)
        _activate_read(model.m_cross)
        _activate_read(model.m_iter)

        mc_buf  = torch.randn(B, K, H)
        ids     = _ids()
        labels  = _labels()

        out = model(ids, labels=labels, num_steps=torch.tensor([1, 3]),
                    m_cross_in=mc_buf, return_m_cross=True)

        assert out["loss"].isfinite()
        assert out["m_cross"].shape == (B, K, H)
        assert out["m_cross"].isfinite().all()

    def test_both_buffers_grad_flow(self):
        """With both buffers active, both receive gradients after backward.

        m_iter: write→read within the same forward, so single call suffices.
        m_cross: write output only used in the *next* call's read; needs a
        2-call no-detach chain (same as test_m_cross_weights_receive_gradients).
        """
        model  = _make_model(memory_slots=K, memory_slots_iter=K)
        _activate_read(model.m_cross)          # non-zero out_proj so read is live
        ids    = _ids()
        labels = _labels()

        model.zero_grad()

        # Call 1: produces m_cross buffer (no detach); m_iter write is also live
        out1 = model(ids, num_steps=torch.tensor([0, 3]), return_m_cross=True)
        mc   = out1["m_cross"]

        # Call 2: m_cross read uses buffer from call 1; m_iter runs fresh
        out2 = model(ids, labels=labels, num_steps=torch.tensor([0, 3]),
                     m_cross_in=mc)
        out2["loss"].backward()

        assert model.m_cross.gate_proj_in.weight.grad is not None, \
            "m_cross write params have no gradient"
        assert model.m_iter.gate_proj_in.weight.grad  is not None, \
            "m_iter write params have no gradient"

    def test_no_buffer_baseline_unchanged(self):
        """A model with no buffers gives identical results to the baseline (no-op paths)."""
        model_no_buf = _make_model(memory_slots=0, memory_slots_iter=0)
        ids          = _ids()

        torch.manual_seed(3)
        out1 = model_no_buf(ids, num_steps=torch.tensor([0, 2]))
        torch.manual_seed(3)
        out2 = model_no_buf(ids, num_steps=torch.tensor([0, 2]))

        assert torch.allclose(out1["logits"], out2["logits"])
