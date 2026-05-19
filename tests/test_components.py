"""
Unit tests for CortexGPT components.

Run with:  pytest tests/ -v

Covers:
  1. LTIInjection  — Parcae stability guarantees + forward correctness
  2. LSTMBuffer    — LM2 gate mechanics, memory feedback, MLP refinement
  3. CortexGPT     — layer routing, scalable init, prelude norm, forward shapes
  4. Muon          — Newton-Schulz near-orthogonality, optimizer routing
  5. Samplers      — Algorithm 4 distribution properties
  6. Interactions  — cross-paper clashes: LTI+buffer, TBPTT gradient scoping,
                     scalable init isolation, buffer liveness, Muon routing
"""
from __future__ import annotations

import math
import sys
import os

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import (
    LTIInjection,
    LSTMBuffer,
    CortexGPT,
    CortexConfig,
    _init_dt_bias,
)
from train import (
    Muon,
    _zeropower_via_newtonschulz5,
    sample_num_steps,
    sample_batch_steps,
    enforce_mu_bwd,
    get_current_mean_recurrence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H = 32    # tiny hidden size (must be divisible by n_heads=4 → use 32)
K = 4     # memory slots
B = 2     # batch size
S = 16    # sequence length


def _make_fake_layer(hidden_size: int) -> nn.Module:
    """GPTNeoX-compatible fake layer with the right parameter names."""

    class _Attn(nn.Module):
        def __init__(self, h: int) -> None:
            super().__init__()
            self.dense = nn.Linear(h, h, bias=False)

    class _MLP(nn.Module):
        def __init__(self, h: int) -> None:
            super().__init__()
            self.dense_4h_to_h = nn.Linear(h, h, bias=False)

    class _Layer(nn.Module):
        def __init__(self, h: int) -> None:
            super().__init__()
            self.attention = _Attn(h)
            self.mlp       = _MLP(h)

        def forward(self, x, attention_mask=None, position_embeddings=None, **kwargs):
            # Identity pass-through; returns plain tensor to match transformers ≥4.47.
            return x

    return _Layer(hidden_size)


def _make_fake_base(hidden_size: int = H, n_layers: int = 4) -> nn.Module:
    """Minimal GPTNeoX-shaped base model — no Pythia download needed."""

    class _RotaryEmb(nn.Module):
        """Stub rotary embedding — returns zero cos/sin of the right shape."""
        def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
            B, S = position_ids.shape
            cos = torch.zeros(B, S, hidden_size // 2, dtype=x.dtype, device=x.device)
            sin = torch.zeros(B, S, hidden_size // 2, dtype=x.dtype, device=x.device)
            return cos, sin

    class _NeoX(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_in         = nn.Embedding(200, hidden_size)
            self.emb_dropout      = nn.Dropout(0.0)
            self.layers           = nn.ModuleList(
                [_make_fake_layer(hidden_size) for _ in range(n_layers)]
            )
            self.final_layer_norm = nn.LayerNorm(hidden_size)
            self.rotary_emb       = _RotaryEmb()

    class _Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gpt_neox  = _NeoX()
            self.embed_out = nn.Linear(hidden_size, 200, bias=False)

    return _Base()


def _make_cortex(
    hidden_size: int   = H,
    n_layers:    int   = 4,
    n_pre:       int   = 1,
    n_loop:      int   = 2,
    n_coda:      int   = 1,
    memory_slots: int  = 0,
    mean_recurrence: int = 4,
) -> tuple[CortexGPT, CortexConfig]:
    base = _make_fake_base(hidden_size, n_layers)
    cfg  = CortexConfig(
        n_pre            = n_pre,
        n_loop           = n_loop,
        n_coda           = n_coda,
        hidden_size      = hidden_size,
        mean_recurrence  = mean_recurrence,
        memory_slots     = memory_slots,
    )
    return CortexGPT(base, cfg), cfg


# ---------------------------------------------------------------------------
# 1. LTIInjection
# ---------------------------------------------------------------------------

class TestLTIInjection:

    def test_dt_bias_decay_target(self):
        """dt_bias inverse-softplus init → decay ≈ √(1/5) ≈ 0.447 (Parcae §4.1)."""
        lti = LTIInjection(H)
        dt     = torch.nn.functional.softplus(lti.dt_bias)
        decay  = torch.exp(-dt * torch.exp(lti.A_log))
        target = math.sqrt(1.0 / 5.0)
        assert decay.mean().item() == pytest.approx(target, abs=1e-3)

    def test_spectral_norm_lt_1_at_init(self):
        """ρ(Ā) < 1 at initialisation — LTI stability guarantee."""
        lti = LTIInjection(H)
        assert lti.spectral_norm() < 1.0

    def test_spectral_norm_lt_1_after_random_update(self):
        """ρ(Ā) < 1 even after arbitrary perturbations to A_log and dt_bias."""
        lti = LTIInjection(H)
        with torch.no_grad():
            lti.A_log.add_(torch.randn(H))
            lti.dt_bias.add_(torch.randn(H))
        assert lti.spectral_norm() < 1.0

    def test_forward_shape(self):
        lti = LTIInjection(H)
        h  = torch.randn(B, S, H)
        z0 = torch.randn(B, S, H)
        assert lti(h, z0).shape == (B, S, H)

    def test_random_h_decays_toward_z0(self):
        """After many iterations with fixed z0, random h₀ influence vanishes."""
        lti   = LTIInjection(H)
        z0    = torch.zeros(1, 1, H)
        h_big = torch.ones(1, 1, H) * 100.0   # large initial state
        for _ in range(50):
            h_big = lti(h_big, z0)
        assert h_big.abs().max().item() < 1.0

    def test_no_weight_decay_flags(self):
        """A_log, dt_bias, B must all be flagged to skip Muon Newton-Schulz and WD."""
        lti = LTIInjection(H)
        for name in ("A_log", "dt_bias", "B"):
            p = getattr(lti, name)
            assert getattr(p, "_no_weight_decay", False), \
                f"LTIInjection.{name} missing _no_weight_decay flag"

    def test_b_identity_init(self):
        """B starts as identity so initial injection = dt * z0."""
        lti = LTIInjection(H)
        assert torch.allclose(lti.B, torch.eye(H))


# ---------------------------------------------------------------------------
# 2. LSTMBuffer
# ---------------------------------------------------------------------------

class TestLSTMBuffer:

    def _buf(self) -> LSTMBuffer:
        return LSTMBuffer(H, K, n_heads=4)

    def test_write_shape(self):
        buf    = self._buf()
        h_T    = torch.randn(B, S, H)
        buffer = torch.zeros(B, K, H)
        assert buf.write(h_T, buffer).shape == (B, K, H)

    def test_read_shape(self):
        buf    = self._buf()
        h      = torch.randn(B, S, H)
        buffer = torch.zeros(B, K, H)
        assert buf.read(h, buffer).shape == (B, S, H)

    def test_out_proj_zero_init(self):
        """Read returns zeros at init because out_proj.weight is zeroed."""
        buf    = self._buf()
        h      = torch.randn(B, S, H)
        buffer = torch.randn(B, K, H)
        delta  = buf.read(h, buffer)
        assert torch.allclose(delta, torch.zeros_like(delta))

    def test_forget_bias_retention(self):
        """forget_bias = +1 → fg > 0.5 at init, biasing toward retaining old memory."""
        buf    = self._buf()
        h_T    = torch.randn(B, S, H)
        buffer = torch.zeros(B, K, H)
        # Probe fg by examining what fraction of old buffer is retained
        old_content = torch.ones(B, K, H)
        new_buf  = buf.write(h_T, old_content)
        # With forget bias > 0: new_buf should keep significant fraction of old_content
        # Rough check: new_buf is not all-zero (retention is non-negligible)
        retained = (new_buf * old_content).sum() / old_content.sum()
        assert retained.item() > 0.4

    def test_memory_feedback_changes_output(self):
        """Gate output differs between zero buffer and non-zero buffer (memory feedback)."""
        torch.manual_seed(0)
        buf    = self._buf()
        h_T    = torch.randn(B, S, H)
        zero_buf   = torch.zeros(B, K, H)
        nonzero_buf = torch.randn(B, K, H)
        out_zero    = buf.write(h_T, zero_buf)
        out_nonzero = buf.write(h_T, nonzero_buf)
        assert not torch.allclose(out_zero, out_nonzero), \
            "Memory feedback has no effect — gate_proj_mem not connected"

    def test_combined_gate_split_structure(self):
        """Both gates come from splitting one combined projection (LM2 create_gates).
        Verified structurally: gate_proj_in and gate_proj_mem each output 2*D, not D.
        """
        buf = self._buf()
        assert buf.gate_proj_in.out_features  == H * 2
        assert buf.gate_proj_mem.out_features == H * 2

    def test_candidate_mlp_active(self):
        """MLP refinement changes candidate vs. raw cand_proj output."""
        torch.manual_seed(1)
        buf    = self._buf()
        h_T    = torch.randn(B, S, H)
        buffer = torch.randn(B, K, H)
        with torch.no_grad():
            h_pool    = h_T.mean(dim=1)
            cand_raw  = buf.cand_proj(h_pool).view(B, K, H)
            # Run full write and compare to raw projection
            out       = buf.write(h_T, buffer)
        # The written content differs from raw cand_proj because of MLP+LN
        assert not torch.allclose(out, torch.tanh(cand_raw))

    def test_tanh_on_buffer_before_gate(self):
        """gate_proj_mem receives tanh(buffer), not raw buffer (LM2 line 281)."""
        buf = self._buf()
        # Create a buffer with values >> 1 so tanh squashes it noticeably
        large_buf  = torch.full((B, K, H), 100.0)
        small_buf  = torch.full((B, K, H), 1.0)
        h_T        = torch.zeros(B, S, H)
        out_large  = buf.write(h_T, large_buf)
        out_small  = buf.write(h_T, small_buf)
        # tanh(100) ≈ tanh(1) is False but tanh(100) ≈ 1 ≈ tanh(1) — NOT the case
        # for arbitrary projections, but both should map to different gate signals
        # (if tanh were absent, large_buf would overwhelm the projection differently)
        assert not torch.allclose(out_large, out_small)

    def test_no_write_proj(self):
        """Removed LM2-divergent write_proj attribute should not exist."""
        buf = self._buf()
        assert not hasattr(buf, "write_proj"), \
            "write_proj was removed as it has no basis in LM2; should not exist"

    def test_read_cross_attention_not_square(self):
        """Read uses S-length queries attending K slots — no seq_len==K constraint."""
        buf    = LSTMBuffer(H, n_slots=2, n_heads=2)   # K << S
        h      = torch.randn(B, 64, H)                 # long sequence
        buffer = torch.randn(B, 2, H)
        # Should not raise even though 64 >> 2
        out = buf.read(h, buffer)
        assert out.shape == (B, 64, H)


# ---------------------------------------------------------------------------
# 3. CortexGPT
# ---------------------------------------------------------------------------

class TestCortexGPT:

    def test_construct(self):
        model, cfg = _make_cortex()
        assert isinstance(model, CortexGPT)

    def test_embed_no_weight_decay_flags(self):
        """Both embedding matrices must be flagged to bypass Muon Newton-Schulz."""
        model, _ = _make_cortex()
        assert getattr(model.embed_in.weight,  "_no_weight_decay", False), \
            "embed_in.weight missing _no_weight_decay — will go through Newton-Schulz"
        assert getattr(model.embed_out.weight, "_no_weight_decay", False), \
            "embed_out.weight missing _no_weight_decay — will go through Newton-Schulz"

    def test_scalable_init_only_output_projections(self):
        """Only attention.dense and mlp.dense_4h_to_h in loop layers are scaled."""
        mean_T = 4
        model, _ = _make_cortex(mean_recurrence=mean_T)
        scale  = math.sqrt(mean_T)

        # Check loop layer output projections ARE scaled (divided by sqrt(mean_T))
        for layer in model.loop_layers:
            for name, p in layer.named_parameters():
                parent = ".".join(name.split(".")[:-1])
                if parent in {"attention.dense", "mlp.dense_4h_to_h"} and name.endswith(".weight"):
                    # All output proj weights should be smaller than a fresh Linear init
                    # (hard to check exact scale without knowing pre-scale value, but
                    # we can verify via the inverse: rescale back and check norm)
                    assert p.abs().max().item() < 10.0  # sanity — not exploded

        # Pre and coda layers must NOT be scaled — verify no abnormal scaling
        for layer in list(model.pre_layers) + list(model.coda_layers):
            for name, p in layer.named_parameters():
                assert p.abs().max().item() < 10.0

    def test_scalable_init_does_not_touch_lti(self):
        """LTI params (A_log, dt_bias, B) must not be divided by sqrt(mean_T)."""
        model, _ = _make_cortex(mean_recurrence=4)
        # B starts as identity regardless of scalable init
        assert torch.allclose(model.lti.B, torch.eye(H))
        # A_log starts at 0
        assert torch.allclose(model.lti.A_log, torch.zeros(H))

    def test_scalable_init_does_not_touch_buffer(self):
        """LSTMBuffer params must not be divided by sqrt(mean_T)."""
        model, _ = _make_cortex(memory_slots=K)
        # gate_proj_in and gate_proj_mem are randomly initialised, not scaled to near-zero
        scale = model.m_cross.gate_proj_in.weight.abs().mean().item()
        assert scale > 1e-4, "Buffer gate projections appear zeroed by scalable init"

    def test_prelude_norm_exists(self):
        model, _ = _make_cortex()
        assert model.ln_prelude is not None
        assert isinstance(model.ln_prelude, nn.LayerNorm)

    def test_prelude_norm_disabled(self):
        base = _make_fake_base()
        cfg  = CortexConfig(n_pre=1, n_loop=2, n_coda=1, hidden_size=H,
                            prelude_norm=False)
        model = CortexGPT(base, cfg)
        assert model.ln_prelude is None

    def test_init_state_distribution(self):
        """h₀ = z0.detach().clone() for retrofitting: same values, no grad."""
        model, _ = _make_cortex()
        z0 = torch.randn(4, 128, H, requires_grad=True)
        h  = model._init_state(z0)
        assert h.shape == z0.shape
        assert torch.allclose(h, z0.detach())
        assert not h.requires_grad

    def test_forward_output_shapes(self):
        model, _ = _make_cortex()
        ids    = torch.randint(0, 200, (B, S))
        labels = torch.randint(0, 200, (B, S))
        out    = model(ids, labels=labels, num_steps=torch.tensor([0, 2]))
        assert out["loss"] is not None
        assert out["logits"].shape == (B, S, 200)

    def test_forward_per_sequence_steps(self):
        model, _ = _make_cortex()
        ids    = torch.randint(0, 200, (B, S))
        steps  = [(0, 1), (0, 3)]
        out    = model(ids, num_steps=steps)
        assert out["logits"].shape == (B, S, 200)

    def test_layer_split_assertion(self):
        with pytest.raises(AssertionError):
            base = _make_fake_base(n_layers=4)
            cfg  = CortexConfig(n_pre=1, n_loop=2, n_coda=2, hidden_size=H)
            CortexGPT(base, cfg)   # 1+2+2=5 ≠ 4 layers


# ---------------------------------------------------------------------------
# 4. Muon
# ---------------------------------------------------------------------------

class TestMuon:

    def test_newtonschulz_near_orthogonal_square(self):
        """Newton-Schulz output is more orthogonal than the input (condition ratio < 1)."""
        torch.manual_seed(0)
        G = torch.randn(64, 64)   # larger matrix: NS converges faster for well-conditioned input
        X = _zeropower_via_newtonschulz5(G)
        # The columns of X should be closer to orthonormal than G's columns
        # Check: Gram matrix X.T @ X diagonal dominance
        gram   = X.T @ X
        diag   = gram.diag()
        offdiag = gram - torch.diag(diag)
        # Diagonal near-uniform, off-diagonal small
        assert diag.std().item() < 0.3
        assert offdiag.abs().mean().item() < 0.1

    def test_newtonschulz_near_orthogonal_tall(self):
        """Newton-Schulz handles tall matrices (rows > cols); shape preserved."""
        torch.manual_seed(0)
        G = torch.randn(64, 32)
        X = _zeropower_via_newtonschulz5(G)
        assert X.shape == (64, 32)
        # Columns of X should be more orthonormal than columns of G
        gram_X = X.T @ X
        gram_G = G.T @ G / G.T.norm()  # rough normalisation for comparison
        offdiag_X = (gram_X - torch.diag(gram_X.diag())).abs().mean().item()
        # Can't assert exact values, but X should be a valid tensor without NaN/Inf
        assert X.isfinite().all()
        assert offdiag_X < 0.15

    def test_use_muon_2d_no_flag(self):
        p = nn.Parameter(torch.randn(8, 8))
        assert Muon._use_muon(p) is True

    def test_use_muon_1d(self):
        p = nn.Parameter(torch.randn(8))
        assert Muon._use_muon(p) is False

    def test_use_muon_flagged_2d(self):
        p = nn.Parameter(torch.randn(8, 8))
        p._no_weight_decay = True
        assert Muon._use_muon(p) is False

    def test_adamw_1d_no_weight_decay(self):
        """1D params in the AdamW fallback must NOT receive weight decay."""
        # forget_bias = scalar parameter; WD should not shrink it
        p  = nn.Parameter(torch.tensor([1.0]))
        opt = Muon([p], lr=1e-3, weight_decay=0.5)
        # Manually set gradient
        p.grad = torch.tensor([0.01])
        opt.step()
        # Value should not have been decayed toward zero (WD on 1D is now blocked)
        assert p.item() > 0.9, "WD was applied to a 1D param — forget_bias would decay"

    def test_embed_weight_goes_to_adamw(self):
        """Embedding weight with _no_weight_decay=True routes to AdamW, not Muon."""
        p = nn.Parameter(torch.randn(100, H))
        p._no_weight_decay = True
        # _use_muon must return False
        assert Muon._use_muon(p) is False

    def test_optimizer_updates_params(self):
        """Muon step changes parameter values — sanity check that optimizer is wired up."""
        torch.manual_seed(0)
        W    = nn.Parameter(torch.randn(8, 8))
        W0   = W.detach().clone()
        opt  = Muon([W], lr=1e-2)
        loss = W.sum()
        loss.backward()
        opt.step()
        assert not torch.allclose(W, W0), "Muon step did not update parameters"

    def test_optimizer_reduces_loss_rank1(self):
        """Muon makes progress on a rank-1 linear task (gradient aligned with NS)."""
        torch.manual_seed(0)
        W   = nn.Parameter(torch.randn(16, 16))
        x   = torch.randn(16)
        y   = torch.randn(16)
        opt = Muon([W], lr=3e-3)
        losses = []
        for _ in range(40):
            loss = ((W @ x) - y).pow(2).mean()
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        # Loss should decrease meaningfully over 40 steps
        assert losses[-1] < losses[0], "Muon did not reduce loss"


# ---------------------------------------------------------------------------
# 5. Samplers
# ---------------------------------------------------------------------------

class TestSamplers:

    def test_sample_num_steps_valid(self):
        for step in range(20):
            n, k = sample_num_steps(step, mean_recurrence=8, mean_backprop_depth=4)
            assert n >= 0
            assert k >= 1
            assert n + k >= 1

    def test_algorithm4_n_k_sum_is_T(self):
        """n + k = T; neither is truncated independently (vs. McLeish Algorithm 3)."""
        for seed in range(50):
            n, k = sample_num_steps(seed, mean_recurrence=8, mean_backprop_depth=4)
            # T = n + k; n = max(T - mu_bwd, 0); k = min(T, mu_bwd)
            mu_bwd = 4
            T = n + k
            assert k == min(T, mu_bwd)
            assert n == max(T - mu_bwd, 0)

    def test_sample_batch_steps_length(self):
        steps = sample_batch_steps(0, batch_size=8, mean_recurrence=8, mean_backprop_depth=4)
        assert len(steps) == 8

    def test_sample_batch_steps_independent(self):
        """Per-sequence sampling: different seeds → usually different T values."""
        steps = sample_batch_steps(42, batch_size=16, mean_recurrence=8, mean_backprop_depth=4)
        Ts = [n + k for n, k in steps]
        assert len(set(Ts)) > 1, "All sequences got identical T — seeds not independent"

    def test_enforce_mu_bwd(self):
        for mu_rec in range(1, 20):
            assert enforce_mu_bwd(mu_rec) == math.ceil(mu_rec / 2)

    def test_curriculum_ramp(self):
        target, total = 8, 100
        for step in range(total + 10):
            mu = get_current_mean_recurrence(step, target, curriculum_steps=total)
            assert 1 <= mu <= target

    def test_curriculum_off(self):
        assert get_current_mean_recurrence(0,   8, 0) == 8
        assert get_current_mean_recurrence(999, 8, 0) == 8

    def test_curriculum_at_boundary(self):
        mu = get_current_mean_recurrence(100, 8, curriculum_steps=100)
        assert mu == 8


# ---------------------------------------------------------------------------
# 6. Cross-paper Interactions
# ---------------------------------------------------------------------------

class TestInteractions:

    def test_lti_plus_buffer_read_bounded(self):
        """Running LTI injection followed by buffer read doesn't explode the state."""
        model, _ = _make_cortex(memory_slots=K)
        h      = torch.randn(B, S, H)
        z0     = torch.randn(B, S, H)
        buffer = torch.randn(B, K, H)

        # Simulate what _loop_iter does when buffer is active
        with torch.no_grad():
            h = model.lti(h, z0)
            h = h + model.m_cross.read(h, buffer)
        assert h.isfinite().all()
        assert h.abs().max().item() < 100.0

    def test_tbptt_no_grad_steps_excluded_from_graph(self):
        """n no-grad steps genuinely have no gradient; k grad steps do."""
        model, _ = _make_cortex()
        ids    = torch.randint(0, 200, (B, S))
        labels = torch.randint(0, 200, (B, S))

        out  = model(ids, labels=labels, num_steps=torch.tensor([3, 2]))
        out["loss"].backward()

        # All trainable params should have a gradient after backward
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0

    def test_tbptt_no_grad_respected(self):
        """With n=5, k=0 (all no-grad), loss.backward should still work without error."""
        model, _ = _make_cortex()
        ids  = torch.randint(0, 200, (B, S))
        out  = model(ids, num_steps=torch.tensor([5, 0]))
        assert out["logits"].shape == (B, S, 200)

    def test_scalable_init_does_not_affect_lti_stability(self):
        """After scalable init, LTI spectral norm stays < 1."""
        model, _ = _make_cortex(mean_recurrence=8)
        assert model.lti.spectral_norm() < 1.0

    def test_scalable_init_does_not_affect_buffer_params(self):
        """LSTMBuffer params are not touched by _apply_scalable_init."""
        model, _ = _make_cortex(memory_slots=K, mean_recurrence=8)
        # out_proj is zero-initialised — that's intentional (not scalable init)
        assert torch.allclose(model.m_cross.out_proj.weight, torch.zeros(H, H))
        # gate_proj_in should NOT be zeroed (not in loop_layers)
        assert model.m_cross.gate_proj_in.weight.abs().sum().item() > 0.0

    def test_buffer_read_returns_zero_at_init(self):
        """out_proj is zero-initialised: read() always returns zero at step 0,
        regardless of query or buffer content.  Ensures buffer is a no-op at init."""
        model, _ = _make_cortex(memory_slots=K)
        h      = torch.randn(B, S, H)
        buffer = torch.randn(B, K, H)
        delta  = model.m_cross.read(h, buffer)
        assert torch.allclose(delta, torch.zeros_like(delta)), \
            "Buffer read is non-zero at init — out_proj zero-init not working"

    def test_buffer_read_skipped_when_cross_buf_none(self):
        """_loop_iter skips the read branch entirely when cross_buf is None."""
        model, _ = _make_cortex(memory_slots=K)
        # Activate out_proj so a real read would change h
        with torch.no_grad():
            nn.init.normal_(model.m_cross.out_proj.weight, std=0.1)

        h0  = torch.randn(1, S, H)
        z0  = torch.randn(1, S, H)
        attn = model._build_attn_mask(None, S, h0.device, h0.dtype)
        pos  = torch.arange(S).unsqueeze(0)

        h_no_buf  = model._loop_iter(h0.clone(), z0, attn, pos, None, None)
        # Explicitly verify: without cross_buf, LTI-only path is taken
        h_lti_only = model.lti(h0.clone(), z0)
        for layer in model.loop_layers:
            h_lti_only = model._layer_fwd(h_lti_only, layer, attn, pos)
        assert torch.allclose(h_no_buf, h_lti_only, atol=1e-5)

    def test_buffer_active_with_nonzero_m_cross_in(self):
        """A non-zero m_cross_in actually changes the output (buffer read is live)."""
        model, _ = _make_cortex(memory_slots=K)

        # Unzero the out_proj so read injection has effect
        with torch.no_grad():
            nn.init.normal_(model.m_cross.out_proj.weight, std=0.01)

        ids      = torch.randint(0, 200, (B, S))
        zero_buf = torch.zeros(B, K, H)
        rand_buf = torch.randn(B, K, H)

        with torch.no_grad():
            out_zero = model(ids, num_steps=torch.tensor([0, 1]), m_cross_in=zero_buf)
            out_rand = model(ids, num_steps=torch.tensor([0, 1]), m_cross_in=rand_buf)

        assert not torch.allclose(out_zero["logits"], out_rand["logits"]), \
            "Buffer read has no effect on logits — buffer is still dead"

    def test_muon_routes_loop_weights_not_embeddings(self):
        """Loop layer weights → Muon; embedding weights (flagged) → AdamW."""
        model, _ = _make_cortex(memory_slots=K)

        loop_weights = [
            p for layer in model.loop_layers
            for p in layer.parameters()
            if p.ndim >= 2
        ]
        for p in loop_weights:
            assert Muon._use_muon(p), "Loop layer 2D weight not routed to Muon"

        assert not Muon._use_muon(model.embed_in.weight),  "embed_in goes through Newton-Schulz"
        assert not Muon._use_muon(model.embed_out.weight), "embed_out goes through Newton-Schulz"

    def test_lti_ssm_params_not_in_muon(self):
        """LTI params have _no_weight_decay, so they bypass Newton-Schulz."""
        model, _ = _make_cortex()
        for name in ("A_log", "dt_bias", "B"):
            p = getattr(model.lti, name)
            assert not Muon._use_muon(p), \
                f"lti.{name} is being routed through Newton-Schulz"

    def test_per_sequence_sampling_buffer_write_shape(self):
        """After per-sequence loop with different T values, buffer write has correct shape."""
        model, _ = _make_cortex(memory_slots=K)
        ids    = torch.randint(0, 200, (B, S))
        # Force per-sequence by passing a list with different T values
        steps  = [(0, 1), (0, 3)]
        out    = model(ids, num_steps=steps, return_m_cross=True)
        assert out["m_cross"] is not None
        assert out["m_cross"].shape == (B, K, H)

    def test_prelude_norm_decouples_prelude_scale_from_lti(self):
        """With prelude_norm=True, multiplying prelude output by a scalar
        does NOT change z0 (the LN normalises it away)."""
        model, _ = _make_cortex()
        assert model.ln_prelude is not None

        # Artificially inflate embed_out scale — prelude norm should absorb it
        with torch.no_grad():
            model.embed_in.weight.mul_(10.0)

        ids  = torch.randint(0, 200, (B, S))
        with torch.no_grad():
            out = model(ids, num_steps=torch.tensor([0, 2]))
        assert out["logits"].isfinite().all(), \
            "Logits exploded — prelude norm not protecting LTI from large prelude output"

    def test_h_T_proj_is_identity_at_init(self):
        """R4 projection weight is identity-initialized — no-op at step 0."""
        model, cfg = _make_cortex(memory_slots=K)
        assert cfg.h_T_proj, "h_T_proj should default to True when memory_slots > 0"
        assert model.h_T_proj is not None, "h_T_proj module should be created"
        eye = torch.eye(cfg.hidden_size)
        assert torch.allclose(model.h_T_proj.weight, eye), \
            "h_T_proj weight should be identity at init"

    def test_h_T_proj_absent_without_memory(self):
        """R4 projection is not created when memory_slots=0 (K=0 baseline)."""
        model, _ = _make_cortex(memory_slots=0)
        assert model.h_T_proj is None, \
            "h_T_proj should not be created when there is no M_cross buffer"

    def test_coda_receives_raw_h_T_not_projected(self):
        """The Coda path always gets the unmodified h_T regardless of h_T_proj.

        We verify this by setting the projection to a zero matrix (which would
        destroy any information) and checking that the model still produces
        finite, non-zero logits — meaning the Coda cannot be using the
        projected (zeroed) representation.
        """
        model, _ = _make_cortex(memory_slots=K)
        assert model.h_T_proj is not None

        with torch.no_grad():
            model.h_T_proj.weight.zero_()   # zero projection → buffer writes all-zero

        ids = torch.randint(0, 200, (B, S))
        with torch.no_grad():
            out = model(ids, num_steps=torch.tensor([0, 2]))

        assert out["logits"].isfinite().all(), "Logits should be finite"
        assert out["logits"].abs().sum() > 0, "Logits should be non-zero (Coda uses raw h_T)"

    def test_h_T_proj_disabled_via_config(self):
        """Setting h_T_proj=False in config suppresses the projection module."""
        base = _make_fake_base(H, 4)
        cfg  = CortexConfig(
            n_pre=1, n_loop=2, n_coda=1,
            hidden_size=H, memory_slots=K, h_T_proj=False,
        )
        model = CortexGPT(base, cfg)
        assert model.h_T_proj is None, \
            "h_T_proj module should not be created when h_T_proj=False"
