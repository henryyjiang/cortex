"""
CortexGPT: Pre/Loop/Coda architecture with Parcae LTI injection on Pythia (GPTNeoX).

Key components
--------------
  LTIInjection   — Parcae diagonal injection, ρ(Ā) < 1 by construction
  LSTMBuffer     — LM2-style K-slot LSTM-gated memory (M_cross / M_iter)
  CortexGPT      — wraps Pythia layers, adds recurrence + optional buffer

Layer split (Pythia 160M, 12 layers): n_pre=2, n_loop=8, n_coda=2
Layer split (Pythia 1B,   24 layers): n_pre=4, n_loop=16, n_coda=4

Changes from v1
---------------
  [model #2]  h₀ ~ N(0, σ²) random init instead of z0.clone()
              (both McLeish §3 and Parcae §4.1 — random state, not prelude copy)
  [model #3]  _apply_scalable_init now targets output projections only
              (attention.dense + mlp.dense_4h_to_h), not all weight matrices
  [model #4]  prelude_norm: LayerNorm on prelude output before first injection
              (Parcae §4.1, App. J — prevents late-stage state explosion at 1B+)
  [model #5]  dt_bias inverse-softplus init targeting decay ≈ 0.447
              (Parcae §4.1 — replaces flat -2.0 init that gave decay ≈ 0.881)
  [model #12] per-sequence depth sampling via _loop_per_sequence
              (Parcae §4.2 — each sequence in the batch gets its own sampled T)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CortexConfig:
    # Model identity
    base_model_name: str = "EleutherAI/pythia-160m"

    # Layer split — must sum to total Pythia layer count
    n_pre:  int = 2
    n_loop: int = 8
    n_coda: int = 2

    # Hidden dim (must match base model)
    hidden_size: int = 768

    # Recurrence
    mean_recurrence:     int = 8   # mean T (log-normal Poisson target)
    mean_backprop_depth: int = 4   # k for TBPTT; should satisfy ⌈mean_recurrence/2⌉

    # Stability additions (Parcae)
    prelude_norm:          bool = True   # LN on prelude output before injection (App. J)
    per_sequence_sampling: bool = True   # different T per sequence in batch (§4.2)

    # M_cross buffer (K=0 → baseline Run 1 with no buffer)
    memory_slots: int = 0
    memory_heads: int = 4

    # M_iter buffer (Run 2 only, K_iter=0 disables)
    memory_slots_iter: int = 0

    # R4 dual-role mitigation (Lu et al. 2025, arXiv:2507.02199 / Parcae §C matrix)
    # Projects h_T into a separate embedding space before M_cross write, decoupling
    # the buffer-write path from the Coda-prediction path.  Identity-initialized so
    # it is a no-op at step 0.  Only active when memory_slots > 0.
    h_T_proj: bool = True

    # Scalable init: divide loop output-projection weights by √mean_T at construction.
    # Correct for from-scratch training (Parcae); should be disabled when retrofitting
    # from pretrained weights because it cripples the pretrained loop layers from step 0,
    # causing loss >> random until the coda re-adapts.
    scalable_init: bool = True

    # h₀ initialisation strategy:
    #   "random"  — TruncNormal(0, 1/√D): Parcae §4.1, best gradient diversity for
    #               from-scratch training.
    #   "z0"      — clone of pre-block output: preserves pretrained representations,
    #               correct for retrofitting (McLeish et al.).
    h0_init: str = "random"

    # Retrofit layer surgery: index in the pretrained model's layer list where the
    # loop block begins.  None = contiguous assignment (all layers used in order).
    # When set, pre = layers[:n_pre], loop = layers[loop_start_idx:loop_start_idx+n_loop],
    # coda = last n_coda layers.  Intermediate layers are discarded.
    loop_start_idx: Optional[int] = None


LAYER_SPLITS: dict[str, tuple[int, int, int]] = {
    "EleutherAI/pythia-160m": (2, 8, 2),
    "EleutherAI/pythia-1b":   (4, 16, 4),
    "EleutherAI/pythia-410m": (3, 12, 3),
}

# Retrofit surgery splits: (n_pre, n_loop, n_coda, loop_start_idx)
# Pre = layers[0:n_pre], loop = layers[loop_start_idx:loop_start_idx+n_loop],
# coda = last n_coda layers.  Layers between pre and loop_start are discarded.
# For pythia-160m (12 layers): 2 pre + skip 2-5 + 4 loop (6-9) + 2 coda (10-11).
RETROFIT_SPLITS: dict[str, tuple[int, int, int, int]] = {
    "EleutherAI/pythia-160m": (2, 4, 2, 6),
}


# ---------------------------------------------------------------------------
# LTI Injection (Parcae §4.1 — DiagonalInjection)
# ---------------------------------------------------------------------------

def _init_dt_bias(tensor: torch.Tensor, decay_target: float = math.sqrt(1.0 / 5.0)) -> None:
    """
    Inverse-softplus init so that softplus(dt_bias) · exp(A_log=0) = 1 gives
    the target initial decay = exp(-softplus(dt_bias)).

    decay_target = sqrt(1/5) ≈ 0.447  (Parcae §4.1)

    Derivation:
      decay = exp(-softplus(dt))
      → softplus(dt) = -log(decay_target)            let x = -log(decay_target)
      → dt = softplus⁻¹(x) = x + log(-expm1(-x))
    """
    with torch.no_grad():
        x   = -math.log(decay_target)                 # ≈ 0.8047 for decay=0.447
        inv = x + math.log(-math.expm1(-x))           # inverse softplus ≈ 0.212
        tensor.fill_(inv)


class LTIInjection(nn.Module):
    """
    Stable input injection via a discrete LTI system (Parcae §4.1).

    h_{t+1} = exp(-dt·A) ⊙ h_t  +  dt · (z₀ @ B.T)

    Parameterization ensures ρ(Ā) < 1 for all parameter values:
      A  = exp(A_log)         > 0  always  (A_log ∈ ℝ, init = 0 → A = 1)
      dt = softplus(dt_bias)  > 0  always  (init → decay ≈ 0.447)
      decay = exp(-dt·A)      ∈ (0,1)  always

    B is identity-initialized; all three are excluded from weight decay.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.A_log   = nn.Parameter(torch.zeros(hidden_size))
        self.dt_bias = nn.Parameter(torch.empty(hidden_size))
        self.B       = nn.Parameter(torch.eye(hidden_size))

        _init_dt_bias(self.dt_bias)   # decay ≈ 0.447 at init

        # Excluded from weight decay and Muon Newton-Schulz (structural params)
        self.A_log._no_weight_decay   = True
        self.dt_bias._no_weight_decay = True
        self.B._no_weight_decay       = True

    def forward(self, h: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        dt    = F.softplus(self.dt_bias)               # (D,)
        decay = torch.exp(-dt * torch.exp(self.A_log)) # (D,) in (0, 1)
        return h * decay + dt * (z0 @ self.B.T)

    @torch.no_grad()
    def spectral_norm(self) -> float:
        """Max decay factor — should stay < 1 throughout training."""
        dt = F.softplus(self.dt_bias)
        return torch.exp(-dt * torch.exp(self.A_log)).max().item()

    @torch.no_grad()
    def contraction_factor(self) -> float:
        """Mean decay factor across hidden dimensions."""
        dt = F.softplus(self.dt_bias)
        return torch.exp(-dt * torch.exp(self.A_log)).mean().item()


# ---------------------------------------------------------------------------
# LSTM Buffer (LM2-style, arXiv 2502.06049)
# ---------------------------------------------------------------------------

class LSTMBuffer(nn.Module):
    """
    K-slot LSTM-gated memory buffer.

    Write: LSTM-style gated update where both gates receive a combined signal
           from the pooled input *and* the current buffer state (memory feedback).
           Matches LM2 create_gates: gate_in = f(inputs) + g(tanh(memory)).
           One combined projection outputs 2·D, split evenly into ig/fg.

    Read:  cross-attention — sequence tokens query the K buffer slots —
           result additively injected into the loop state.
           (Cleaner than LM2's forced-square design; no seq_len==K constraint.)

    Key LM2 §3 details preserved
    ------------------------------
    - Memory feedback: tanh(buffer) projected into gate signal each write.
    - Combined gate projection split: both gates share the same intermediate
      representation, coupling their retain/update decisions (LM2 create_gates).
    - Forget gate bias +1.0: biases toward retention at init (LM2 §3.3).
    - out_proj zero-init: read injection is a no-op at step 0, preserving the
      pretrained transformer output at the start of training.
    """

    def __init__(self, hidden_size: int, n_slots: int, n_heads: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_slots     = n_slots
        self.n_heads     = n_heads
        self.head_dim    = hidden_size // n_heads
        assert hidden_size % n_heads == 0

        # ── Write path ───────────────────────────────────────────────────────
        # Both gates are derived from the same combined signal (LM2 create_gates).
        # gate_in = gate_proj_in(h_pool) + gate_proj_mem(tanh(buffer))
        # The 2·D output is split in half: first D → input gate, second D → forget gate.
        self.gate_proj_in  = nn.Linear(hidden_size, hidden_size * 2)          # input side
        self.gate_proj_mem = nn.Linear(hidden_size, hidden_size * 2)          # memory side
        self.forget_bias   = nn.Parameter(torch.ones(1))   # +1.0 per LM2 §3.3
        self.input_bias    = nn.Parameter(torch.zeros(1))

        # Candidate refinement (LM2 attend_over_memory: LN → MLP → LN).
        # cand_proj maps the pooled input into the K-slot space (analogous to LM2's
        # attention step that mixes input information into memory space).  The result
        # is then residually combined with the current buffer and refined through a
        # 2-layer ReLU MLP with LayerNorm on both ends — matching the structure of
        # attend_over_memory exactly (attended_memory_layernorm + MLP + layernorm2).
        self.cand_proj  = nn.Linear(hidden_size, n_slots * hidden_size)
        self.cand_ln1   = nn.LayerNorm(hidden_size)
        self.cand_mlp1  = nn.Linear(hidden_size, hidden_size)
        self.cand_mlp2  = nn.Linear(hidden_size, hidden_size)
        self.cand_ln2   = nn.LayerNorm(hidden_size)

        # ── Read path ────────────────────────────────────────────────────────
        self.q_proj  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # additive injection starts at zero

    def write(self, h_T: torch.Tensor, buffer: torch.Tensor) -> torch.Tensor:
        """
        h_T   : [B, S, D]  — final Loop state
        buffer: [B, K, D]  — current K-slot buffer
        Returns updated buffer [B, K, D].

        Gate computation (LM2 create_gates):
          gate_in  = gate_proj_in(mean(h_T)) [B,K,2D]
                   + gate_proj_mem(tanh(buffer)) [B,K,2D]     ← both gates, combined signal
          ig, fg   = chunk(sigmoid(gate_in + bias), 2, dim=-1)     each [B, K, D]

        Candidate (LM2 attend_over_memory: LN → 2-layer ReLU MLP → LN):
          cand_base = cand_proj(mean(h_T)).view(B, K, D)
          cand      = LN1(buffer + cand_base)              ← residual from current buffer
          cand      = LN2(cand + mlp2(relu(mlp1(cand))))  ← MLP refinement
          candidate = tanh(cand)

          new_buf  = fg ⊙ buffer  +  ig ⊙ candidate
        """
        B, S, D = h_T.shape
        K = self.n_slots

        # Pool sequence → single summary vector [B, D]
        h_pool = h_T.mean(dim=1)

        # Input side: [B, 2D] → expand to [B, K, 2D]
        in_signal  = self.gate_proj_in(h_pool).unsqueeze(1).expand(-1, K, -1)  # [B, K, 2D]

        # Memory side: tanh(buffer) [B, K, D] → [B, K, 2D]  (LM2 line 281: tanh before proj)
        mem_signal = self.gate_proj_mem(torch.tanh(buffer))                    # [B, K, 2D]

        # Combined → split into ig/fg (both gates share the same intermediate repr)
        combined = in_signal + mem_signal                                       # [B, K, 2D]
        ig_logits, fg_logits = combined.chunk(2, dim=-1)                       # each [B, K, D]

        ig = torch.sigmoid(ig_logits + self.input_bias)                        # [B, K, D]
        fg = torch.sigmoid(fg_logits + self.forget_bias)                       # [B, K, D]

        # Candidate refinement (LM2 attend_over_memory: LN → MLP → LN).
        # cand_proj mixes input information into slot space, then the residual with
        # the current buffer is refined through a 2-layer MLP — structurally matching:
        #   memory = LN(memory + attended_memory)      ← LM2 attended_memory_layernorm
        #   memory = LN(memory + relu(fc2(relu(fc1(memory)))))  ← LM2 layernorm2
        cand_base = self.cand_proj(h_pool).view(B, K, D)              # [B, K, D]
        cand      = self.cand_ln1(buffer + cand_base)                  # LN1 with buffer residual
        mlp_out   = self.cand_mlp2(F.relu(self.cand_mlp1(cand)))
        candidate = torch.tanh(self.cand_ln2(cand + mlp_out))         # LN2 with MLP residual

        return fg * buffer + ig * candidate

    def read(self, h: torch.Tensor, buffer: torch.Tensor) -> torch.Tensor:
        """
        h     : [B, S, D]  — current queries
        buffer: [B, K, D]  — K-slot memory (keys/values)
        Returns [B, S, D] delta to add into h.
        """
        B, S, D = h.shape
        K, nh, hd = self.n_slots, self.n_heads, self.head_dim

        q = self.q_proj(h).view(B, S, nh, hd).transpose(1, 2)
        k = self.k_proj(buffer).view(B, K, nh, hd).transpose(1, 2)
        v = self.v_proj(buffer).view(B, K, nh, hd).transpose(1, 2)

        attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# CortexGPT
# ---------------------------------------------------------------------------

class CortexGPT(nn.Module):
    """
    Pythia (GPTNeoX) retrofitted into Pre / Loop / Coda with Parcae LTI injection.

    Forward call
    ------------
    out = model(input_ids, labels=labels, num_steps=...)
    out["loss"], out["log_ppl"], out["logits"]

    num_steps variants
    ------------------
    torch.Tensor([n, k])       — per-batch: all sequences share the same T=n+k
    list of (n_i, k_i) len B   — per-sequence: each sequence gets its own T
    None                        — eval mode: T = mean_recurrence, all grad
    """

    def __init__(self, base_model, config: CortexConfig) -> None:
        super().__init__()
        self.config = config
        neox = base_model.gpt_neox

        n_pre, n_loop, n_coda = config.n_pre, config.n_loop, config.n_coda
        n_total = len(neox.layers)
        loop_start = config.loop_start_idx

        if loop_start is None:
            # Contiguous: every pretrained layer is assigned to a block.
            assert n_pre + n_loop + n_coda == n_total, (
                f"Layer split {n_pre}+{n_loop}+{n_coda}={n_pre+n_loop+n_coda} "
                f"≠ model layers {n_total}"
            )
            pre_idx  = list(range(0, n_pre))
            loop_idx = list(range(n_pre, n_pre + n_loop))
            coda_idx = list(range(n_pre + n_loop, n_total))
        else:
            # McLeish surgery: non-contiguous selection; layers between pre and
            # loop_start are discarded, as are any layers between loop end and coda.
            assert loop_start >= n_pre, (
                f"loop_start_idx {loop_start} must be ≥ n_pre {n_pre}"
            )
            assert loop_start + n_loop <= n_total - n_coda, (
                f"loop block [{loop_start}:{loop_start+n_loop}] overlaps coda "
                f"(last {n_coda} of {n_total})"
            )
            pre_idx  = list(range(0, n_pre))
            loop_idx = list(range(loop_start, loop_start + n_loop))
            coda_idx = list(range(n_total - n_coda, n_total))

        # ── Pythia components ───────────────────────────────────────────────
        self.embed_in    = neox.embed_in
        self.emb_dropout = neox.emb_dropout
        self.pre_layers  = nn.ModuleList([neox.layers[i] for i in pre_idx])
        self.loop_layers = nn.ModuleList([neox.layers[i] for i in loop_idx])
        self.coda_layers = nn.ModuleList([neox.layers[i] for i in coda_idx])
        self.final_norm  = neox.final_layer_norm
        self.embed_out   = base_model.embed_out

        # Transformers ≥4.47 moved RoPE computation out of individual layers.
        # position_embeddings=(cos,sin) must now be pre-computed at the model
        # level and passed down; position_ids is no longer accepted by layers.
        self.rotary_emb  = neox.rotary_emb

        # Exclude embedding tables from Newton-Schulz in Muon.
        # Embedding weights are 2D, so Muon._use_muon() would route them through
        # Newton-Schulz, which destroys the geometry of the embedding space by
        # forcing all token rows to be mutually orthogonal.  Sparse/local
        # Adam updates are correct for embeddings; flag both tables here.
        self.embed_in.weight._no_weight_decay  = True
        self.embed_out.weight._no_weight_decay = True

        # ── New modules ─────────────────────────────────────────────────────
        self.lti = LTIInjection(config.hidden_size)

        # Prelude norm: stabilises prelude output before it enters B in the LTI.
        # Without this, a single prelude layer can grow its output norm unboundedly
        # late in training, causing state explosion at first injection.
        # Ref: Parcae App. J (1.3B failure mode diagnosis).
        self.ln_prelude: Optional[nn.LayerNorm] = (
            nn.LayerNorm(config.hidden_size) if config.prelude_norm else None
        )

        self.m_cross: Optional[LSTMBuffer] = (
            LSTMBuffer(config.hidden_size, config.memory_slots, config.memory_heads)
            if config.memory_slots > 0 else None
        )
        self.m_iter: Optional[LSTMBuffer] = (
            LSTMBuffer(config.hidden_size, config.memory_slots_iter, config.memory_heads)
            if config.memory_slots_iter > 0 else None
        )

        # R4 dual-role mitigation: separate linear projection applied to h_T before
        # writing to M_cross.  The Coda always receives the raw h_T; only the buffer
        # write path sees the projected representation.  Identity-initialized so the
        # projection is a no-op at step 0, letting pretrained weights stabilise first.
        # Analogous to Parcae's C matrix (parcae.py:33-39) but applied to the buffer
        # write path rather than the Coda input.  Only created when memory_slots > 0.
        self.h_T_proj: Optional[nn.Linear] = None
        if config.h_T_proj and config.memory_slots > 0:
            self.h_T_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            nn.init.eye_(self.h_T_proj.weight)

        # Scalable init on loop block output projections (from-scratch only)
        if config.scalable_init:
            self._apply_scalable_init()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _apply_scalable_init(self) -> None:
        """
        Divide Loop block *output projection* weights by √mean_T.

        Only targets the residual-stream contributions:
          attention.dense       — attention output projection
          mlp.dense_4h_to_h     — MLP output projection (down-proj)

        QKV projections, MLP up-projections, biases, and LayerNorm params
        are intentionally left at their pretrained values.

        Ref: Huginn §3 / Takase et al. 2023 (scalable init) /
             McLeish et al. 2025 (monkeypatch std).
        """
        scale = math.sqrt(self.config.mean_recurrence)
        OUTPUT_PROJ_KEYS = {"attention.dense", "mlp.dense_4h_to_h"}
        for layer in self.loop_layers:
            for name, param in layer.named_parameters():
                # name: e.g. "attention.dense.weight"
                parent = ".".join(name.split(".")[:-1])
                if parent in OUTPUT_PROJ_KEYS and name.endswith(".weight") and param.dim() >= 2:
                    with torch.no_grad():
                        param.div_(scale)

    # ------------------------------------------------------------------
    # Attention mask
    # ------------------------------------------------------------------

    def _build_attn_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,  # kept for API compat; not used — bool mask avoids dtype constraints
    ) -> torch.Tensor:
        # Boolean causal mask: True=attend, False=mask.
        # SDPA accepts bool without dtype constraints, avoiding query-dtype mismatch
        # when apply_rotary_pos_emb upcasts queries to float32 internally.
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask4d = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        if attention_mask is not None and not attention_mask.all():
            pad    = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            mask4d = mask4d & pad  # [B, 1, S, S]

        return mask4d

    # ------------------------------------------------------------------
    # Layer forward
    # ------------------------------------------------------------------

    def _layer_fwd(self, x, layer, attn_mask, pos_emb):
        out = layer(x, attention_mask=attn_mask, position_embeddings=pos_emb)
        # Transformers ≥4.47 GPTNeoXLayer.forward returns a plain tensor, not a tuple.
        # Older versions returned (hidden_states, ...). Handle both.
        return out[0] if isinstance(out, tuple) else out

    # ------------------------------------------------------------------
    # Single loop iteration
    # ------------------------------------------------------------------

    def _loop_iter(
        self,
        h: torch.Tensor,
        z0: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_emb: tuple,
        cross_buf: Optional[torch.Tensor],
        iter_buf:  Optional[torch.Tensor],
    ) -> torch.Tensor:
        # LTI: stable combination of prior state and current encoding
        h = self.lti(h, z0)

        # Buffer injections (additive into first loop layer input)
        if self.m_cross is not None and cross_buf is not None:
            h = h + self.m_cross.read(h, cross_buf)
        if self.m_iter is not None and iter_buf is not None:
            h = h + self.m_iter.read(h, iter_buf)

        for layer in self.loop_layers:
            h = self._layer_fwd(h, layer, attn_mask, pos_emb)
        return h

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def _init_state(self, reference: torch.Tensor) -> torch.Tensor:
        """
        h₀ ~ TruncNormal(0, 1/√D) for from-scratch training (Parcae §4.1).

        "z0" mode clones the prelude output; used only when retrofitting from
        pretrained weights (retrofit/combined training modes).
        """
        if self.config.h0_init == "z0":
            return reference.detach().clone()
        # "random": TruncNormal(0, 1/√D) — Parcae §4.1
        std = 1.0 / math.sqrt(self.config.hidden_size)
        h   = torch.empty_like(reference)
        nn.init.trunc_normal_(h, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return h

    # ------------------------------------------------------------------
    # Batched loop (all sequences share the same n, k)
    # ------------------------------------------------------------------

    def _loop_batched(
        self,
        h: torch.Tensor,
        z0: torch.Tensor,
        attn4d: torch.Tensor,
        pos_emb: tuple,
        n_nograd: int,
        k_grad: int,
        m_cross_in: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B = h.shape[0]
        iter_buf: Optional[torch.Tensor] = None
        if self.m_iter is not None:
            iter_buf = h.new_zeros(B, self.config.memory_slots_iter, self.config.hidden_size)

        with torch.no_grad():
            for _ in range(n_nograd):
                h = self._loop_iter(h, z0, attn4d, pos_emb, m_cross_in, iter_buf)
                if self.m_iter is not None:
                    iter_buf = self.m_iter.write(h, iter_buf)

        for _ in range(k_grad):
            h = self._loop_iter(h, z0, attn4d, pos_emb, m_cross_in, iter_buf)
            if self.m_iter is not None:
                iter_buf = self.m_iter.write(h, iter_buf)

        return h

    # ------------------------------------------------------------------
    # Per-sequence loop (each sequence gets its own sampled T)
    # ------------------------------------------------------------------

    def _loop_per_sequence(
        self,
        h: torch.Tensor,
        z0: torch.Tensor,
        attn4d: torch.Tensor,
        pos_emb: tuple,
        per_seq_steps: list[tuple[int, int]],
        m_cross_in: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Each sequence i gets (n_i, k_i) independently sampled by the caller.

        Sequences are processed individually to preserve per-sample gradient
        scoping. This faithfully approximates E_{T~Λ}[loss] within each batch
        rather than collapsing to a single T per micro-batch.

        Ref: Parcae §4.2, Appendix G — Table 8 shows 4× PPL improvement at T=1
             compared to per-batch sampling for a 100M model.
        """
        B     = h.shape[0]
        h_out = torch.zeros_like(h)   # zeros: no uninitialized memory if loop errors
        cos, sin = pos_emb

        for i in range(B):
            n_i, k_i = per_seq_steps[i]
            h_i   = h[i:i+1]
            z0_i  = z0[i:i+1]
            # attn4d is [1,1,S,S] if no padding, [B,1,S,S] if padded
            attn_i  = attn4d[i:i+1] if attn4d.shape[0] == B else attn4d
            # cos/sin are [B,S,head_dim] or [1,S,head_dim] (broadcast)
            pos_emb_i = (cos[i:i+1] if cos.shape[0] == B else cos,
                         sin[i:i+1] if sin.shape[0] == B else sin)
            cross_i = m_cross_in[i:i+1] if m_cross_in is not None else None

            iter_buf_i: Optional[torch.Tensor] = None
            if self.m_iter is not None:
                iter_buf_i = h_i.new_zeros(
                    1, self.config.memory_slots_iter, self.config.hidden_size
                )

            with torch.no_grad():
                for _ in range(n_i):
                    h_i = self._loop_iter(h_i, z0_i, attn_i, pos_emb_i, cross_i, iter_buf_i)
                    if self.m_iter is not None:
                        iter_buf_i = self.m_iter.write(h_i, iter_buf_i)

            for _ in range(k_i):
                h_i = self._loop_iter(h_i, z0_i, attn_i, pos_emb_i, cross_i, iter_buf_i)
                if self.m_iter is not None:
                    iter_buf_i = self.m_iter.write(h_i, iter_buf_i)

            h_out[i] = h_i[0]

        return h_out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   Optional[torch.Tensor] = None,
        position_ids:     Optional[torch.Tensor] = None,
        labels:           Optional[torch.Tensor] = None,
        # Per-batch:     torch.Tensor([n, k])
        # Per-sequence:  list[(n_i, k_i)] length == batch_size
        # None:          eval — T = mean_recurrence
        num_steps:        Optional[Union[torch.Tensor, list]] = None,
        m_cross_in:       Optional[torch.Tensor] = None,
        return_m_cross:   bool = False,
    ) -> dict:
        B, S   = input_ids.shape
        device = input_ids.device

        # ── Parse num_steps → per_seq_steps: list[(n_i, k_i)] ───────────────
        if num_steps is None:
            per_seq_steps = [(0, self.config.mean_recurrence)] * B
        elif isinstance(num_steps, list):
            assert len(num_steps) == B
            per_seq_steps = [(int(n), int(k)) for n, k in num_steps]
        else:
            n_nograd = int(num_steps[0])
            k_grad   = int(num_steps[1])
            per_seq_steps = [(n_nograd, k_grad)] * B

        # Use faster batched loop when all sequences share the same (n, k)
        use_per_seq = (self.config.per_sequence_sampling
                       and len(set(per_seq_steps)) > 1)

        if attention_mask is None:
            attention_mask = input_ids.new_ones(B, S)
        if position_ids is None:
            position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        dtype  = self.embed_in.weight.dtype
        attn4d = self._build_attn_mask(attention_mask, S, device, dtype)

        # ── Token embedding ──────────────────────────────────────────────────
        x = self.emb_dropout(self.embed_in(input_ids))

        # Pre-compute RoPE embeddings once for the whole forward pass.
        # Transformers ≥4.47 removed position_ids support from individual layers;
        # each layer now expects a pre-computed (cos, sin) tuple passed as
        # position_embeddings.  x is used only for dtype/device resolution.
        pos_emb = self.rotary_emb(x, position_ids)   # (cos, sin), each [B, S, head_dim]

        # ── Pre block ────────────────────────────────────────────────────────
        for layer in self.pre_layers:
            x = self._layer_fwd(x, layer, attn4d, pos_emb)

        # Prelude norm: normalise prelude output before it enters the LTI B term.
        # The prelude output scale grows during training; without normalisation the
        # first injection can cause state explosion late in runs (Parcae App. J).
        z0 = self.ln_prelude(x) if self.ln_prelude is not None else x

        # ── Loop block ───────────────────────────────────────────────────────
        h = self._init_state(z0)

        if use_per_seq:
            h = self._loop_per_sequence(h, z0, attn4d, pos_emb, per_seq_steps, m_cross_in)
        else:
            n_nograd, k_grad = per_seq_steps[0]
            h = self._loop_batched(h, z0, attn4d, pos_emb, n_nograd, k_grad, m_cross_in)

        h_T = h

        # Write h_T into M_cross buffer.
        # Apply the R4 projection before write so the buffer path and the Coda path
        # operate on independent representations (Lu et al. 2025 / Parcae C matrix).
        # The Coda below always receives the unmodified h_T.
        new_m_cross: Optional[torch.Tensor] = None
        if self.m_cross is not None:
            h_T_for_write = self.h_T_proj(h_T) if self.h_T_proj is not None else h_T
            if m_cross_in is None:
                m_cross_in = h_T.new_zeros(B, self.config.memory_slots, self.config.hidden_size)
            new_m_cross = self.m_cross.write(h_T_for_write, m_cross_in)

        # ── Coda block ───────────────────────────────────────────────────────
        x = h_T
        for layer in self.coda_layers:
            x = self._layer_fwd(x, layer, attn4d, pos_emb)
        x = self.final_norm(x)

        # ── Loss ─────────────────────────────────────────────────────────────
        logits = self.embed_out(x).float()

        loss = log_ppl = None
        if labels is not None:
            loss    = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
            )
            log_ppl = loss.detach().exp()

        out = {"loss": loss, "log_ppl": log_ppl, "logits": logits}
        if return_m_cross:
            out["m_cross"] = new_m_cross
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_cortex_gpt(
    model_name:         str          = "EleutherAI/pythia-160m",
    memory_slots:       int          = 0,
    memory_slots_iter:  int          = 0,
    trust_remote_code:  bool         = False,
    torch_dtype:        torch.dtype  = torch.bfloat16,
    device_map:         str          = "cpu",
    from_scratch:       bool         = True,
    scalable_init:      bool         = True,
    h0_init:            str          = "random",
    prelude_norm:       bool         = True,
    retrofit_surgery:   bool         = False,
) -> tuple[CortexGPT, CortexConfig]:
    from transformers import AutoModelForCausalLM, AutoConfig

    if retrofit_surgery:
        assert not from_scratch, "retrofit_surgery requires pretrained weights (from_scratch=False)"
        msplit = RETROFIT_SPLITS.get(model_name)
        if msplit is None:
            raise ValueError(f"Retrofit surgery not configured for {model_name!r}. Add to RETROFIT_SPLITS.")
        n_pre, n_loop, n_coda, loop_start_idx = msplit
    else:
        split = LAYER_SPLITS.get(model_name)
        if split is None:
            raise ValueError(f"Unknown model {model_name!r}. Add it to LAYER_SPLITS.")
        n_pre, n_loop, n_coda = split
        loop_start_idx = None

    if from_scratch:
        cfg_hf = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        base = AutoModelForCausalLM.from_config(cfg_hf).to(dtype=torch_dtype)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    hidden_size = base.gpt_neox.embed_in.embedding_dim

    cfg = CortexConfig(
        base_model_name   = model_name,
        n_pre             = n_pre,
        n_loop            = n_loop,
        n_coda            = n_coda,
        hidden_size       = hidden_size,
        memory_slots      = memory_slots,
        memory_slots_iter = memory_slots_iter,
        scalable_init     = scalable_init,
        h0_init           = h0_init,
        prelude_norm      = prelude_norm,
        loop_start_idx    = loop_start_idx,
    )

    model = CortexGPT(base, cfg)
    # The pretrained layers arrive in torch_dtype, but the new modules
    # (LTIInjection, LSTMBuffer, h_T_proj, ln_prelude) are float32 by default.
    # Cast the entire model so all parameters share the same dtype.
    model = model.to(dtype=torch_dtype)
    return model, cfg
