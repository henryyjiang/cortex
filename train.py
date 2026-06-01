"""
CortexGPT training script — Stage 1 (healing) + Stage 2 pre-training.

Key features
  - Muon optimizer (Newton-Schulz orthogonalisation for weight matrices)
  - Muon momentum warmup: 0.85 → 0.95 over 300 steps  (Parcae train.py)
  - Weight decay annealed to zero over training          (Parcae train.py)
  - Parcae Algorithm 4 sampler: sample T first, derive n/k (App. H)
  - Per-sequence depth sampling (each sequence gets own T) (Parcae §4.2)
  - µbwd = ⌈µrec/2⌉ enforced and tracked through curriculum (Parcae App. I)
  - Recurrence depth curriculum: linear ramp 1 → mean_recurrence (McLeish §4.2)
  - Skip-nonfinite-grads: bad batches log and skip, don't corrupt opt state
  - Two-phase training: phase 1 (healing on Pile), phase 2 (mixed data)
  - WandB logging, periodic checkpoint save/resume
  - Single-GPU and multi-GPU (DDP) support

Usage (single GPU):
    python train.py

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 train.py --batch_size 64 --micro_batch_size 4

Key flags:
    --model_name              EleutherAI/pythia-160m
    --mean_recurrence         8      (target mean T; µbwd set automatically)
    --curriculum_steps        0      (0 = no ramp, start at mean_recurrence)
    --phase2_start_tokens     0      (0 = no phase switch; e.g. 25_000_000_000)
    --phase2_dataset          HuggingFaceFW/fineweb-edu
    --lr                      3e-4   (Muon param LR; AdamW fallback uses same)
    --weight_decay            0.1    (anneals to 0 over training)
    --muon_momentum           0.95
    --muon_momentum_warmup    300    (steps to ramp from 0.85 to muon_momentum)
    --max_tokens              30_000_000_000

Changes from v1
---------------
  [train #6]  Muon optimizer replaces AdamW for weight matrices
  [train #7]  Muon momentum warmup (0.85 → target over warmup_steps)
  [train #8]  Weight decay linearly annealed to 0 over training
  [train #9]  Skip-nonfinite-grads: non-finite grad_norm skips opt step
  [train #10] Recurrence curriculum: linear ramp of mean_recurrence
  [train #11] Parcae Algorithm 4 sampler (sample T, derive n/k — no truncation)
  [train #13] Two-phase training: phase2_start_tokens triggers dataset switch
  [train #14] µbwd = ⌈µrec/2⌉ enforced; tracks curriculum automatically
"""
from __future__ import annotations

import argparse
import datetime
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

from model import CortexConfig, CortexGPT, LAYER_SPLITS, build_cortex_gpt
from data import build_dataloader, load_tokenizer


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------

def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz quintic iteration — approximates G · (G^T G)^{-1/2},
    i.e. the nearest orthogonal matrix to G in the Frobenius norm.

    5 steps suffice for near-orthogonality in practice.
    Ref: Keller Jordan 2024 (https://github.com/KellerJordan/Muon).
    """
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float() / (G.norm() + 1e-7)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ A @ X)
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum Orthogonalised by Newton-Schulz.

    For 2D+ weight matrices that are not SSM/structural (_no_weight_decay)
    and not embeddings: applies Newton-Schulz to the gradient, then a
    momentum-style update.  This effectively orthogonalises the update
    direction and removes the need to tune a separate LR per layer.

    For everything else (1D params, biases, LayerNorm scales, SSM params,
    and embedding tables tagged with _no_weight_decay): falls back to AdamW.

    Ref: Keller Jordan 2024; Prairie et al. (Parcae) 2025 with momentum
         warmup and weight decay annealing.

    Args
    ----
    params          : parameter groups
    lr              : learning rate applied to Muon updates (and AdamW fallback)
    momentum        : Muon momentum (subject to external warmup schedule)
    nesterov        : use Nesterov momentum (default True)
    ns_steps        : Newton-Schulz iterations (5 is sufficient)
    weight_decay    : applied to Muon params; SSM params always skip WD
    """

    def __init__(
        self,
        params,
        lr:           float = 3e-4,
        momentum:     float = 0.95,
        nesterov:     bool  = True,
        ns_steps:     int   = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _use_muon(p: nn.Parameter) -> bool:
        """True for 2D+ weight matrices that aren't structural/SSM params."""
        return p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd       = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if self._use_muon(p):
                    # ── Muon update ─────────────────────────────────────
                    state = self.state[p]
                    if "buf" not in state:
                        state["buf"] = torch.zeros_like(g)
                    buf = state["buf"]
                    buf.mul_(momentum).add_(g)

                    # Nesterov: lookahead along the momentum direction
                    g_eff = g.add(buf, alpha=momentum) if nesterov else buf

                    # Newton-Schulz orthogonalisation
                    update = _zeropower_via_newtonschulz5(g_eff, steps=ns_steps)
                    # Normalise scale: preserve RMS norm proportional to fan-ratio
                    scale = max(1.0, g.shape[0] / g.shape[1]) ** 0.5
                    update.mul_(scale)

                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(update, alpha=-lr)

                else:
                    # ── AdamW fallback (biases, LN, embeddings, SSM) ────
                    state = self.state[p]
                    if "step" not in state:
                        state["step"]       = 0
                        state["exp_avg"]    = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)

                    state["step"] += 1
                    # Parcae uses (0.95, 0.95) for the Adam fallback.
                    # beta1 is kept constant (not subject to Muon momentum warmup):
                    # the warmup intentionally dips to 0.85 to let the Muon gradient
                    # buffer start fresh, which is harmful for Adam's EMA stability.
                    beta1, beta2, eps = 0.95, 0.95, 1e-8

                    state["exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    t    = state["step"]
                    bias = (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
                    denom = state["exp_avg_sq"].sqrt().add_(eps)

                    # WD only for 2D+ non-flagged params; 1D params (biases,
                    # LayerNorm gamma/beta, scalar gates) and SSM params are exempt.
                    # 2D+ non-SSM params are already handled by the Muon path above.
                    if wd != 0.0 and not getattr(p, "_no_weight_decay", False) and p.ndim >= 2:
                        p.mul_(1.0 - lr * wd)
                    p.addcdiv_(state["exp_avg"], denom, value=-lr * bias)

        return loss


# ---------------------------------------------------------------------------
# Recurrence sampler — Parcae Algorithm 4
# ---------------------------------------------------------------------------

def sample_num_steps(
    optimizer_step:      int,
    mean_recurrence:     int,
    mean_backprop_depth: int,
) -> tuple[int, int]:
    """
    Parcae Algorithm 4 (App. H): sample total T first, then derive n and k.

        T ~ LogNormal-Poisson(µ_rec)      [heavy-tailed, mean ≈ µ_rec]
        n = max(T - µ_bwd, 0)             [no-grad steps]
        k = min(T,   µ_bwd)               [grad steps]

    This avoids the distributional mismatch in McLeish Algorithm 3, where
    setting k = µ_bwd as a constant truncates and compresses the forward-
    pass distribution, hurting generalisation to other test-time depths.

    Returns (n, k) as plain Python ints.
    """
    seed = 514229 + optimizer_step
    gen  = torch.Generator(device="cpu").manual_seed(seed % (2**31 - 1))

    sigma = 0.5
    mu    = math.log(max(mean_recurrence, 1)) - (sigma ** 2) / 2
    rate  = torch.zeros(1).log_normal_(mean=mu, std=sigma, generator=gen)
    T     = max(1, int(torch.poisson(rate, generator=gen).item()))

    n = max(T - mean_backprop_depth, 0)
    k = min(T, mean_backprop_depth)
    return n, k


def sample_batch_steps(
    optimizer_step:      int,
    batch_size:          int,
    mean_recurrence:     int,
    mean_backprop_depth: int,
) -> list[tuple[int, int]]:
    """
    Per-sequence variant: sample independent T_i for each sequence i.

    Returns list[(n_i, k_i)] of length batch_size.
    Each sequence gets a different depth from the same Λ, faithfully
    approximating E_{T~Λ}[loss] within the batch rather than collapsing
    to a single T per micro-batch.  Ref: Parcae §4.2, Appendix G.
    """
    steps = []
    for i in range(batch_size):
        seed = 514229 + optimizer_step * 10007 + i
        gen  = torch.Generator(device="cpu").manual_seed(seed % (2**31 - 1))

        sigma = 0.5
        mu    = math.log(max(mean_recurrence, 1)) - (sigma ** 2) / 2
        rate  = torch.zeros(1).log_normal_(mean=mu, std=sigma, generator=gen)
        T     = max(1, int(torch.poisson(rate, generator=gen).item()))

        n = max(T - mean_backprop_depth, 0)
        k = min(T, mean_backprop_depth)
        steps.append((n, k))
    return steps


# ---------------------------------------------------------------------------
# Recurrence curriculum
# ---------------------------------------------------------------------------

def get_current_mean_recurrence(
    step:           int,
    target_mean:    int,
    curriculum_steps: int,
) -> int:
    """
    Linear ramp from 1 → target_mean over curriculum_steps optimizer steps.
    After curriculum_steps, holds constant at target_mean.

    Ref: McLeish et al. §4.2 — scheduling mean recurrence is both data-
         and compute-efficient, reducing FLOPs for the same loss.
    """
    if curriculum_steps <= 0 or step >= curriculum_steps:
        return target_mean
    frac = step / curriculum_steps
    return max(1, round(1 + frac * (target_mean - 1)))


def enforce_mu_bwd(mean_recurrence: int) -> int:
    """
    µbwd = ⌈µrec / 2⌉

    Parcae Appendix I shows that growing µrec without growing µbwd
    proportionally degrades performance at high test-time T.
    µbwd = ⌈µrec/2⌉ is the validated choice across all their ablations.
    """
    return math.ceil(mean_recurrence / 2)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(
    step:           int,
    max_steps:      int,
    lr:             float,
    warmup_steps:   int,
    cooldown_steps: int,
    min_lr_ratio:   float = 0.1,
) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    stable_end = max_steps - cooldown_steps
    if step <= stable_end:
        return lr
    progress = (step - stable_end) / max(cooldown_steps, 1)
    return lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("CortexGPT pre-training")

    # Model
    p.add_argument("--model_name",           default="EleutherAI/pythia-160m")
    p.add_argument("--memory_slots",         type=int,   default=0)
    p.add_argument("--memory_slots_iter",    type=int,   default=0)
    p.add_argument("--training_mode",        default="cortex",
                   choices=["cortex", "parcae", "retrofit", "cortex_retrofit"],
                   help=(
                       "cortex          — (default) from scratch, scalable_init=True, h0=TruncNormal, "
                       "prelude_norm=True. Full Cortex architecture with Parcae training recipe. "
                       "parcae          — from scratch, same init as cortex. Pure Parcae baseline "
                       "(no Cortex-specific components); useful for ablations. "
                       "retrofit        — Pythia weights, scalable_init=False, h0=z0, prelude_norm=False, "
                       "McLeish layer surgery. Diagnostic only; not expected to converge at <100B tokens. "
                       "cortex_retrofit — Pythia weights + McLeish surgery, then Parcae+Cortex training: "
                       "scalable_init=False, h0=z0, prelude_norm=True."
                   ))

    # Recurrence — µbwd is derived automatically as ⌈µrec/2⌉
    p.add_argument("--mean_recurrence",      type=int,   default=8,
                   help="Target mean T; µbwd = ⌈T/2⌉ is set automatically")
    p.add_argument("--curriculum_steps",     type=int,   default=0,
                   help="Linear ramp mean_recurrence 1→target over N steps (0=off)")

    # Data — two-phase training
    p.add_argument("--seq_len",              type=int,   default=2048)
    p.add_argument("--max_tokens",           type=int,   default=30_000_000_000)
    p.add_argument("--data_buffer_size",     type=int,   default=10_000)
    p.add_argument("--num_data_workers",     type=int,   default=2)
    # Phase 2: set phase2_start_tokens > 0 to switch datasets mid-run
    p.add_argument("--phase2_start_tokens",  type=int,   default=0,
                   help="Switch to phase2_dataset after this many tokens (0=off)")
    p.add_argument("--phase2_dataset",       default="HuggingFaceFW/fineweb-edu",
                   help="HF dataset name for phase 2 (healing→general transition)")
    p.add_argument("--phase2_text_column",   default="text")

    # Optimiser — Muon
    p.add_argument("--lr",                   type=float, default=3e-4)
    p.add_argument("--weight_decay",         type=float, default=0.1,
                   help="Initial WD; annealed to 0 over training (Parcae)")
    p.add_argument("--muon_momentum",        type=float, default=0.95)
    p.add_argument("--muon_momentum_warmup", type=int,   default=300,
                   help="Ramp Muon momentum from 0.85 → muon_momentum over N steps")
    p.add_argument("--grad_clip",            type=float, default=1.0)
    p.add_argument("--warmup_ratio",         type=float, default=0.01)
    p.add_argument("--cooldown_ratio",       type=float, default=0.1)

    # Batch
    p.add_argument("--batch_size",           type=int,   default=512,
                   help="Effective batch size in tokens across all GPUs + grad accum")
    p.add_argument("--micro_batch_size",     type=int,   default=4,
                   help="Per-GPU per-step sequences")

    # Infrastructure
    p.add_argument("--out_dir",              default="runs/cortex-160m")
    p.add_argument("--resume_path",          default=None)
    p.add_argument("--save_interval",        type=int,   default=1000)
    p.add_argument("--log_interval",         type=int,   default=10)
    p.add_argument("--seed",                 type=int,   default=42)
    p.add_argument("--dtype",                default="bfloat16",
                   choices=["float32", "bfloat16"])
    p.add_argument("--compile",              action="store_true")
    p.add_argument("--wandb_project",        default="cortex-gpt")
    p.add_argument("--wandb_disabled",       action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.distributed.init_process_group(
            backend="nccl", rank=rank, world_size=world_size,
            timeout=datetime.timedelta(hours=2),
        )
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def is_main(rank: int) -> bool:
    return rank == 0


def teardown_distributed() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    out_dir:        Path,
    step:           int,
    model:          nn.Module,
    optimizer:      torch.optim.Optimizer,
    scheduler_state: dict,
    cfg:            CortexConfig,
    total_tokens:   int,
) -> None:
    ckpt_dir = out_dir / f"checkpoint_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "step":         step,
            "model":        unwrapped.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler_state,
            "total_tokens": total_tokens,
            "config":       vars(cfg),
        },
        ckpt_dir / "checkpoint.pt",
    )
    print(f"[step {step}] Saved -> {ckpt_dir}")


def load_checkpoint(
    resume_path: str,
    model:       nn.Module,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
) -> tuple[int, int, dict]:
    ckpt = torch.load(os.path.join(resume_path, "checkpoint.pt"), map_location=device)
    unwrapped = model.module if isinstance(model, DDP) else model
    unwrapped.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from step {ckpt['step']}, {ckpt['total_tokens']:,} tokens")
    return ckpt["step"], ckpt["total_tokens"], ckpt.get("scheduler", {})


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    rank, world_size, device = setup_distributed()
    main = is_main(rank)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    weight_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    out_dir = Path(args.out_dir)
    if main:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    if main:
        print(f"Loading {args.model_name} ...")
    _mode_cfg = {
        # (from_scratch, scalable_init, h0_init, prelude_norm, retrofit_surgery)
        # cortex/parcae: from-scratch with Parcae recipe (scalable_init + prelude_norm).
        # retrofit modes: pretrained weights + McLeish surgery; prelude_norm=False for
        # plain retrofit (disrupts pretrained dist), True for cortex_retrofit which
        # pairs surgery with the full Parcae stability stack.
        "cortex":          (True,  True,  "random", True,  False),
        "parcae":          (True,  True,  "random", True,  False),
        "retrofit":        (False, False, "z0",     False, True),
        "cortex_retrofit": (False, False, "z0",     True,  True),
    }
    _from_scratch, _scalable_init, _h0_init, _prelude_norm, _retrofit_surgery = _mode_cfg[args.training_mode]

    model, cfg = build_cortex_gpt(
        model_name        = args.model_name,
        memory_slots      = args.memory_slots,
        memory_slots_iter = args.memory_slots_iter,
        torch_dtype       = weight_dtype,
        device_map        = str(device),
        from_scratch      = _from_scratch,
        scalable_init     = _scalable_init,
        h0_init           = _h0_init,
        prelude_norm      = _prelude_norm,
        retrofit_surgery  = _retrofit_surgery,
    )
    cfg.mean_recurrence = args.mean_recurrence
    # µbwd = ⌈µrec/2⌉ enforced at init (Parcae App. I)
    cfg.mean_backprop_depth = enforce_mu_bwd(args.mean_recurrence)

    model = model.to(device=device, dtype=weight_dtype)

    if world_size > 1:
        model = DDP(model, device_ids=[device], find_unused_parameters=False,
                    gradient_as_bucket_view=True)
    if args.compile:
        model = torch.compile(model, fullgraph=False, dynamic=False)

    if main:
        n_params = sum(p.numel() for p in model.parameters())
        n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        split    = LAYER_SPLITS[args.model_name]
        print(f"Params: {n_params:,} total, {n_train:,} trainable")
        print(f"Layer split: Pre={split[0]}, Loop={split[1]}, Coda={split[2]}")
        print(f"mu_rec={cfg.mean_recurrence}, mu_bwd={cfg.mean_backprop_depth} "
              f"(= ceil({cfg.mean_recurrence}/2))")

    # ── Optimizer (Muon) ────────────────────────────────────────────────────
    optimizer = Muon(
        model.parameters(),
        lr           = args.lr,
        momentum     = args.muon_momentum,
        weight_decay = args.weight_decay,
    )

    # ── Token / step budget ─────────────────────────────────────────────────
    tokens_per_seq   = args.seq_len
    seqs_per_step    = args.batch_size // tokens_per_seq
    micro_seqs       = args.micro_batch_size
    grad_accum_steps = max(1, seqs_per_step // (micro_seqs * world_size))
    max_steps        = args.max_tokens // args.batch_size

    warmup_steps   = max(1, int(max_steps * args.warmup_ratio))
    cooldown_steps = max(1, int(max_steps * args.cooldown_ratio))

    if main:
        print(f"Effective batch: {args.batch_size} tokens = "
              f"{seqs_per_step} seqs x {tokens_per_seq} tok")
        print(f"Grad accum: {grad_accum_steps} | World size: {world_size}")
        print(f"Max steps: {max_steps:,} | Warmup: {warmup_steps} | Cooldown: {cooldown_steps}")
        if args.curriculum_steps > 0:
            print(f"Curriculum: mu_rec ramps 1->{args.mean_recurrence} over {args.curriculum_steps} steps")

    # ── Resume ──────────────────────────────────────────────────────────────
    start_step   = 0
    total_tokens = 0
    if args.resume_path:
        start_step, total_tokens, _ = load_checkpoint(
            args.resume_path, model, optimizer, device
        )

    # ── Tokeniser + DataLoaders ──────────────────────────────────────────────
    tokenizer = load_tokenizer(args.model_name)

    def _make_loader(dataset_name: str, text_column: str = "text"):
        return build_dataloader(
            tokenizer    = tokenizer,
            seq_len      = args.seq_len,
            batch_size   = args.micro_batch_size,
            num_workers  = args.num_data_workers,
            seed         = args.seed + rank * 1000,
            rank         = rank,
            world_size   = world_size,
            buffer_size  = args.data_buffer_size,
            dataset_name = dataset_name,
            text_column  = text_column,
        )

    # Phase 1 loader (Pile — healing / distribution matching)
    dataloader = _make_loader("EleutherAI/the_pile_deduplicated")
    data_iter  = iter(dataloader)

    # Phase 2 loader (lazy — only built if phase2_start_tokens > 0)
    phase2_loader: Optional[torch.utils.data.DataLoader] = None
    in_phase2 = False

    if args.phase2_start_tokens > 0 and main:
        print(f"Phase 2 will start at {args.phase2_start_tokens / 1e9:.1f}B tokens "
              f"using {args.phase2_dataset}")

    # ── WandB ───────────────────────────────────────────────────────────────
    if main and not args.wandb_disabled:
        import wandb
        wandb.init(
            project = args.wandb_project,
            name    = out_dir.name,
            config  = vars(args),
            resume  = "allow" if args.resume_path else None,
            dir     = str(out_dir),
        )

    # ── Training state ───────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad(set_to_none=True)

    step        = start_step
    accum_count = 0
    loss_accum  = 0.0
    step_t0     = time.monotonic()
    m_cross: Optional[torch.Tensor] = None

    # Momentum warmup tracking
    momentum_warmup_start = 0.85
    base_momentum         = args.muon_momentum

    while True:
        # ── Phase 2 dataset switch ───────────────────────────────────────────
        if (args.phase2_start_tokens > 0
                and not in_phase2
                and total_tokens >= args.phase2_start_tokens):
            if main:
                print(f"\n[step {step}] Switching to Phase 2 dataset: {args.phase2_dataset}")
            if phase2_loader is None:
                phase2_loader = _make_loader(args.phase2_dataset, args.phase2_text_column)
            data_iter = iter(phase2_loader)
            in_phase2 = True

        # ── Fetch next batch ─────────────────────────────────────────────────
        try:
            batch = next(data_iter)
        except StopIteration:
            # Restart current loader on exhaustion (infinite loop semantics)
            loader = phase2_loader if in_phase2 else dataloader
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)
        eos_mask  = batch["eos_mask"].to(device, non_blocking=True)  # [B, S] bool

        B = input_ids.shape[0]

        # ── Curriculum: current µrec and µbwd ───────────────────────────────
        current_mu_rec = get_current_mean_recurrence(
            step, args.mean_recurrence, args.curriculum_steps
        )
        current_mu_bwd = enforce_mu_bwd(current_mu_rec)

        # ── Sample recurrence depths ─────────────────────────────────────────
        unwrapped  = model.module if isinstance(model, DDP) else model
        has_buffer = (unwrapped.m_cross is not None)
        if unwrapped.config.per_sequence_sampling:
            # Parcae §4.2: each sequence in the batch gets its own T
            per_seq = sample_batch_steps(step, B, current_mu_rec, current_mu_bwd)
            num_steps_arg = per_seq   # list[(n_i, k_i)]
            # For logging: report the mean T across sequences
            mean_T_this_step = sum(n + k for n, k in per_seq) / len(per_seq)
            mean_k_this_step = sum(k for _, k in per_seq) / len(per_seq)
        else:
            n_nograd, k_grad = sample_num_steps(step, current_mu_rec, current_mu_bwd)
            num_steps_arg    = torch.tensor([n_nograd, k_grad])
            mean_T_this_step = float(n_nograd + k_grad)
            mean_k_this_step = float(k_grad)

        is_accumulating = (accum_count + 1) < grad_accum_steps
        ctx = (model.no_sync() if isinstance(model, DDP) and is_accumulating
               else nullcontext())

        with ctx:
            out = model(
                input_ids      = input_ids,
                labels         = labels,
                num_steps      = num_steps_arg,
                m_cross_in     = m_cross,
                return_m_cross = has_buffer,
            )
            loss = out["loss"] / grad_accum_steps
            loss.backward()

        # Carry m_cross within the gradient-accumulation window so consecutive
        # micro-batches share the buffer state.  Sequences that ended a document
        # this micro-batch get their buffer zeroed before the next micro-batch.
        if has_buffer:
            new_mc = out.get("m_cross")
            if new_mc is not None:
                new_mc = new_mc.detach()
                doc_ended = eos_mask.any(dim=1)          # [B] — any EOS in this chunk?
                if doc_ended.any():
                    new_mc = new_mc * (~doc_ended).view(-1, 1, 1).to(dtype=new_mc.dtype)
                m_cross = new_mc

        loss_accum  += out["loss"].detach().item()
        total_tokens += input_ids.numel() * world_size
        accum_count  += 1

        # ── Optimizer step ────────────────────────────────────────────────────
        if accum_count >= grad_accum_steps:
            lr_now = get_lr(step, max_steps, args.lr, warmup_steps, cooldown_steps)

            # Muon momentum warmup: 0.85 → target over muon_momentum_warmup steps
            if args.muon_momentum_warmup > 0:
                frac = min(step / args.muon_momentum_warmup, 1.0)
                current_momentum = momentum_warmup_start + frac * (base_momentum - momentum_warmup_start)
            else:
                current_momentum = base_momentum

            # Weight decay annealing: base_wd → 0 over max_steps (Parcae train.py)
            current_wd = args.weight_decay * max(0.0, 1.0 - step / max_steps)

            for pg in optimizer.param_groups:
                pg["lr"]           = lr_now
                pg["momentum"]     = current_momentum
                pg["weight_decay"] = current_wd

            # Gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            ).item()

            # Skip-nonfinite-grads: don't let a bad batch corrupt optimizer state
            if not math.isfinite(total_norm):
                if main:
                    print(f"[step {step}] Non-finite grad norm ({total_norm:.3f}), "
                          f"skipping optimizer step.")
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                loss_accum  = 0.0
                m_cross     = None
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step        += 1
            accum_count  = 0
            m_cross      = None   # reset buffer at each optimizer step boundary

            # ── Logging ───────────────────────────────────────────────────────
            if step % args.log_interval == 0 and main:
                elapsed  = time.monotonic() - step_t0
                avg_loss = loss_accum / (args.log_interval * grad_accum_steps)
                tok_per_s = (args.batch_size * args.log_interval) / elapsed

                print(
                    f"step={step:6d} | loss={avg_loss:.4f} | "
                    f"lr={lr_now:.2e} | wd={current_wd:.4f} | "
                    f"mom={current_momentum:.4f} | "
                    f"T={mean_T_this_step:.1f} k={mean_k_this_step:.1f} "
                    f"(mu_rec={current_mu_rec} mu_bwd={current_mu_bwd}) | "
                    f"grad={total_norm:.3f} | tok/s={tok_per_s:,.0f} | "
                    f"tokens={total_tokens/1e9:.2f}B"
                    + (" [phase2]" if in_phase2 else "")
                )

                if not args.wandb_disabled:
                    import wandb
                    raw_model = model.module if isinstance(model, DDP) else model
                    wandb.log({
                        "train/loss":            avg_loss,
                        "train/ppl":             math.exp(avg_loss),
                        "train/lr":              lr_now,
                        "train/weight_decay":    current_wd,
                        "train/momentum":        current_momentum,
                        "train/grad_norm":       total_norm,
                        "train/tok_per_s":       tok_per_s,
                        "train/total_tokens":    total_tokens,
                        "train/mean_T":          mean_T_this_step,
                        "train/mean_k":          mean_k_this_step,
                        "train/mu_rec":          current_mu_rec,
                        "train/mu_bwd":          current_mu_bwd,
                        "train/phase":           2 if in_phase2 else 1,
                        "lti/spectral_norm":     raw_model.lti.spectral_norm(),
                        "lti/contraction_factor": raw_model.lti.contraction_factor(),
                    }, step=step)

                loss_accum = 0.0
                step_t0    = time.monotonic()

            # ── Checkpoint ────────────────────────────────────────────────────
            if step % args.save_interval == 0 and main:
                raw_model = model.module if isinstance(model, DDP) else model
                save_checkpoint(out_dir, step, raw_model, optimizer,
                                {"step": step}, cfg, total_tokens)

            # ── Budget check ──────────────────────────────────────────────────
            if step >= max_steps:
                break

        if step >= max_steps:
            break

    # ── Final save ───────────────────────────────────────────────────────────
    if main:
        raw_model = model.module if isinstance(model, DDP) else model
        save_checkpoint(out_dir, step, raw_model, optimizer,
                        {"step": step}, cfg, total_tokens)
        print(f"Training complete. {total_tokens/1e9:.2f}B tokens, {step} steps.")
        if not args.wandb_disabled:
            import wandb
            wandb.finish()

    teardown_distributed()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
