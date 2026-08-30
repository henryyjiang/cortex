# CortexGPT — the from-scratch line

Pre-training a **depth-recurrent language model with an episodic latent memory** from scratch at
~155M parameters, on a Pythia (GPTNeoX) backbone.

**Status: closed.** This line ran its five arms to ~5B tokens, was measured, and is kept as the
from-scratch row of the write-up. Active work moved to the retrofit line in the sibling repo
[cortex-finetune](https://github.com/henryyjiang/cortex-finetune), which converts a *pretrained*
OLMo-2-1B into the same shape instead of training one from nothing. The two repos are separate
codebases and do not share conventions — configuration here is `model.py` + argparse, and
recurrence schedules use the `num_steps=[(0, T)]` format.

## The architecture

`model.py` rebuilds a Pythia checkpoint's layers into **prelude / loop / coda**, where the loop
block is applied `T` times per forward pass, and adds two optional memory buffers:

| component | what it is |
|---|---|
| `LTIInjection` | Parcae diagonal linear-time-invariant injection, `rho(A) < 1` by construction |
| `LSTMBuffer` | LM2-style K-slot LSTM-gated memory, used for `M_cross` and `M_iter` |
| `CortexGPT` | the wrapper: Pythia layers + recurrence + optional buffers |

Layer splits: **160M** (12 layers) 3/6/3, **1B** (16 layers) 4/8/4, **2.8B** (32 layers) 8/16/8.

Two memories, doing different jobs:

- **`M_cross`** — carried *across* segments, so information can cross a context-window boundary.
  This is the episodic memory and the whole point of the project.
- **`M_iter`** — **per-position**: the sequence dim is folded into the batch dim (`[B*S, K, D]`) so
  each token position accumulates only its own loop states across the `T` iterations. The earlier
  mean-pool over the sequence leaked future-token information into past positions and was a bug,
  not a design.

Document boundaries are handled inside `forward()` from the segment's `eos_mask`: only positions up
to the first EOS read the carried state, only the suffix after the last EOS is pooled into the
write, and the ended document's buffer is excluded from the gated update. The train-loop's earlier
any-EOS full zeroing left the carried buffer approximately always zero on packed Pile data, where
almost every 2048-token segment contains an EOS.

## Training

`train.py` trains from scratch by default. Retrofit modes exist as diagnostics but were never the
primary workflow here.

```bash
python train.py --training_mode cortex --model_name EleutherAI/pythia-160m --mean_recurrence 8
```

```bash
torchrun --nproc_per_node=4 train.py --batch_size 64 --micro_batch_size 4
```

`--training_mode` selects the arm: `cortex` (from-scratch recurrent + Coconut-style K=0 carry when
`memory_slots=0`), `parcae` (same recipe, no carry), `pythia` (true non-recurrent transformer
baseline), `ccot` (looped additive carry), plus `retrofit` / `cortex_retrofit` diagnostics.
`--memory_slots K` switches the K=0 carry for the K-slot LM2 buffer.

Recipe details that matter: Muon (Newton-Schulz orthogonalisation) for weight matrices with
momentum warmup 0.85 -> 0.95, weight decay annealed to zero, Parcae Algorithm-4 recurrence sampling
with per-sequence depth, `mu_bwd = ceil(mu_rec / 2)` enforced through the curriculum, a linear
recurrence ramp `1 -> mean_recurrence`, a cross-chunk `M_cross` chain carried un-detached between
consecutive chunks so the *write* path trains, and nonfinite-gradient batches skipped rather than
allowed to corrupt optimizer state.

Data is streamed (`data.py`), so no corpus is downloaded in full: `EleutherAI/the_pile_deduplicated`
by default to match Pythia's pretraining distribution, optionally switching to
`HuggingFaceFW/fineweb-edu` for a higher-quality late phase via `--phase2_start_tokens`. Documents
are sharded rank x worker round-robin so no consumer sees duplicate data.

## Cluster jobs

Slurm scripts for PACE Phoenix, one per arm, each with a matching `_resume` script:

| script | arm |
|---|---|
| `job5b_pythia.sh` | non-recurrent Pythia baseline |
| `job5b_parcae.sh` | recurrent, no carry |
| `job5b.sh` | `cortex-5b` — recurrent, K=0 carry |
| `job5b2.sh` | `cortex-5b-k4` — recurrent, K=4 LM2 buffer |
| `job5b_ccot.sh` | looped additive carry |

Evaluation jobs are `job_eval_*.sh`; `submit_eval_easy_all.sh` submits all five arms under one
shared `EVAL_TAG` so they land in a single results root regardless of queue order.

## Evals

At ~155M scale HellaSwag / ARC-Challenge / GSM8K are still at chance, so the primary suite is the
"easy" one in `evals/eval_easy.py`: LAMBADA (greedy last-word accuracy), BLIMP (67 paradigms,
higher-log-prob sentence wins), SciQ, ARC-Easy and PIQA (mean per-token log-prob of each
continuation).

```bash
python evals/eval_easy.py --checkpoint runs/cortex-5b/checkpoint_XXXXXXX/checkpoint.pt --T 8 \
    --tasks lambada blimp sciq arc_easy piqa --out_dir eval_results/easy/cortex-5b
```

```bash
python evals/aggregate_easy.py eval_results/easy_<tag>
```

`evals/` also holds the long-context runners (`eval_babilong.py`, `eval_longmemeval.py`) and
`eval_gsm8k.py` / `eval_multiple_choice.py`.

## Results

Accuracy at ~5.0B tokens (`eval_results/easy_20260706/`), against the official
`pythia-160m-deduped` at step 3000 = 6.3B tokens (`eval_results/hfref_step3000/`):

| arm | LAMBADA | BLIMP | SciQ | ARC-Easy | PIQA |
|---|---|---|---|---|---|
| `pythia-160m-deduped@step3000` (reference, 6.3B tok) | 0.2243 | 0.7376 | 0.5130 | 0.2772 | 0.5936 |
| pythia-5b (our baseline) | **0.2961** | 0.7642 | **0.5380** | 0.3018 | 0.6099 |
| parcae-5b | 0.2843 | **0.7811** | 0.5130 | 0.3105 | 0.6137 |
| cortex-5b (K=0) | 0.2765 | 0.7762 | 0.4990 | **0.3175** | **0.6153** |
| cortex-5b-k4 (K=4) | 0.2820 | 0.7786 | 0.5170 | 0.3123 | 0.6061 |
| ccot-5b | 0.1977 | 0.7333 | 0.4510 | 0.2789 | 0.5843 |

What this line established:

- **All four non-`ccot` arms beat the official Pythia reference on LAMBADA / BLIMP / PIQA at fewer
  tokens** (5.0B vs 6.3B). The gap that made an earlier round look like a recipe deficit was a
  broken Muon update rate, since fixed.
- **No clear recurrence advantage on short-context evals at this scale.** The four are tightly
  bunched, and the differences between them are within what the suite can resolve here. The
  episodic-memory question is not answerable on these benchmarks — that is why the line moved to a
  purpose-built carry-ablation instrument in the retrofit repo.
- **`ccot` was the only arm that failed to converge**, and is kept as a motivating negative for
  scale-controlled carry: an ungated additive carry degrades every column.

Two caveats to carry forward whenever these numbers are quoted:

1. **The `ccot` arm is not Coconut.** Coconut replaces the embedding at `<|latent|>` slots,
   shift-by-one, KV-cached, as a finetune on CoT data. This arm is a looped-additive /
   universal-transformer baseline and should be labelled as one. A faithful Coconut comparison was
   planned as a separate finetune and never run.
2. **Long context was never validly measured on this line.** The long-context results here were
   produced by a harness with bugs that were only found later; do not quote them.

## Layout

```
model.py        CortexGPT: Pre/Loop/Coda, LTI injection, LSTM buffers
train.py        pre-training loop: Muon, curriculum, chunk chain, DDP, checkpointing
data.py         streaming packed dataloader with EOS-mask document boundaries
evals/          eval runners + aggregation
tests/          component and loading tests
job*.sh         Slurm scripts, one per arm, with resume variants
eval_results/   measurements, one directory per <suite>_<tag>
runs/           checkpoints (cluster-side; not in git)
```

```bash
python -m pytest tests/ -q
```

## References

Parcae (LTI injection, recurrence sampling, stability stack), LM2 (K-slot gated memory), Coconut
(latent carry), and McLeish et al., *Teaching Pretrained Language Models to Think Deeper with
Retrofitted Recurrence* (the Pre/Loop/Coda arrangement and the recurrence curriculum), which is the
basis of the sibling retrofit repo.
