# Easy-eval comparison: 5 in-house models vs official Pythia reference

**Date:** 2026-06-25
**Eval suite:** `evals/eval_easy.py` (LAMBADA, BLIMP, SciQ, ARC-Easy, PIQA)
**Metric conventions:** LAMBADA = greedy exact-match of all last-word tokens; BLIMP = grammatical
sentence must out-score ungrammatical (full-sentence log-prob); SciQ/ARC-Easy/PIQA = mean
per-token log-prob of each choice, highest wins.

## Provenance
- **In-house 5 models:** cluster job `job_eval_easy.sh`, log `logs/Report-10378332.out`,
  results dir `eval_results/easy_20260624_190111/`. Checkpoints all step 152584 (~5.0B tokens)
  EXCEPT ccot at step 150000 (~4.88B). pythia run at T=1; parcae/ccot/cortex/cortex-k4 at the
  saved mean_recurrence (T=8).
- **Official reference:** `EleutherAI/pythia-160m-deduped @ step3000` (~6.3B tokens), evaluated
  LOCALLY on an RTX 5070 Ti via `eval_easy.py --hf_model ... --revision step3000` (T=1).
  Results dir `eval_results/hfref_step3000/`. Same data family (deduped Pile) + tokenizer as the
  in-house runs, so the same harness applies apples-to-apples.
- **NOTE:** the `step2000` (~4.2B) bracket run did NOT complete — the local process died after
  model load (machine likely slept overnight). Only `step3000` is reported here.

## Results — accuracy

| Model (tokens) | LAMBADA | BLIMP | SciQ | ARC-Easy | PIQA |
|---|---|---|---|---|---|
| pythia-5b (5.0B) | 0.0159 | 0.6237 | 0.3250 | 0.2754 | 0.5441 |
| parcae-5b (5.0B) | 0.0353 | 0.6502 | 0.3450 | 0.2596 | 0.5528 |
| ccot-5b (4.9B) | 0.0006 | 0.5986 | 0.2330 | 0.2632 | 0.5392 |
| cortex-5b k0 (5.0B) | 0.0435 | 0.6388 | 0.3500 | 0.2737 | 0.5571 |
| cortex-5b-k4 (5.0B) | 0.0441 | 0.6559 | 0.3490 | 0.2684 | 0.5522 |
| **official Pythia (6.3B)** | **0.2243** | **0.7376** | **0.5130** | 0.2772 | **0.5936** |

Chance levels: LAMBADA ≈ 0 · BLIMP 0.50 · SciQ 0.25 · ARC-Easy 0.25 · PIQA 0.50.

Official raw counts (step3000): lambada 1156/5153 · blimp 49422/67000 · sciq 513/1000 ·
arc_easy 158/570 · piqa 1091/1838.

## Key findings

1. **At ~matched token count (~6.3B vs ~5B), official Pythia dominates every benchmark with
   signal.** vs the BEST in-house model: LAMBADA 0.224 vs 0.044 (~5×), BLIMP 0.738 vs 0.656
   (+8 pts, N=67k → highly significant), SciQ 0.513 vs 0.350 (+16 pts), PIQA 0.594 vs 0.557
   (+4 pts). vs in-house pythia specifically, LAMBADA is ~14× (0.224 vs 0.016).
2. **The gap is recipe-driven, not tokens or architecture.** Official is only ~1.26× the tokens,
   and it is itself a plain vanilla Pythia — yet it beats the architecturally richer recurrent
   models too. So the gap is the training recipe, not the Pre/Loop/Coda / memory machinery.
3. **ARC-Easy is tied at chance for everyone, including official (0.277 vs ~0.27).** Sanity check:
   the suite behaves as designed (ARC-Easy carries no signal at 160M/≤6B), and official is not
   "magically better at everything" — only where there is learnable signal.
4. **ccot-5b is broken (training-convergence failure, not an eval bug).** Worst on every signal
   benchmark; LAMBADA ≈ 0. See `project_cortex_main.md` memory (2026-06-25) for the full
   diagnosis (additive un-gated whole-sequence carry; train loss plateaus ~4.8 vs ~4.08 for the
   converging recurrent models).
5. **Likely recipe suspects behind the in-house gap:** (a) the LR/WD cooldown "sawtooth" artifact
   baked into these 5B checkpoints (each 1.25B resume re-cooled LR & WD toward 0 — wasted tokens;
   already fixed for FUTURE runs via `--schedule_tokens`); (b) peak LR 3e-4 + Muon config vs
   Pythia's 6e-4; (c) much smaller batch (32,768 tokens/step vs official ~2.1M) — note batch size
   is a throughput/stability lever, NOT expected to close the quality gap on its own (larger batch
   is at best equally token-efficient, never more).

## Recommended next step
Do ONE clean single-schedule pythia re-run (correct `schedule_tokens` from the start; consider a
higher/tuned peak LR) and check whether it reaches ~0.22 LAMBADA at ~5–6B tokens. Lock the recipe
in BEFORE committing compute to the 2.8B scale-up. The in-house ablation stays internally valid
(all variants share the recipe), but the absolute numbers currently understate what the recipe can
do, which matters for paper credibility against published Pythia-160m.
