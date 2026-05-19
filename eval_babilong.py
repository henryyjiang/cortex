"""
BABILong evaluation for CortexGPT.

Compares CortexGPT checkpoints (varying K and T) against the Pythia-160M baseline
on BABILong QA1/QA2/QA3.  Reports accuracy vs. context length.

Dataset: RMT-team/BABILong (HuggingFace)
  Each example: {"context": str, "question": str, "answer": str, "task_id": int}

Usage:
    # CortexGPT checkpoint (K=4, T=8)
    python eval_babilong.py \
        --checkpoint runs/cortex-160m-k4-stage1/checkpoint_0010000/checkpoint.pt \
        --memory_slots 4 \
        --T 8 \
        --tasks qa1 qa2 qa3

    # Pythia-160M baseline (no checkpoint needed)
    python eval_babilong.py --baseline --T 1 --tasks qa1 qa2 qa3

    # Sweep K and T (requires multiple checkpoints, one per K)
    python eval_babilong.py \
        --checkpoint_k0 runs/cortex-160m-k0/.../checkpoint.pt \
        --checkpoint_k4 runs/cortex-160m-k4/.../checkpoint.pt \
        --sweep

Output: JSON + CSV tables of accuracy vs. context-length bucket.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import build_cortex_gpt, CortexConfig


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("BABILong evaluation for CortexGPT vs Pythia-160M")

    # Model selection
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=str, default=None,
                      help="Path to a CortexGPT checkpoint.pt file")
    mode.add_argument("--baseline", action="store_true",
                      help="Evaluate vanilla Pythia-160M baseline (no recurrence)")
    mode.add_argument("--sweep", action="store_true",
                      help="Sweep K values using --checkpoint_k0 and --checkpoint_k4")

    p.add_argument("--checkpoint_k0", type=str, default=None)
    p.add_argument("--checkpoint_k1", type=str, default=None)
    p.add_argument("--checkpoint_k4", type=str, default=None)

    # CortexGPT config
    p.add_argument("--model_name", default="EleutherAI/pythia-160m")
    p.add_argument("--memory_slots", type=int, default=4)
    p.add_argument("--T", type=int, default=8,
                   help="Number of loop iterations at eval time")

    # Eval config
    p.add_argument("--tasks", nargs="+", default=["qa1", "qa2", "qa3"],
                   choices=["qa1", "qa2", "qa3"])
    p.add_argument("--seq_len", type=int, default=2048,
                   help="Context window size (tokens) per chunk")
    p.add_argument("--max_examples", type=int, default=500,
                   help="Max examples per task/length-bucket (0 = all)")
    p.add_argument("--length_buckets", nargs="+", type=int,
                   default=[1000, 2000, 4000, 8000, 16000, 32000],
                   help="Context length thresholds in characters for bucketing")

    # Output
    p.add_argument("--out_dir", default="eval_results/babilong")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_cortex_checkpoint(
    checkpoint_path: str,
    model_name: str,
    memory_slots: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model, cfg = build_cortex_gpt(
        model_name    = model_name,
        memory_slots  = memory_slots,
        torch_dtype   = dtype,
        device_map    = "cpu",
    )
    # Restore saved config fields if present
    saved_cfg = ckpt.get("config", {})
    if "mean_recurrence" in saved_cfg:
        cfg.mean_recurrence = saved_cfg["mean_recurrence"]

    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, cfg


def load_baseline(model_name: str, dtype: torch.dtype, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=str(device)
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate_cortex(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    T: int,
    m_cross: Optional[torch.Tensor],
) -> tuple[str, Optional[torch.Tensor]]:
    """
    Run CortexGPT forward in chunks of seq_len, carrying M_cross across chunks.
    Returns the generated token string and the final M_cross buffer.
    """
    tokenizer = None  # used externally; tokens decoded by caller
    out = model(
        input_ids     = input_ids,
        num_steps     = [(0, T)] * input_ids.shape[0],
        m_cross_in    = m_cross,
        return_m_cross= (model.m_cross is not None),
    )
    logits     = out["logits"]          # [1, S, V]
    new_m      = out.get("m_cross")     # [1, K, D] or None

    # Greedy next token from the last position
    next_id = logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)  # [1, 1]
    return next_id, new_m


@torch.no_grad()
def greedy_generate_baseline(
    model,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    logits  = model(input_ids).logits   # [1, S, V]
    next_id = logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
    return next_id


# ---------------------------------------------------------------------------
# Chunked context encoding
# ---------------------------------------------------------------------------

def encode_and_chunk(
    tokenizer,
    context: str,
    question: str,
    seq_len: int,
) -> list[torch.Tensor]:
    """
    Tokenise context+question, split into seq_len-token chunks.
    The last chunk always contains the question at the end.
    Returns list of [1, seq_len] or [1, <=seq_len] tensors.
    """
    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
    q_ids   = tokenizer(
        f"\nQuestion: {question}\nAnswer:", add_special_tokens=False
    ).input_ids

    # Reserve space for question in the last chunk
    q_len  = len(q_ids)
    ctx_budget = max(len(ctx_ids), 1)  # how many context tokens we have

    all_ids = ctx_ids + q_ids
    chunks = []
    for start in range(0, len(all_ids), seq_len):
        chunk = all_ids[start : start + seq_len]
        chunks.append(torch.tensor(chunk, dtype=torch.long).unsqueeze(0))
    return chunks


# ---------------------------------------------------------------------------
# Single-example evaluation
# ---------------------------------------------------------------------------

def eval_one_cortex(
    model,
    tokenizer,
    context: str,
    question: str,
    answer: str,
    T: int,
    seq_len: int,
) -> bool:
    chunks = encode_and_chunk(tokenizer, context, question, seq_len)
    device = next(model.parameters()).device

    m_cross = None
    for chunk in chunks:
        chunk = chunk.to(device)
        _, m_cross = greedy_generate_cortex(model, chunk, max_new_tokens=1, T=T, m_cross=m_cross)

    # Decode the predicted token (last chunk's next token)
    chunk = chunks[-1].to(device)
    out = model(
        input_ids     = chunk,
        num_steps     = [(0, T)],
        m_cross_in    = m_cross,
        return_m_cross= False,
    )
    pred_id   = out["logits"][0, -1].argmax(dim=-1).item()
    pred_text = tokenizer.decode([pred_id]).strip().lower()
    gold      = answer.strip().lower()
    return pred_text == gold


def eval_one_baseline(
    model,
    tokenizer,
    context: str,
    question: str,
    answer: str,
    seq_len: int,
) -> bool:
    chunks = encode_and_chunk(tokenizer, context, question, seq_len)
    device = next(model.parameters()).device

    # Baseline: only use the last seq_len-token chunk (no cross-chunk memory)
    chunk = chunks[-1].to(device)
    out   = model(chunk)
    pred_id   = out.logits[0, -1].argmax(dim=-1).item()
    pred_text = tokenizer.decode([pred_id]).strip().lower()
    gold      = answer.strip().lower()
    return pred_text == gold


# ---------------------------------------------------------------------------
# Task evaluation loop
# ---------------------------------------------------------------------------

def run_task(
    task_name: str,
    model,
    tokenizer,
    is_cortex: bool,
    T: int,
    seq_len: int,
    max_examples: int,
    length_buckets: list[int],
) -> dict[str, dict]:
    """
    Evaluate on one BABILong task.
    Returns dict mapping context-length bucket label → {"correct": int, "total": int}.
    """
    from datasets import load_dataset

    # Map task name → BABILong split name
    split_map = {"qa1": "qa1", "qa2": "qa2", "qa3": "qa3"}
    split = split_map[task_name]

    ds = load_dataset("RMT-team/BABILong", split=split, streaming=True,
                      trust_remote_code=True)

    # Build bucket labels
    bucket_labels = [f"≤{b//1000}K" for b in length_buckets] + [f">{length_buckets[-1]//1000}K"]
    results: dict[str, dict] = {lbl: {"correct": 0, "total": 0} for lbl in bucket_labels}

    seen = 0
    for ex in ds:
        ctx      = ex.get("context", ex.get("text", ""))
        question = ex.get("question", "")
        answer   = ex.get("answer", "")
        if not ctx or not question or not answer:
            continue

        # Assign to length bucket
        ctx_len  = len(ctx)
        bucket   = bucket_labels[-1]
        for i, thresh in enumerate(length_buckets):
            if ctx_len <= thresh:
                bucket = bucket_labels[i]
                break

        if max_examples > 0 and results[bucket]["total"] >= max_examples:
            continue

        if is_cortex:
            correct = eval_one_cortex(model, tokenizer, ctx, question, answer, T, seq_len)
        else:
            correct = eval_one_baseline(model, tokenizer, ctx, question, answer, seq_len)

        results[bucket]["correct"] += int(correct)
        results[bucket]["total"]   += 1
        seen += 1

        if seen % 50 == 0:
            print(f"  [{task_name}] {seen} examples processed...")

        # Stop if all buckets are filled
        if max_examples > 0 and all(r["total"] >= max_examples for r in results.values()):
            break

    # Compute accuracy
    for lbl, r in results.items():
        r["accuracy"] = r["correct"] / r["total"] if r["total"] > 0 else 0.0

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Assemble (label, model, is_cortex, T, K) entries to evaluate ─────────
    runs: list[tuple[str, object, bool, int, int]] = []

    if args.baseline:
        m = load_baseline(args.model_name, dtype, device)
        runs.append(("pythia-160m-baseline", m, False, 1, 0))

    elif args.checkpoint:
        m, _ = load_cortex_checkpoint(
            args.checkpoint, args.model_name, args.memory_slots, dtype, device
        )
        label = f"cortex-K{args.memory_slots}-T{args.T}"
        runs.append((label, m, True, args.T, args.memory_slots))

    elif args.sweep:
        sweep_ckpts = {
            0: args.checkpoint_k0,
            1: args.checkpoint_k1,
            4: args.checkpoint_k4,
        }
        for K_val, ckpt_path in sweep_ckpts.items():
            if not ckpt_path:
                continue
            m, _ = load_cortex_checkpoint(ckpt_path, args.model_name, K_val, dtype, device)
            label = f"cortex-K{K_val}-T{args.T}"
            runs.append((label, m, True, args.T, K_val))
        # Also add baseline
        m_base = load_baseline(args.model_name, dtype, device)
        runs.append(("pythia-160m-baseline", m_base, False, 1, 0))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    all_results: dict = {}
    for label, model, is_cortex, T, K_val in runs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")
        all_results[label] = {}

        for task in args.tasks:
            print(f"\n--- Task: {task} ---")
            task_results = run_task(
                task_name      = task,
                model          = model,
                tokenizer      = tokenizer,
                is_cortex      = is_cortex,
                T              = T,
                seq_len        = args.seq_len,
                max_examples   = args.max_examples,
                length_buckets = args.length_buckets,
            )
            all_results[label][task] = task_results

            # Print table
            print(f"\n  Context length → accuracy ({label}, {task})")
            print(f"  {'Bucket':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
            print(f"  {'-'*40}")
            for bucket, r in task_results.items():
                if r["total"] > 0:
                    print(f"  {bucket:<12} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>8.3f}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Print summary CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("model,task,bucket,correct,total,accuracy\n")
        for label, task_dict in all_results.items():
            for task, bucket_dict in task_dict.items():
                for bucket, r in bucket_dict.items():
                    f.write(f"{label},{task},{bucket},{r['correct']},{r['total']},{r['accuracy']:.4f}\n")
    print(f"CSV summary saved → {csv_path}")


if __name__ == "__main__":
    main()
