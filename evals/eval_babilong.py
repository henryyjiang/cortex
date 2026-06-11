"""
BABILong evaluation for CortexGPT.

Accuracy vs. context length on BABILong QA1/QA2/QA3.
The model reads the context in seq_len-token chunks, carrying M_cross across
chunks, then predicts the answer token at the end.

Dataset: RMT-team/BABILong  (HuggingFace)

Usage:
    python evals/eval_babilong.py \
        --checkpoint runs/cortex-5b/checkpoint_0154441/checkpoint.pt \
        --tasks qa1 qa2 qa3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from model_utils import load_checkpoint


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("BABILong evaluation for CortexGPT")
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--model_name",    default="EleutherAI/pythia-160m")
    p.add_argument("--memory_slots",  type=int, default=0)
    p.add_argument("--T",             type=int, default=None,
                   help="Recurrence depth at eval (None = use checkpoint mean_recurrence)")
    p.add_argument("--tasks",         nargs="+", default=["qa1", "qa2", "qa3"],
                   choices=["qa1", "qa2", "qa3"])
    p.add_argument("--seq_len",       type=int, default=2048)
    p.add_argument("--max_examples",  type=int, default=500,
                   help="Max examples per task/length bucket (0 = all)")
    p.add_argument("--length_buckets", nargs="+", type=int,
                   default=[1000, 2000, 4000, 8000, 16000, 32000])
    p.add_argument("--out_dir",       default="eval_results/babilong")
    p.add_argument("--dtype",         default="bfloat16", choices=["float32", "bfloat16"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Chunked context encoding
# ---------------------------------------------------------------------------

def encode_and_chunk(tokenizer, context: str, question: str, seq_len: int):
    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
    q_ids   = tokenizer(f"\nQuestion: {question}\nAnswer:", add_special_tokens=False).input_ids
    all_ids = ctx_ids + q_ids
    chunks  = []
    for start in range(0, len(all_ids), seq_len):
        chunk = all_ids[start : start + seq_len]
        chunks.append(torch.tensor(chunk, dtype=torch.long).unsqueeze(0))
    return chunks


# ---------------------------------------------------------------------------
# Single-example evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_one(model, tokenizer, context, question, answer, T, seq_len) -> bool:
    chunks = encode_and_chunk(tokenizer, context, question, seq_len)
    device = next(model.parameters()).device
    num_steps = None if T is None else [(0, T)]

    m_cross = None
    for chunk in chunks:
        chunk = chunk.to(device)
        out   = model(input_ids=chunk, num_steps=num_steps,
                      m_cross_in=m_cross, return_m_cross=(model.m_cross is not None))
        m_cross = out.get("m_cross")

    chunk = chunks[-1].to(device)
    out   = model(input_ids=chunk, num_steps=num_steps,
                  m_cross_in=m_cross, return_m_cross=False)
    pred  = tokenizer.decode([out["logits"][0, -1].argmax(dim=-1).item()]).strip().lower()
    return pred == answer.strip().lower()


# ---------------------------------------------------------------------------
# Task loop
# ---------------------------------------------------------------------------

def run_task(task_name, model, tokenizer, T, seq_len, max_examples, length_buckets):
    from datasets import load_dataset

    # BABILong uses config name for context length (e.g. '1k', '4k') and
    # split for the task (e.g. 'qa1'). Load each bucket config separately.
    config_names = [f"{b // 1000}k" for b in length_buckets]
    results = {cfg: {"correct": 0, "total": 0} for cfg in config_names}

    for cfg in config_names:
        try:
            ds = load_dataset("RMT-team/BABILong", cfg, split=task_name,
                              streaming=True, trust_remote_code=True)
        except Exception as e:
            print(f"  [{task_name}/{cfg}] skipping — {e}")
            continue

        seen = 0
        for ex in ds:
            ctx      = ex.get("context", ex.get("text", ""))
            question = ex.get("question", "")
            answer   = ex.get("answer", "")
            if not ctx or not question or not answer:
                continue

            if eval_one(model, tokenizer, ctx, question, answer, T, seq_len):
                results[cfg]["correct"] += 1
            results[cfg]["total"] += 1
            seen += 1

            if seen % 50 == 0:
                print(f"  [{task_name}/{cfg}] {seen} examples processed...")

            if max_examples > 0 and seen >= max_examples:
                break

    for r in results.values():
        r["accuracy"] = r["correct"] / r["total"] if r["total"] > 0 else 0.0
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint, args.model_name,
                                 args.memory_slots, dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    T = args.T
    print(f"T={T if T is not None else cfg.mean_recurrence}  tasks={args.tasks}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}
    label = Path(args.checkpoint).parent.parent.name   # run dir name as label
    all_results[label] = {}

    for task in args.tasks:
        print(f"\n--- {task} ---")
        task_results = run_task(task, model, tokenizer, T, args.seq_len,
                                args.max_examples, args.length_buckets)
        all_results[label][task] = task_results

        print(f"  {'Bucket':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
        print(f"  {'-'*40}")
        for bucket, r in task_results.items():
            if r["total"] > 0:
                print(f"  {bucket:<12} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>8.3f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("model,task,bucket,correct,total,accuracy\n")
        for lbl, task_dict in all_results.items():
            for task, bucket_dict in task_dict.items():
                for bucket, r in bucket_dict.items():
                    f.write(f"{lbl},{task},{bucket},{r['correct']},{r['total']},{r['accuracy']:.4f}\n")

    print(f"\nResults saved → {out_dir}")


if __name__ == "__main__":
    main()
