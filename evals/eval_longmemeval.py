"""
LongMemEval evaluation for CortexGPT.

Accuracy vs. conversation turn depth on LongMemEval.
The model reads multi-turn conversation history in chunks, carrying M_cross
across chunks, then predicts the answer token.

Dataset: xiaowu0162/LongMemEval  (HuggingFace, test split)

Usage:
    python evals/eval_longmemeval.py \
        --checkpoint runs/cortex-5b/checkpoint_0154441/checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model_utils import load_checkpoint


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("LongMemEval evaluation for CortexGPT")
    p.add_argument("--checkpoint",   type=str, required=True)
    p.add_argument("--model_name",   default="EleutherAI/pythia-160m")
    p.add_argument("--memory_slots", type=int, default=0)
    p.add_argument("--T",            type=int, default=None,
                   help="Recurrence depth at eval (None = use checkpoint mean_recurrence)")
    p.add_argument("--seq_len",      type=int, default=2048)
    p.add_argument("--max_examples", type=int, default=200,
                   help="Max examples per turn-depth bucket (0 = all)")
    p.add_argument("--depth_buckets", nargs="+", type=int, default=[5, 10, 20, 50])
    p.add_argument("--out_dir",      default="eval_results/longmemeval")
    p.add_argument("--dtype",        default="bfloat16", choices=["float32", "bfloat16"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Conversation formatting and chunking
# ---------------------------------------------------------------------------

def format_conversation(turns: list[dict], question: str) -> str:
    lines = [f"{t.get('role','user').capitalize()}: {t.get('content','')}" for t in turns]
    lines += [f"Question: {question}", "Answer:"]
    return "\n".join(lines)


def encode_and_chunk(tokenizer, text: str, seq_len: int):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    chunks = []
    for start in range(0, max(len(ids), 1), seq_len):
        chunk = ids[start : start + seq_len]
        chunks.append(torch.tensor(chunk, dtype=torch.long).unsqueeze(0))
    return chunks


# ---------------------------------------------------------------------------
# Single-example evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_one(model, tokenizer, turns, question, answer, T, seq_len) -> bool:
    text   = format_conversation(turns, question)
    chunks = encode_and_chunk(tokenizer, text, seq_len)
    device = next(model.parameters()).device
    num_steps = None if T is None else [(0, T)]

    m_cross = None
    for chunk in chunks[:-1]:
        chunk = chunk.to(device)
        out   = model(input_ids=chunk, num_steps=num_steps,
                      m_cross_in=m_cross, return_m_cross=(model.m_cross is not None))
        if model.m_cross is not None:
            m_cross = out.get("m_cross")

    chunk = chunks[-1].to(device)
    out   = model(input_ids=chunk, num_steps=num_steps,
                  m_cross_in=m_cross, return_m_cross=False)
    pred  = tokenizer.decode([out["logits"][0, -1].argmax(dim=-1).item()]).strip().lower()
    return pred == answer.strip().lower()


# ---------------------------------------------------------------------------
# Dataset evaluation loop
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, T, seq_len, max_examples, depth_buckets):
    from datasets import load_dataset

    bucket_labels = [f"≤{d}turns" for d in depth_buckets] + [f">{depth_buckets[-1]}turns"]
    results = {lbl: {"correct": 0, "total": 0} for lbl in bucket_labels}

    ds   = load_dataset("xiaowu0162/LongMemEval", split="test", streaming=True)
    seen = 0
    for ex in ds:
        turns    = ex.get("history", ex.get("messages", []))
        question = ex.get("question", "")
        answer   = ex.get("answer", "")
        depth    = ex.get("turn_depth", ex.get("depth", len(turns)))
        if not turns or not question or not answer:
            continue

        bucket = bucket_labels[-1]
        for i, thresh in enumerate(depth_buckets):
            if depth <= thresh:
                bucket = bucket_labels[i]
                break

        if max_examples > 0 and results[bucket]["total"] >= max_examples:
            continue

        if eval_one(model, tokenizer, turns, question, answer, T, seq_len):
            results[bucket]["correct"] += 1
        results[bucket]["total"] += 1
        seen += 1

        if seen % 50 == 0:
            print(f"  {seen} examples processed...")

        if max_examples > 0 and all(r["total"] >= max_examples for r in results.values()):
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
    print(f"T={T if T is not None else cfg.mean_recurrence}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_eval(model, tokenizer, T, args.seq_len, args.max_examples, args.depth_buckets)
    label   = Path(args.checkpoint).parent.parent.name

    print(f"\n  {'Bucket':<16} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print(f"  {'-'*44}")
    for bucket, r in results.items():
        if r["total"] > 0:
            print(f"  {bucket:<16} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>8.3f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({label: results}, f, indent=2)

    with open(out_dir / "summary.csv", "w") as f:
        f.write("model,bucket,correct,total,accuracy\n")
        for bucket, r in results.items():
            f.write(f"{label},{bucket},{r['correct']},{r['total']},{r['accuracy']:.4f}\n")

    print(f"\nResults saved → {out_dir}")


if __name__ == "__main__":
    main()
