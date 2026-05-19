"""
LongMemEval evaluation for CortexGPT.

Compares CortexGPT checkpoints (varying K and T) against the Pythia-160M baseline
on LongMemEval.  Reports accuracy vs. conversation turn depth.

Dataset: xiaowu0162/LongMemEval (HuggingFace)
  Each example contains a multi-turn conversation history + a question requiring
  recall of information from an earlier turn.

Usage:
    # CortexGPT checkpoint (K=4, T=8)
    python eval_longmemeval.py \
        --checkpoint runs/cortex-160m-k4-stage1/checkpoint_0010000/checkpoint.pt \
        --memory_slots 4 \
        --T 8

    # Pythia-160M baseline
    python eval_longmemeval.py --baseline

    # Sweep K values
    python eval_longmemeval.py \
        --checkpoint_k0 runs/cortex-160m-k0/.../checkpoint.pt \
        --checkpoint_k4 runs/cortex-160m-k4/.../checkpoint.pt \
        --sweep

Output: JSON + CSV of accuracy vs. turn depth bucket (5/10/20/50 turns).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import build_cortex_gpt


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("LongMemEval evaluation for CortexGPT vs Pythia-160M")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=str, default=None,
                      help="Path to a CortexGPT checkpoint.pt file")
    mode.add_argument("--baseline", action="store_true",
                      help="Evaluate vanilla Pythia-160M (no recurrence)")
    mode.add_argument("--sweep", action="store_true",
                      help="Sweep K values using --checkpoint_k0 and --checkpoint_k4")

    p.add_argument("--checkpoint_k0", type=str, default=None)
    p.add_argument("--checkpoint_k1", type=str, default=None)
    p.add_argument("--checkpoint_k4", type=str, default=None)

    p.add_argument("--model_name", default="EleutherAI/pythia-160m")
    p.add_argument("--memory_slots", type=int, default=4)
    p.add_argument("--T", type=int, default=8)

    p.add_argument("--seq_len", type=int, default=2048,
                   help="Tokens per chunk when encoding long conversations")
    p.add_argument("--max_examples", type=int, default=200,
                   help="Max examples per turn-depth bucket (0 = all)")
    # Turn depth buckets: classify examples by how many turns ago the target fact appeared
    p.add_argument("--depth_buckets", nargs="+", type=int,
                   default=[5, 10, 20, 50],
                   help="Turn-depth thresholds for bucketing (e.g. ≤5 turns ago)")

    p.add_argument("--out_dir", default="eval_results/longmemeval")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading  (shared with eval_babilong)
# ---------------------------------------------------------------------------

def load_cortex_checkpoint(
    checkpoint_path: str,
    model_name: str,
    memory_slots: int,
    dtype: torch.dtype,
    device: torch.device,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model, cfg = build_cortex_gpt(
        model_name   = model_name,
        memory_slots = memory_slots,
        torch_dtype  = dtype,
        device_map   = "cpu",
    )
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
# Conversation encoding
# ---------------------------------------------------------------------------

def format_conversation(turns: list[dict], question: str) -> str:
    """
    Render a multi-turn conversation as a flat text string.
    Each turn is {"role": "user"|"assistant", "content": str}.
    """
    lines = []
    for t in turns:
        role    = t.get("role", "user").capitalize()
        content = t.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def encode_and_chunk(tokenizer, text: str, seq_len: int) -> list[torch.Tensor]:
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
def eval_one_cortex(
    model,
    tokenizer,
    turns: list[dict],
    question: str,
    answer: str,
    T: int,
    seq_len: int,
) -> bool:
    text   = format_conversation(turns, question)
    chunks = encode_and_chunk(tokenizer, text, seq_len)
    device = next(model.parameters()).device

    m_cross = None
    for chunk in chunks[:-1]:
        chunk = chunk.to(device)
        out   = model(
            input_ids     = chunk,
            num_steps     = [(0, T)],
            m_cross_in    = m_cross,
            return_m_cross= (model.m_cross is not None),
        )
        if model.m_cross is not None:
            m_cross = out.get("m_cross")

    # Final chunk: decode next token
    chunk = chunks[-1].to(device)
    out   = model(
        input_ids     = chunk,
        num_steps     = [(0, T)],
        m_cross_in    = m_cross,
        return_m_cross= False,
    )
    pred_id   = out["logits"][0, -1].argmax(dim=-1).item()
    pred_text = tokenizer.decode([pred_id]).strip().lower()
    return pred_text == answer.strip().lower()


@torch.no_grad()
def eval_one_baseline(
    model,
    tokenizer,
    turns: list[dict],
    question: str,
    answer: str,
    seq_len: int,
) -> bool:
    text   = format_conversation(turns, question)
    chunks = encode_and_chunk(tokenizer, text, seq_len)
    device = next(model.parameters()).device

    # Baseline: only last chunk (no cross-chunk memory)
    chunk     = chunks[-1].to(device)
    out       = model(chunk)
    pred_id   = out.logits[0, -1].argmax(dim=-1).item()
    pred_text = tokenizer.decode([pred_id]).strip().lower()
    return pred_text == answer.strip().lower()


# ---------------------------------------------------------------------------
# Dataset evaluation loop
# ---------------------------------------------------------------------------

def assign_depth_bucket(
    n_turns_since_fact: int,
    depth_buckets: list[int],
    bucket_labels: list[str],
) -> str:
    for i, thresh in enumerate(depth_buckets):
        if n_turns_since_fact <= thresh:
            return bucket_labels[i]
    return bucket_labels[-1]


def run_eval(
    model,
    tokenizer,
    is_cortex: bool,
    T: int,
    seq_len: int,
    max_examples: int,
    depth_buckets: list[int],
) -> dict[str, dict]:
    from datasets import load_dataset

    bucket_labels = [f"≤{d}turns" for d in depth_buckets] + [f">{depth_buckets[-1]}turns"]
    results: dict[str, dict] = {lbl: {"correct": 0, "total": 0} for lbl in bucket_labels}

    ds = load_dataset("xiaowu0162/LongMemEval", split="test", streaming=True,
                      trust_remote_code=True)

    seen = 0
    for ex in ds:
        turns    = ex.get("history", ex.get("messages", []))
        question = ex.get("question", "")
        answer   = ex.get("answer", "")
        # depth: how many turns ago the relevant information appeared
        depth    = ex.get("turn_depth", ex.get("depth", len(turns)))

        if not turns or not question or not answer:
            continue

        bucket = assign_depth_bucket(depth, depth_buckets, bucket_labels)
        if max_examples > 0 and results[bucket]["total"] >= max_examples:
            continue

        if is_cortex:
            correct = eval_one_cortex(model, tokenizer, turns, question, answer, T, seq_len)
        else:
            correct = eval_one_baseline(model, tokenizer, turns, question, answer, seq_len)

        results[bucket]["correct"] += int(correct)
        results[bucket]["total"]   += 1
        seen += 1

        if seen % 50 == 0:
            print(f"  {seen} examples processed...")

        if max_examples > 0 and all(r["total"] >= max_examples for r in results.values()):
            break

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

    runs: list[tuple[str, object, bool, int]] = []

    if args.baseline:
        m = load_baseline(args.model_name, dtype, device)
        runs.append(("pythia-160m-baseline", m, False, 1))

    elif args.checkpoint:
        m, _ = load_cortex_checkpoint(
            args.checkpoint, args.model_name, args.memory_slots, dtype, device
        )
        runs.append((f"cortex-K{args.memory_slots}-T{args.T}", m, True, args.T))

    elif args.sweep:
        sweep_ckpts = {0: args.checkpoint_k0, 1: args.checkpoint_k1, 4: args.checkpoint_k4}
        for K_val, ckpt_path in sweep_ckpts.items():
            if not ckpt_path:
                continue
            m, _ = load_cortex_checkpoint(ckpt_path, args.model_name, K_val, dtype, device)
            runs.append((f"cortex-K{K_val}-T{args.T}", m, True, args.T))
        m_base = load_baseline(args.model_name, dtype, device)
        runs.append(("pythia-160m-baseline", m_base, False, 1))

    all_results: dict = {}
    for label, model, is_cortex, T in runs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        results = run_eval(
            model          = model,
            tokenizer      = tokenizer,
            is_cortex      = is_cortex,
            T              = T,
            seq_len        = args.seq_len,
            max_examples   = args.max_examples,
            depth_buckets  = args.depth_buckets,
        )
        all_results[label] = results

        print(f"\n  Turn depth → accuracy ({label})")
        print(f"  {'Bucket':<16} {'Correct':>8} {'Total':>8} {'Acc':>8}")
        print(f"  {'-'*44}")
        for bucket, r in results.items():
            if r["total"] > 0:
                print(f"  {bucket:<16} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>8.3f}")

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("model,bucket,correct,total,accuracy\n")
        for label, bucket_dict in all_results.items():
            for bucket, r in bucket_dict.items():
                f.write(f"{label},{bucket},{r['correct']},{r['total']},{r['accuracy']:.4f}\n")
    print(f"CSV summary saved → {csv_path}")


if __name__ == "__main__":
    main()
