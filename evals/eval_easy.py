"""
Easy / small-model evaluation for CortexGPT, Pythia, Parcae and CCoT checkpoints.

Targets the benchmarks that give usable signal at very small (≈155M) scale,
where HellaSwag / ARC-Challenge / GSM8K are still at chance:

  LAMBADA   — last-word prediction accuracy (EleutherAI/lambada_openai).
  BLIMP     — linguistic minimal pairs; pick the higher-log-prob full sentence
              (nyu-mll/blimp, 67 paradigms aggregated).
  SciQ      — 4-way multiple choice (allenai/sciq).
  ARC-Easy  — 4-way multiple choice (allenai/ai2_arc).
  PIQA      — 2-way multiple choice (ybisk/piqa).

Scoring conventions:
  - Multiple choice (SciQ, ARC-Easy, PIQA): mean per-token log-prob of each
    choice continuation given its context; highest wins (shared with
    eval_multiple_choice.log_prob_of_completion / run_task).
  - BLIMP: total log-prob of each full sentence; the grammatical ("good")
    sentence must outscore the ungrammatical ("bad") one.
  - LAMBADA: greedy argmax of every target-word token must match (the standard
    LAMBADA accuracy metric).

Usage:
    python evals/eval_easy.py \
        --checkpoint runs/pythia-5b/checkpoint_XXXXXXX/checkpoint.pt \
        --T 1 \
        --tasks lambada blimp sciq arc_easy piqa \
        --out_dir eval_results/easy/pythia-5b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model_utils import load_checkpoint
from model import build_pythia
# Reuse the multiple-choice scorer/runner and two of its dataset loaders so the
# MC methodology is identical to eval_multiple_choice.py.
from eval_multiple_choice import log_prob_of_completion, run_task, load_arc, load_piqa


TASK_CHOICES = ["lambada", "blimp", "sciq", "arc_easy", "piqa"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Easy / small-model evaluation for CortexGPT")
    p.add_argument("--checkpoint",   type=str, default=None,
                   help="Local .pt checkpoint to evaluate (CortexGPT/Pythia/CCoT)")
    p.add_argument("--hf_model",     type=str, default=None,
                   help="Instead of --checkpoint, eval an official HF Pythia model "
                        "(e.g. EleutherAI/pythia-160m-deduped) as a reference. "
                        "Forces the non-recurrent T=1 path.")
    p.add_argument("--revision",     type=str, default=None,
                   help="HF git revision for --hf_model, e.g. step2000 (~4.2B tokens) "
                        "or step3000 (~6.3B tokens)")
    p.add_argument("--model_name",   default="EleutherAI/pythia-160m")
    p.add_argument("--memory_slots", type=int, default=None,
                   help="Override K; default reads memory_slots from the checkpoint config")
    p.add_argument("--T",            type=int, default=None,
                   help="Recurrence depth at eval (None = use checkpoint mean_recurrence; "
                        "pass 1 for the non-recurrent pythia baseline)")
    p.add_argument("--tasks",        nargs="+", default=TASK_CHOICES, choices=TASK_CHOICES)
    p.add_argument("--max_examples", type=int, default=0,
                   help="0 = all. For BLIMP this caps examples PER paradigm.")
    p.add_argument("--blimp_paradigms", nargs="+", default=None,
                   help="Subset of BLIMP paradigm configs (default: all 67)")
    p.add_argument("--seq_len",      type=int, default=2048)
    p.add_argument("--out_dir",      default="eval_results/easy")
    p.add_argument("--dtype",        default="bfloat16", choices=["float32", "bfloat16"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# SciQ loader → (context, [choices], correct_idx)  [MC scoring]
# ---------------------------------------------------------------------------

def load_sciq(max_examples: int):
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="validation")
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    examples = []
    for ex in ds:
        question = ex["question"]
        support  = (ex.get("support") or "").strip()
        # Prepend the supporting passage when present (many SciQ rows have none).
        context  = f"{support} {question}" if support else question
        # Correct answer first; correct_idx = 0.
        choices  = [ex["correct_answer"], ex["distractor1"],
                    ex["distractor2"], ex["distractor3"]]
        examples.append((context, choices, 0))
    return examples


# ---------------------------------------------------------------------------
# BLIMP — full-sentence log-prob, grammatical must beat ungrammatical
# ---------------------------------------------------------------------------

@torch.no_grad()
def sentence_logprob(model, ids: list[int], T: Optional[int],
                     seq_len: int, device: torch.device) -> float:
    """Total (summed) log-prob of a full token sequence under the model."""
    all_ids   = ids[-seq_len:]
    input_ids = torch.tensor(all_ids, dtype=torch.long).unsqueeze(0).to(device)
    num_steps = None if T is None else [(0, T)]
    logits    = model(input_ids=input_ids, num_steps=num_steps)["logits"][0]   # [S, V]

    log_probs = F.log_softmax(logits[:-1], dim=-1)        # [S-1, V]
    target    = input_ids[0, 1:]                          # [S-1]
    tok_lp    = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return tok_lp.sum().item()


def run_blimp(paradigms, model, tokenizer, T, seq_len, device, max_examples):
    from datasets import load_dataset, get_dataset_config_names
    if not paradigms:
        paradigms = get_dataset_config_names("nyu-mll/blimp")

    per_paradigm = {}
    total_correct = total = 0
    for p in paradigms:
        ds = load_dataset("nyu-mll/blimp", p, split="train")
        if max_examples > 0:
            ds = ds.select(range(min(max_examples, len(ds))))
        correct = 0
        for ex in ds:
            good = tokenizer(ex["sentence_good"], add_special_tokens=False).input_ids
            bad  = tokenizer(ex["sentence_bad"],  add_special_tokens=False).input_ids
            if not good or not bad:
                continue
            sg = sentence_logprob(model, good, T, seq_len, device)
            sb = sentence_logprob(model, bad,  T, seq_len, device)
            if sg > sb:
                correct += 1
        n = len(ds)
        acc = correct / n if n else 0.0
        per_paradigm[p] = {"correct": correct, "total": n, "accuracy": acc}
        total_correct += correct
        total += n
        print(f"  [blimp/{p}] {correct}/{n} = {acc:.4f}")

    overall = total_correct / total if total else 0.0
    return {"correct": total_correct, "total": total, "accuracy": overall,
            "paradigms": per_paradigm}


# ---------------------------------------------------------------------------
# LAMBADA — greedy last-word accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def lambada_correct(model, tokenizer, text: str, T: Optional[int],
                    seq_len: int, device: torch.device) -> Optional[bool]:
    """True if the model's greedy prediction reproduces every token of the
    final whitespace-delimited word.  Returns None if the example is unusable."""
    text = text.strip()
    idx  = text.rfind(" ")
    if idx <= 0:
        return None                              # no context / single word
    context, target = text[:idx], text[idx:]     # target keeps its leading space

    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
    tgt_ids = tokenizer(target,  add_special_tokens=False).input_ids
    if not tgt_ids or not ctx_ids:
        return None

    all_ids = (ctx_ids + tgt_ids)[-seq_len:]
    n_t     = min(len(tgt_ids), len(all_ids) - 1)
    if n_t <= 0:
        return None
    input_ids = torch.tensor(all_ids, dtype=torch.long).unsqueeze(0).to(device)
    num_steps = None if T is None else [(0, T)]
    logits    = model(input_ids=input_ids, num_steps=num_steps)["logits"][0]   # [S, V]

    # logits[i] predicts token i+1, so the target tokens (last n_t positions)
    # are predicted by logits at positions [-n_t-1 : -1].
    preds  = logits[-n_t - 1:-1].argmax(dim=-1)
    target = torch.tensor(all_ids[-n_t:], device=device)
    return bool((preds == target).all().item())


def run_lambada(model, tokenizer, T, seq_len, device, max_examples):
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    correct = total = 0
    for i, ex in enumerate(ds):
        res = lambada_correct(model, tokenizer, ex["text"], T, seq_len, device)
        if res is None:
            continue
        total   += 1
        correct += int(res)
        if (i + 1) % 500 == 0:
            print(f"  [lambada] {i+1}/{len(ds)}  acc={correct/max(total,1):.4f}")
    acc = correct / total if total else 0.0
    return {"correct": correct, "total": total, "accuracy": acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MC_LOADERS = {
    "sciq":     load_sciq,
    "arc_easy": lambda n: load_arc("ARC-Easy", n),
    "piqa":     load_piqa,
}


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    if bool(args.checkpoint) == bool(args.hf_model):
        raise SystemExit("Pass exactly one of --checkpoint or --hf_model")

    if args.hf_model:
        rev = f"@{args.revision}" if args.revision else ""
        print(f"Loading HF reference model: {args.hf_model}{rev}")
        model, cfg = build_pythia(
            model_name=args.hf_model, from_scratch=False,
            torch_dtype=dtype, device_map="cpu", revision=args.revision,
        )
        model = model.to(device).eval()
        tok_name = args.hf_model
        args.T = 1                      # official Pythia is non-recurrent
    else:
        print(f"Loading checkpoint: {args.checkpoint}")
        model, cfg = load_checkpoint(args.checkpoint, args.model_name,
                                     args.memory_slots, dtype, device)
        tok_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    T = args.T
    print(f"T={T if T is not None else cfg.mean_recurrence}  tasks={args.tasks}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for task in args.tasks:
        print(f"\n{'='*55}\nTask: {task}\n{'='*55}")
        if task in MC_LOADERS:
            examples = MC_LOADERS[task](args.max_examples)
            result   = run_task(task, examples, model, tokenizer, T,
                                 args.seq_len, device)
        elif task == "blimp":
            result = run_blimp(args.blimp_paradigms, model, tokenizer, T,
                               args.seq_len, device, args.max_examples)
        elif task == "lambada":
            result = run_lambada(model, tokenizer, T, args.seq_len, device,
                                 args.max_examples)
        else:
            raise ValueError(f"Unknown task {task!r}")
        all_results[task] = result
        print(f"  {task}: {result['correct']}/{result['total']} = {result['accuracy']:.4f}")

    print(f"\n{'Task':<18} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("-" * 45)
    for task, r in all_results.items():
        print(f"{task:<18} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>8.4f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(out_dir / "summary.csv", "w") as f:
        f.write("task,correct,total,accuracy\n")
        for task, r in all_results.items():
            f.write(f"{task},{r['correct']},{r['total']},{r['accuracy']:.4f}\n")

    print(f"\nResults saved -> {out_dir}")


if __name__ == "__main__":
    main()
