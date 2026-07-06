"""Combine per-model easy-eval results into one accuracy table.

The per-model job scripts (job_eval_easy_<model>.sh) each write
<results_root>/<model>/results.json; this prints whatever subset exists,
so it is safe to run while some jobs are still queued.

Usage:
    python evals/aggregate_easy.py [results_root]

Without an argument, uses the most recent eval_results/easy_* directory.
"""
import json
import sys
from pathlib import Path

CANONICAL = ["pythia-5b", "parcae-5b", "ccot-5b", "cortex-5b", "cortex-5b-k4"]
TASKS = ["lambada", "blimp", "sciq", "arc_easy", "piqa"]


def main():
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        roots = sorted(Path("eval_results").glob("easy_*"))
        if not roots:
            sys.exit("no eval_results/easy_* directories found")
        root = roots[-1]

    found = {p.parent.name: p for p in sorted(root.glob("*/results.json"))}
    if not found:
        sys.exit(f"no <model>/results.json files under {root}")

    # Canonical model order first, then anything else (e.g. HF reference runs).
    models = [m for m in CANONICAL if m in found]
    models += sorted(set(found) - set(CANONICAL))
    missing = [m for m in CANONICAL if m not in found]

    col_w = 12
    label_w = max(len(m) for m in models) + 2
    header = f"{'Model':<{label_w}}" + "".join(f"{t:>{col_w}}" for t in TASKS)
    sep = "-" * len(header)

    print(f"\nEasy benchmarks -- accuracy ({root})")
    print(sep)
    print(header)
    print(sep)
    for m in models:
        with open(found[m]) as f:
            d = json.load(f)
        vals = [f"{d[t]['accuracy']:.4f}" if t in d else "N/A" for t in TASKS]
        print(f"{m:<{label_w}}" + "".join(f"{v:>{col_w}}" for v in vals))
    print(sep)
    if missing:
        print(f"still missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
