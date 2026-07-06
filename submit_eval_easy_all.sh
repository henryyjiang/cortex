#!/bin/bash
# Submit the five per-model easy-eval jobs (one model per 24h job) with a
# shared EVAL_TAG so they all write into the same results root regardless of
# when each comes off the queue.  Run on a login node:
#   ./submit_eval_easy_all.sh            # tag = today's date
#   ./submit_eval_easy_all.sh mytag      # explicit tag
#
# When all five finish, print the combined table with:
#   python evals/aggregate_easy.py eval_results/easy_<tag>
set -e
cd "$(dirname "$0")"

export EVAL_TAG="${1:-$(date +%Y%m%d)}"
echo "EVAL_TAG=$EVAL_TAG  ->  results root eval_results/easy_$EVAL_TAG"

for SCRIPT in job_eval_easy_pythia.sh \
              job_eval_easy_parcae.sh \
              job_eval_easy_ccot.sh \
              job_eval_easy_cortex.sh \
              job_eval_easy_cortex_k4.sh; do
    sbatch "$SCRIPT"
done
