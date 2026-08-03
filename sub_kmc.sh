#!/usr/bin/env bash
#SBATCH -p cm1 -A pc_lnpmc -q cm1_normal
#SBATCH -N 1 -t 72:00:00 -n 48 -c 1 --mem=192G
#SBATCH -J kmc_npt12_adaptive

set -euo pipefail

# Adaptive two-stage sweep (terminal-blocks-v2), em_scale = 1.0.
# Resource-relevant values are explicit: physics scales, seed/thread counts,
# step cutoffs, and the power grid. Convergence-test parameters and grid
# geometry knobs (--refine-half-width-decades etc.) follow built-in defaults.
# Pilot runs highest power first; re-submit this script to resume.
python -B "./tm_npt_kmc_production.py" \
  --workflow-mode adaptive-two-stage \
  --core-radius-a 50 \
  --shell-thickness-a 25 \
  --tm-fraction 0.08 \
  --surface-quench-mode outer_layer \
  --sigma-esa-scale 1 \
  --s12-scale 1 \
  --em-scale 1.0 \
  --pilot-power-min 3000 \
  --pilot-power-max 50000 \
  --pilot-power-count 12 \
  --pilot-num-sims 8 \
  --refine-power-count 12 \
  --refine-num-sims 16 \
  --thread-count 8 \
  --pilot-step-cutoff 5000000 \
  --checkpoint-extension-steps 5000000 \
  --max-step-cutoff 50000000 \
  --output-root r50-8p0-EM1-adaptive
  --power-parallel-total-slots 24
