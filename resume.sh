#!/bin/bash
set -euo pipefail


python -B ./tm_npt_kmc_production.py \
  --core-radius-a 50 \
  --surface-quench-mode outer_layer \
  --shell-thickness-a 25 \
  --tm-fraction 0.08 \
  --npt-cr-mode exported \
  --sigma-esa-scale 1.0 \
  --s12-scale 1 \
  --power-sampling-mode centered-gaussian \
  --power-gaussian-sigma-decades 0.05 \
  --power-center 13000 \
  --power-min 3000 \
  --power-max 50000 \
  --power-count 12 \
  --num-sims 8 \
  --base-seed 1000 \
  --thread-count 8 \
  --power-parallel-total-slots 24 \
  --em-mode all \
  --em-scale 1.0 \
  --simulation-length 50000000 \
  --output-root r50-8p0-baseline-50M \
  --resume \
