#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Hardcoded production run:
# - NPT baseline for both one-site and CR channels
# - mapped NPT scales for the tested-baseline essential knobs
# - Q21/s54/s45 stay at the raw NPT baseline for now
# - EM off

python -B "$SCRIPT_DIR/tm_dre_5level_kmc_production.py" \
  --interaction-mode npt \
  --npt-cr-mode exported \
  --sigma-esa-scale 600 \
  --s12-scale 30 \
  --power-sampling-mode centered-gaussian \
  --power-center 10000 \
  --power-min 3000 \
  --power-max 30000 \
  --power-count 20 \
  --num-sims 8 \
  --thread-count 8 \
  --q21-scale 1 \
  --s54-scale 1 \
  --s45-scale 1 \
  --fixed-W3_NR-scale 1 \
  --fixed-W5_NR-scale 1 \
  --em-mode in_loop \
  --em-scale 0.01 \
  --cutoff-mode physical-time \
  --simulation-time 2.0
#  --simulation-length 2000000 \
