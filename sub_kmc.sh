#!/usr/bin/env bash
#SBATCH -p cm1 -A pc_lnpmc -q cm1_normal
#SBATCH -N 1 -t 48:00:00 -n 48 -c 1 --mem=192G
#SBATCH -J kmc_npt12_parallel

set -euo pipefail

# Current NPT PA-loop test:
# - 5 nm core (core radius 50 A)
# - 2.5 nm shell (shell thickness 25 A)
# - NPT exported CR rows
# - sigma_esa_scale = 1
# - beta_s12 = 0.003 cm, residual s12_scale = 1
# - q21/s54/s45 unchanged
# - EM enabled with em_scale = 1.0
# - outer-layer surface quenching enabled
# - centered powers run up to 10M steps; power-range limits damp to 2M steps
#
# Power parallelism is handled automatically inside tm_npt_kmc_production.py.
# With the current Slurm allocation (-n 48) and --thread-count 8, the script
# will run up to floor(48 / 8) = 6 powers concurrently.

python -B "./tm_npt_kmc_production.py" \
  --core-radius-a 50 \
  --surface-quench-mode outer_layer \
  --shell-thickness-a 25 \
  --tm-fraction 0.08 \
  --npt-cr-mode exported \
  --sigma-esa-scale 1.0 \
  --beta-s12 0.003 \
  --s12-scale 1 \
  --power-sampling-mode centered-gaussian \
  --power-center 10000 \
  --power-min 3000 \
  --power-max 50000 \
  --power-count 12 \
  --num-sims 8 \
  --thread-count 8 \
  --q21-scale 1 \
  --s54-scale 1 \
  --s45-scale 1 \
  --em-mode all \
  --em-scale 1.0 \
  --simulation-length 5000000 \
  --max-simulation-length 30000000 \
  --output-root r50-8p0-EM1p0
#  --cutoff-mode physical-time \
#  --simulation-time 1.0
