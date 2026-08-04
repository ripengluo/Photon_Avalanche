#!/usr/bin/env bash
#SBATCH -p cm1 -A pc_lnpmc -q cm1_normal
#SBATCH -N 1 -t 72:00:00 -n 48 -c 1 --mem=192G
#SBATCH -J kmc_npt12_parallel

set -euo pipefail

# Current NPT PA-loop test:
# - 5 nm core (core radius 50 A)
# - 2.5 nm shell (shell thickness 25 A)
# - NPT exported CR rows
# - sigma_esa_scale = 1
# - s12_scale = 1 (no beta_s12 correction)
# - q21/s54/s45 unchanged
# - EM enabled with em_scale = 1.0
# - outer-layer surface quenching enabled
# - uniform step cutoff: every power runs the same 5M steps
#
# Power parallelism is handled automatically inside tm_npt_kmc_production.py.
# With the current Slurm allocation (-n 48) and --thread-count 8, the script
# will run up to floor(48 / 8) = 6 powers concurrently.

python -B "./tm_npt_kmc_production.py" \
  --core-radius-a 80 \
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
  --thread-count 8 \
  --em-mode all \
  --em-scale 1.0 \
  --simulation-length 50000000 \
  --output-root r80-8p0-baseline \
#  --power-parallel-total-slots 24
#  --cutoff-mode physical-time \
#  --simulation-time 1.0
