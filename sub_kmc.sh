#!/usr/bin/env bash
#SBATCH -p cm1 -A pc_lnpmc -q cm1_normal
#SBATCH -N 1 -t 48:00:00 -n 48 -c 1 --mem=192G
#SBATCH -J kmc_npt12_parallel

set -euo pipefail

# Archived EM_effect-4nm/1.0 baseline:
# - 4 nm core (core radius 40 A)
# - 5.5 nm shell (shell thickness 55 A)
# - NPT exported CR rows
# - sigma_esa_scale = 600
# - s12_scale = 30
# - q21/s54/s45/W3NR/W5NR unchanged
# - EM enabled with em_scale = 1.0
# - no surface quenching
#
# Power parallelism is handled automatically inside tm_npt_kmc_production.py.
# With the current Slurm allocation (-n 80) and --thread-count 8, the script
# will run up to floor(80 / 8) = 10 powers concurrently.

python -B "./tm_npt_kmc_production.py" \
  --core-radius-a 130 \
  --surface-quench-mode outer_layer \
  --shell-thickness-a 25 \
  --tm-fraction 0.0456 \
  --npt-cr-mode exported \
  --sigma-esa-scale 600 \
  --s12-scale 30 \
  --power-sampling-mode centered-gaussian \
  --power-center 10000 \
  --power-min 3000 \
  --power-max 30000 \
  --power-count 12 \
  --num-sims 8 \
  --thread-count 8 \
  --q21-scale 1 \
  --s54-scale 1 \
  --s45-scale 1 \
  --fixed-W3_NR-scale 1 \
  --fixed-W5_NR-scale 1 \
  --em-mode all \
  --em-scale 0.01 \
  --simulation-length 10000000
#  --cutoff-mode physical-time \
#  --simulation-time 1.0
