#!/usr/bin/env bash
#SBATCH --partition=cm1
#SBATCH --account=pc_lnpmc
#SBATCH --qos=cm1_normal
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --job-name=kmc_npt_prepared
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

JOB_TAG="${SLURM_JOB_ID:-manual}"
OUTPUT_ROOT="${OUTPUT_ROOT:-14nm-8p0Tm}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/home/rpluo/Desktop/hdd_large/KMC_trajectories/accelerate_convergence}"
TOTAL_SLOTS="$(( ${SLURM_NTASKS:-24} * ${SLURM_CPUS_PER_TASK:-1} ))"

echo "output_root=$OUTPUT_ROOT"
echo "archive_root=$ARCHIVE_ROOT"
echo "total_slots=$TOTAL_SLOTS"

exec python -B ./tm_npt_kmc_production.py \
  --output-root "$OUTPUT_ROOT" \
  --trajectory-archive-root "$ARCHIVE_ROOT" \
  --power-sampling-mode centered-gaussian \
  --power-center 11000 \
  --power-gaussian-sigma-decades 0.05 \
  --power-min 3000 \
  --power-max 50000 \
  --power-count 12 \
  --simulation-length 70000000 \
  --num-sims 8 \
  --base-seed 1000 \
  --thread-count 8 \
  --restart-workers 8 \
  --doping-seed 23 \
  --core-radius-a 70 \
  --shell-thickness-a 25 \
  --tm-fraction 0.08 \
  --npt-cr-mode exported \
  --sigma-esa-scale 1 \
  --q21-scale 1 \
  --s54-scale 1 \
  --s45-scale 1 \
  --s12-scale 1 \
  --em-mode all \
  --em-scale 1 \
  --surface-quench-mode outer_layer \
  --surface-species Surface \
  --surface-fraction 0.2 \
  --power-parallel-total-slots "$TOTAL_SLOTS" \
  "$@"
