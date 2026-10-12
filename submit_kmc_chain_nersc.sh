#!/usr/bin/env bash
# Usage: NERSC_ACCOUNT=m1234 OUTPUT_ROOT="$SCRATCH/kmc/run1" \
#   NPMC_COMMAND=/path/to/NPMC bash submit_kmc_chain_nersc.sh 3 [runner options]
# Activate the Python/NPT environment before submission.
# Use a NEW output directory on $SCRATCH or another shared filesystem.
# Same command/settings can submit another chain if JOB_COUNT was insufficient.
# Defaults: 24 hours/job, 15 minutes to flush checkpoints, one CPU node.
# Override WALLTIME, CHECKPOINT_SECONDS, or CPUS in the submission environment.
# --continue-run is not --resume: it finishes the original target, not an extension.
# Trajectories stay in OUTPUT_ROOT; incomplete analysis is retried next job.
# Hard kills/unclean NPMC exits stop the chain rather than trust a partial DB.
# Native NPMC checkpoints restore state/time/step, not the RNG stream.
# NERSC: https://docs.nersc.gov/jobs/examples/#dependencies
# Slurm: https://slurm.schedmd.com/sbatch.html
set -euo pipefail

if [[ "${1:-}" == "--run" ]]; then
  REPO="$2"
  export OUTPUT_ROOT="$3"
  shift 3
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  runner=""
  forward_checkpoint() {
    if [[ -n "$runner" ]]; then
      kill -USR1 "$runner" 2>/dev/null || true
    fi
  }
  trap forward_checkpoint USR1
  trap forward_checkpoint TERM
  # sub_kmc.sh execs Python, so this PID receives the batch-only warning.
  bash "$REPO/sub_kmc.sh" "$@" --continue-run --output-root "$OUTPUT_ROOT" &
  runner=$!
  while true; do
    if wait "$runner"; then
      status=0
      break
    else
      status=$?
    fi
    # A trapped signal interrupts wait; wait again for the actual runner exit.
    if (( status > 128 )) && kill -0 "$runner" 2>/dev/null; then
      continue
    fi
    break
  done
  if (( status == 75 )); then
    echo "Clean checkpoint pause. The next chained job can continue."
    exit 0
  fi
  exit "$status"
fi

if [[ ! "${1:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: NERSC_ACCOUNT=... OUTPUT_ROOT=... $0 JOB_COUNT [runner options]" >&2
  exit 2
fi
JOBS="$1"
shift
: "${NERSC_ACCOUNT:?Set NERSC_ACCOUNT to your NERSC CPU allocation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:?Set OUTPUT_ROOT or SCRATCH}/kmc/14nm-8p0Tm}"
CPUS="${CPUS:-256}"
CHECKPOINT_SECONDS="${CHECKPOINT_SECONDS:-900}"
WALLTIME="${WALLTIME:-48:00:00}"
if [[ ! "$CPUS" =~ ^[1-9][0-9]*$ || ! "$CHECKPOINT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPUS and CHECKPOINT_SECONDS must be positive integers." >&2
  exit 2
fi
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd)"
dependency=()
for ((i=1; i<=JOBS; i++)); do
  job_id="$(sbatch --parsable \
    --account="$NERSC_ACCOUNT" --qos=regular --constraint=cpu \
    --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --hint=nomultithread --mem=0 \
    --time="$WALLTIME" --signal="B:USR1@$CHECKPOINT_SECONDS" \
    --job-name=kmc_chain --output="$OUTPUT_ROOT/slurm-%j.out" \
    --chdir="$REPO" --export=ALL --kill-on-invalid-dep=yes \
    "${dependency[@]}" \
    "$REPO/submit_kmc_chain_nersc.sh" --run "$REPO" "$OUTPUT_ROOT" \
    --npmc-command "${NPMC_COMMAND:-NPMC}" "$@")"
  job_id="${job_id%%;*}"
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Unexpected sbatch response: $job_id; no more jobs submitted." >&2
    exit 1
  fi
  echo "Submitted $i/$JOBS: $job_id (output: $OUTPUT_ROOT)"
  dependency=("--dependency=afterok:$job_id")
done
