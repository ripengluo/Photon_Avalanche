#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_ROOT [tm_npt_kmc_production.py options]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="$1"
shift
EXTENSION_STEPS="${EXTENSION_STEPS:-70000000}"

python -B "$SCRIPT_DIR/tm_npt_kmc_production.py" \
  --resume \
  --output-root "$RUN_ROOT" \
  --simulation-length "$EXTENSION_STEPS" \
  "$@"
