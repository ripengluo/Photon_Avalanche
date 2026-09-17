#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_ROOT [plot_n4_convergence.py options]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="$1"
shift

python -B "$SCRIPT_DIR/plot_n4_convergence.py" "$RUN_ROOT" "$@"
