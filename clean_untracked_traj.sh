#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/rpluo/Desktop/project_MFML_UCNP/kmc_ANP/Tm_4p5-NPT"
ARCHIVE_ROOT="/home/rpluo/Desktop/hdd_large/KMC_trajectories/Tm_4p5-NPT"
DELETE=0

usage() {
  cat <<'EOF'
Usage: ./clean_untracked_traj.sh [--delete] [--work-root PATH] [--archive-root PATH]

Compare archived kMC trajectory sqlite files against current initial_state.sqlite
symlink targets in the working tree.

Default mode is dry-run. Pass --delete to remove untracked archive files.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --delete)
      DELETE=1
      shift
      ;;
    --work-root)
      WORK_ROOT="$2"
      shift 2
      ;;
    --archive-root)
      ARCHIVE_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$WORK_ROOT" ]]; then
  echo "Work root does not exist: $WORK_ROOT" >&2
  exit 1
fi
if [[ ! -d "$ARCHIVE_ROOT" ]]; then
  echo "Archive root does not exist: $ARCHIVE_ROOT" >&2
  exit 1
fi

declare -A linked_targets=()
while IFS= read -r -d '' link_path; do
  target="$(readlink -f "$link_path" || true)"
  if [[ -n "$target" && "$target" == "$ARCHIVE_ROOT"/* ]]; then
    linked_targets["$target"]=1
  fi
done < <(find "$WORK_ROOT" -type l -name initial_state.sqlite -print0)

tracked_count="${#linked_targets[@]}"
untracked_count=0

while IFS= read -r -d '' archive_path; do
  resolved_archive_path="$(readlink -f "$archive_path" || true)"
  if [[ -z "$resolved_archive_path" ]]; then
    continue
  fi
  if [[ -n "${linked_targets[$resolved_archive_path]+x}" ]]; then
    continue
  fi

  untracked_count=$((untracked_count + 1))
  if [[ "$DELETE" -eq 1 ]]; then
    rm -f -- "$archive_path"
    echo "deleted $archive_path"
  else
    echo "dry-run would delete $archive_path"
  fi
done < <(find "$ARCHIVE_ROOT" -type f -name '*.sqlite' -print0)

echo "Tracked archive files: $tracked_count"
if [[ "$DELETE" -eq 1 ]]; then
  echo "Deleted untracked archive files: $untracked_count"
else
  echo "Untracked archive files: $untracked_count"
  echo "Run with --delete to remove them."
fi
