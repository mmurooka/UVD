#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="."
CAMERA_NAME="front"
PYTHON_BIN="${PYTHON_BIN:-python}"
ANNOTATE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/annotate_compliance.py"

usage() {
  cat <<'EOF'
Usage:
  scripts/annotate_compliance_all.sh [ROOT_DIR] [--camera_name NAME] [--python_bin PYTHON] [-- <annotate_compliance.py args...>]

Examples:
  scripts/annotate_compliance_all.sh /path/to/dataset
  scripts/annotate_compliance_all.sh /path/to/dataset --camera_name wrist -- --task_description "The robot manipulates an object near obstacles."
EOF
}

EXTRA_ARGS=()
POSITIONAL=()
while (($# > 0)); do
  case "$1" in
    --camera_name)
      CAMERA_NAME="$2"
      shift 2
      ;;
    --python_bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if ((${#POSITIONAL[@]} > 1)); then
  usage
  exit 1
fi

if ((${#POSITIONAL[@]} == 1)); then
  ROOT_DIR="${POSITIONAL[0]}"
fi

PATTERN="${CAMERA_NAME}_rgb_image.rmb.segments.yaml"

mapfile -d '' SEGMENT_YAML_FILES < <(find "$ROOT_DIR" -type f -name "$PATTERN" -print0 | sort -z)

if ((${#SEGMENT_YAML_FILES[@]} == 0)); then
  echo "No segment YAML files found for pattern: $PATTERN under $ROOT_DIR" >&2
  exit 1
fi

echo "Found ${#SEGMENT_YAML_FILES[@]} segment YAML files for pattern: $PATTERN"

for segment_yaml in "${SEGMENT_YAML_FILES[@]}"; do
  echo "=== Annotating: $segment_yaml"
  "$PYTHON_BIN" "$ANNOTATE_SCRIPT" "$segment_yaml" "${EXTRA_ARGS[@]}"
done
