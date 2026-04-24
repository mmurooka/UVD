#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="."
CAMERA_NAME="front"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEGMENT_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/segment_video.py"

usage() {
  cat <<'EOF'
Usage:
  scripts/segment_all_videos.sh [ROOT_DIR] [--camera_name NAME] [--python_bin PYTHON] [-- <segment_video.py args...>]

Examples:
  scripts/segment_all_videos.sh /path/to/dataset
  scripts/segment_all_videos.sh /path/to/dataset --camera_name wrist -- --segmentation_algorithm kernel_cpd
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

PATTERN="${CAMERA_NAME}_rgb_image.rmb.mp4"

mapfile -d '' VIDEO_FILES < <(find "$ROOT_DIR" -type f -name "$PATTERN" -print0 | sort -z)

if ((${#VIDEO_FILES[@]} == 0)); then
  echo "No videos found for pattern: $PATTERN under $ROOT_DIR" >&2
  exit 1
fi

echo "Found ${#VIDEO_FILES[@]} videos for pattern: $PATTERN"

for video_file in "${VIDEO_FILES[@]}"; do
  echo "=== Segmenting: $video_file"
  "$PYTHON_BIN" "$SEGMENT_SCRIPT" "$video_file" "${EXTRA_ARGS[@]}"
done
