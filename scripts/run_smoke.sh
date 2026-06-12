#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f "$HOME/myenv/bin/activate" ]; then
  # Local Mac environment.
  source "$HOME/myenv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
  # Remote ws2 environment.
  source ".venv/bin/activate"
else
  python3 -m venv .venv
  source ".venv/bin/activate"
  python -m pip install -r requirements.txt
fi

video="${1:-video_inputs/IMG_5618.MOV}"
out="${2:-video_output_frame_smoke}"

python analyze_video_streamlines.py "$video" \
  --progress \
  --frame-step 1 \
  --max-frames 5 \
  --roi 1250,120,670,760 \
  --detector ridge \
  --ridge-threshold 5 \
  --wake-x 1700 \
  --wake-y-range 250,760 \
  --speed-analysis \
  --speed-frame-gap 1 \
  --speed-step 1 \
  --speed-flow-direction left \
  --speed-max-angle-deg 75 \
  --overlay-video "$out/frame_overlay.mp4" \
  --speed-video "$out/speed_overlay.mp4" \
  --out-dir "$out"

echo "Smoke outputs written to $out"
