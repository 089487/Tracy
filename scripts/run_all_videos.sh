#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f "$HOME/myenv/bin/activate" ]; then
  source "$HOME/myenv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
else
  python3 -m venv .venv
  source ".venv/bin/activate"
  python -m pip install -r requirements.txt
fi

videos=()
for video in video_inputs/*.MOV video_inputs/*.mov; do
  [ -e "$video" ] || continue
  videos+=("$video")
done

total="${#videos[@]}"
index=0
for video in "${videos[@]}"; do
  index=$((index + 1))
  stem="$(basename "$video")"
  stem="${stem%.*}"
  out="video_output_${stem}"
  if [ -f "$out/.done" ] || { [ -s "$out/frame_overlay.mp4" ] && [ -s "$out/speed_overlay.mp4" ] && [ -s "$out/speed_summary.csv" ]; }; then
    echo "[$index/$total] Skipping $video -> $out (already completed)"
    continue
  fi
  echo "[$index/$total] Processing $video -> $out"
  python analyze_video_streamlines.py "$video" \
    --progress \
    --frame-step 1 \
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
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$out/.done"
done

echo "Finished all videos."
