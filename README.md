# Wind Tunnel Streamline Analysis

This project extracts wind-tunnel smoke / laser streamlines from images and
videos. The current practical goal is to produce useful visual evidence for a
fluid-mechanics report:

- mark streamline coordinates from photos or videos;
- compare large and small models using dimensionless quantities;
- estimate wake width `W`, apparent flow speed, and streamline deflection.

The report framing is:

> Use the large model as an approximate real Dyson-like object, use the small
> model as a scale model, and judge similarity using geometry, Reynolds number,
> and dimensionless wake width `W/D`.

## Environment

Use the prepared virtual environment:

```bash
source ~/myenv/bin/activate
```

The required Python libraries are already installed there.

## Main Files

| File | Purpose |
|---|---|
| `extract_streamlines.py` | Main single-image streamline detector. Produces colored overlays and coordinates. |
| `extract_streamline_report.py` | Experimental report-style image generator with separate curved and right-horizontal regions. |
| `analyze_video_streamlines.py` | Video pipeline: frame streamlines, mean streamlines, wake width, speed arrows, videos. |
| `analyze_similarity.py` | Computes `d/D`, `r_out/D`, `r_in/D`, `Re`, and `W/D` from `measurements.csv`. |
| `steps.md` | Report and experiment workflow notes. |
| `measurements.csv` | Input table for large/small model dimensions and wind speed. |

## Current Recommended Single-Image Baseline

For `image1.jpg`, the current best working baseline is the colored ridge
overlay:

```bash
source ~/myenv/bin/activate
python extract_streamlines.py image1.jpg \
  --out-dir image1_color_tuned_ridge5_len12 \
  --roi 1150,120,556,760 \
  --detector ridge \
  --ridge-threshold 5 \
  --ridge-height 31 \
  --min-value 20 \
  --min-saturation 15 \
  --horizontal-ratio 0 \
  --min-length 12 \
  --min-horizontal-span 8 \
  --close-width 9 \
  --stitch-gap 160 \
  --stitch-y-tolerance 7 \
  --stitch-overlap 35 \
  --sample-every 2 \
  --smooth-window 7
```

Important outputs:

- `image1_color_tuned_ridge5_len12/overlay.png`: colored detected streamlines.
- `image1_color_tuned_ridge5_len12/streamlines.csv`: point coordinates.
- `image1_color_tuned_ridge5_len12/streamlines.json`: grouped coordinates.
- `image1_color_tuned_ridge5_len12/mask.png`: detected bright-line mask.
- `image1_color_tuned_ridge5_len12/skeleton.png`: one-pixel centerlines.

Use `overlay.png` when tuning parameters. It is more honest than the black
report map because it shows what the detector actually found.

### Tracer

`extract_streamlines.py` now defaults to `--tracer peak`: it finds subpixel
brightness maxima in every image column of the ridge response and links them
into tracks using a slope-predicting tracker. This follows each individual
wavy line much more precisely than the old mask-skeleton path, which is still
available via `--tracer skeleton`.

The ridge response is also contrast-normalized by the local brightness
headroom (`--contrast-norm`, on by default). The right side of `image1.jpg`
has a bright green background, so the same streamline produces a much weaker
raw contrast there than on the darker left side; normalization lets one
`--ridge-threshold` work across both regions. Disable with
`--no-contrast-norm` to reproduce old behavior.

## ROI Notes

The ROI format is:

```text
--roi x,y,width,height
```

For `image1.jpg`, the image width is about `1706 px`. The current right-side
ROI:

```text
1150,120,556,760
```

means:

- start at `x = 1150`;
- include the right side through the image boundary;
- include the dense horizontal streamline region and the model wake region.

If the left side of the wake is missing, move `x` smaller. If too much unrelated
structure is included, move `x` larger or reduce the height.

## Threshold Tuning

For the right-side dense horizontal streamlines, `ridge` is usually better than
`line` or `hybrid`.

Useful settings:

| Setting | Effect |
|---|---|
| `--ridge-threshold 5` | Current baseline. Good balance between density and noise. |
| `--ridge-threshold 4` | Denser, but may add more short fragments near bright objects. |
| `--ridge-threshold 2` or `3` | Not always better. Lines can merge into blobs and skeletonize poorly. |
| `--min-length` higher | Removes short fragments, but may delete real short curved pieces. |
| `--stitch-y-tolerance` higher | Joins broken pieces more aggressively; can wrongly merge neighboring lines. |
| `--close-width` higher | Connects broken horizontal segments; can also merge adjacent lines. |

Recommended comparison commands:

```bash
python extract_streamlines.py image1.jpg \
  --out-dir image1_color_tuned_ridge4 \
  --roi 1150,120,556,760 \
  --detector ridge \
  --ridge-threshold 4 \
  --min-value 20 \
  --min-saturation 15 \
  --horizontal-ratio 0 \
  --min-length 10 \
  --min-horizontal-span 5 \
  --close-width 9 \
  --stitch-gap 160 \
  --stitch-y-tolerance 7 \
  --sample-every 2 \
  --smooth-window 7
```

```bash
python extract_streamlines.py image1.jpg \
  --out-dir image1_color_tuned_ridge4_len18 \
  --roi 1150,120,556,760 \
  --detector ridge \
  --ridge-threshold 4 \
  --min-value 20 \
  --min-saturation 15 \
  --horizontal-ratio 0 \
  --min-length 18 \
  --min-horizontal-span 12 \
  --close-width 9 \
  --stitch-gap 160 \
  --stitch-y-tolerance 7 \
  --sample-every 2 \
  --smooth-window 7
```

## Curved Region vs Right Horizontal Region

Physically, the image should be interpreted in two regions:

- **middle/right near the object**: streamlines may bend because the object
  disturbs the flow;
- **far right**: streamlines should be nearly horizontal free-stream lines.

For report-style visuals, `extract_streamline_report.py` supports splitting
these regions:

```bash
python extract_streamline_report.py image1.jpg \
  --out-dir image1_report_curve_then_right_horizontal \
  --curved-roi 700,170,720,730 \
  --right-roi 1280,170,426,730 \
  --right-horizontal \
  --right-y-tolerance 3.5 \
  --right-point-step 5 \
  --report-thickness 2
```

This creates:

- curved detections in the middle region;
- horizontal free-stream lines on the far right;
- `streamline_map.png`, a black report-style visualization.

This report-style map is useful for explanation, but it should be described as
a reconstructed / qualitative streamline map, not as raw detection only.

## Video Smoke Test

The videos are under `video_inputs/*.MOV`. Run a short smoke test first:

```bash
source ~/myenv/bin/activate
python analyze_video_streamlines.py video_inputs/IMG_5616.MOV \
  --frame-step 1 \
  --max-frames 5 \
  --roi 1150,120,556,760 \
  --detector ridge \
  --ridge-threshold 5 \
  --min-value 20 \
  --min-saturation 15 \
  --horizontal-ratio 0 \
  --min-length 12 \
  --min-horizontal-span 8 \
  --close-width 9 \
  --stitch-gap 160 \
  --stitch-y-tolerance 7 \
  --speed-analysis \
  --speed-frame-gap 1 \
  --speed-step 1 \
  --speed-flow-direction left \
  --overlay-video video_output_smoke/frame_overlay.mp4 \
  --speed-video video_output_smoke/speed_overlay.mp4 \
  --out-dir video_output_smoke
```

Speed direction note:

- The physical flow direction in these images is treated as right-to-left.
- Use `--speed-flow-direction left`.
- If arrows point right in the free stream, the frame-pair direction or sign is
  wrong; the script can flip the optical-flow direction to match the expected
  leftward flow.

## Full Video Run

For one video:

```bash
python analyze_video_streamlines.py video_inputs/IMG_5616.MOV \
  --frame-step 1 \
  --roi 1150,120,556,760 \
  --detector ridge \
  --ridge-threshold 5 \
  --min-value 20 \
  --min-saturation 15 \
  --horizontal-ratio 0 \
  --min-length 12 \
  --min-horizontal-span 8 \
  --close-width 9 \
  --stitch-gap 160 \
  --stitch-y-tolerance 7 \
  --speed-analysis \
  --speed-frame-gap 1 \
  --speed-step 1 \
  --speed-flow-direction left \
  --overlay-video video_output_IMG_5616/frame_overlay.mp4 \
  --speed-video video_output_IMG_5616/speed_overlay.mp4 \
  --out-dir video_output_IMG_5616
```

For all videos, use:

```bash
bash scripts/run_all_videos.sh
```

The batch script is designed to skip output folders that already finished.

## Video Outputs

The video script writes:

| Output | Meaning |
|---|---|
| `frame_streamlines.csv` | Streamline points for each processed frame. |
| `mean_streamlines.csv` | Time-averaged streamline tracks. |
| `wake_width_summary.csv` | Estimated wake width `W`. |
| `deflection_summary.csv` | Auxiliary streamline deflection metrics. |
| `speed_summary.csv` | Optical-flow apparent speed summary. |
| `speed_samples.csv` | Grid-level speed vector samples. |
| `mean_overlay.png` | Mean streamlines on one representative frame. |
| `wake_width_overlay.png` | Wake width bracket on one representative frame. |
| `speed_overlay.png` | Velocity arrows on one representative frame. |
| `frame_overlay.mp4` | Streamline annotations through time. |
| `speed_overlay.mp4` | Velocity-arrow video. |

Speed is in `px/s` unless a calibration is provided:

```bash
python analyze_video_streamlines.py video_inputs/IMG_5616.MOV \
  --speed-analysis \
  --length-per-px 0.00025 \
  --speed-unit m/s
```

## Similarity Analysis for Report

Fill `measurements.csv`:

```csv
model,D,d,r_out,r_in,V,rho,mu,W
large,,,,, ,1.2,1.8e-5,
small,,,,, ,1.2,1.8e-5,
```

Then run:

```bash
python analyze_similarity.py measurements.csv \
  --out similarity_summary.csv \
  --plot video_output/dimensionless_comparison.png
```

The script computes:

- `d/D`
- `r_out/D`
- `r_in/D`
- `Re = rho * V * D / mu`
- `W/D`
- large-vs-small percent differences when both rows exist

## Report Interpretation

Use this logic:

1. Check geometric similarity using `d/D`, `r_out/D`, and `r_in/D`.
2. Check dynamic similarity using Reynolds number.
3. Check flow similarity using `W/D` and streamline shape.
4. If geometry, `Re`, and `W/D` are close, the small model can reasonably
   represent the large model.
5. If `Re` is not close, treat the small-model result as qualitative unless
   wind speed is adjusted.

Recommended report wording:

> The large model is used as an approximate full-scale Dyson-like geometry,
> while the small model is treated as a scale model. The comparison is based on
> geometric similarity, Reynolds number similarity, and dimensionless wake width
> `W/D`. Streamline overlays provide qualitative flow-field evidence, while
> `W/D` and `Re` provide dimensionless quantitative comparison.

## Practical Notes

- Use colored `overlay.png` for detector tuning.
- Use black `streamline_map.png` only for qualitative report illustrations.
- Do not claim that the black report map is raw measurement only if horizontal
  lines were extended or reconstructed.
- Right-side free-stream lines should point / flow leftward in velocity plots.
- For dense horizontal smoke lines, `ridge` detector is usually the safest
  first choice.
