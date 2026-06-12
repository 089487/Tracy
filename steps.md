# Wind Tunnel Similarity Analysis Steps

## 1. Research Question

Use the large model as an approximate real Dyson-like object and the small
model as a scale model. The report should answer:

> Can the small model represent the large model / real object flow field?

The evidence should come from three comparisons:

- Geometric similarity: `d/D`, `r_out/D`, `r_in/D`
- Dynamic similarity: `Re = rho * V * D / mu`
- Flow-field similarity: `W/D` and mean streamline shape

## 2. Required Measurements

Fill `measurements.csv` before writing the final report.

| Field | Meaning | Source |
|---|---|---|
| `model` | `large` or `small` | fixed label |
| `D` | outer diameter | caliper / CAD |
| `d` | inner diameter | caliper / CAD |
| `r_out` | outer rounding radius | CAD / measurement |
| `r_in` | inner diffuser/rounding radius | CAD / measurement |
| `V` | wind-tunnel velocity | tunnel setting / anemometer |
| `rho` | air density | default `1.2 kg/m^3` |
| `mu` | air viscosity | default `1.8e-5 Pa*s` |
| `W` | wake width | image/video extraction or manual measurement |

Use the same length unit for `D`, `d`, `r_out`, `r_in`, and `W`. If they are
pixel measurements, `W/D` is still valid as long as `W` and `D` come from the
same image scale.

## 3. Video Analysis Pipeline

Run a quick smoke test first:

```bash
source ~/myenv/bin/activate
python analyze_video_streamlines.py IMG_5622.MOV \
  --frame-step 1 \
  --max-frames 5 \
  --roi 1250,120,670,760 \
  --detector ridge \
  --ridge-threshold 5 \
  --speed-analysis \
  --speed-frame-gap 1 \
  --speed-step 1 \
  --overlay-video video_output_smoke/frame_overlay.mp4 \
  --speed-video video_output_smoke/speed_overlay.mp4 \
  --out-dir video_output_smoke
```

Then run the normal analysis:

```bash
source ~/myenv/bin/activate
python analyze_video_streamlines.py IMG_5622.MOV \
  --frame-step 1 \
  --roi 1250,120,670,760 \
  --detector ridge \
  --ridge-threshold 5 \
  --speed-analysis \
  --speed-frame-gap 1 \
  --speed-step 1 \
  --overlay-video video_output/frame_overlay.mp4 \
  --speed-video video_output/speed_overlay.mp4 \
  --out-dir video_output
```

Recommended tuning for the dense right-side streamlines:

```bash
python analyze_video_streamlines.py IMG_5622.MOV \
  --roi 1250,120,670,760 \
  --detector ridge \
  --ridge-threshold 5 \
  --stitch-gap 120 \
  --stitch-y-tolerance 6 \
  --track-y-tolerance 5 \
  --speed-analysis \
  --speed-frame-gap 1 \
  --speed-step 1 \
  --overlay-video video_output/frame_overlay.mp4 \
  --speed-video video_output/speed_overlay.mp4 \
  --out-dir video_output
```

If the body, support rod, or reflection is included, add an ROI:

```bash
python analyze_video_streamlines.py IMG_5622.MOV \
  --roi 550,120,1200,760 \
  --detector ridge \
  --ridge-threshold 5 \
  --out-dir video_output
```

## 4. Outputs From Video Analysis

The video script writes:

- `video_output/frame_streamlines.csv`: per-frame streamline points
- `video_output/mean_streamlines.csv`: time-averaged streamlines
- `video_output/wake_width_summary.csv`: wake width estimate `W`
- `video_output/deflection_summary.csv`: auxiliary streamline deflection metrics
- `video_output/speed_summary.csv`: optical-flow apparent speed summary
- `video_output/speed_samples.csv`: sampled velocity vectors in `px/s`
- `video_output/mean_overlay.png`: mean streamlines on a representative frame
- `video_output/wake_width_overlay.png`: wake-width bracket on the frame
- `video_output/speed_overlay.png`: velocity arrows over the frame
- `video_output/frame_overlay.mp4`: annotated processed-frame video
- `video_output/speed_overlay.mp4`: velocity-arrow video
- `video_output/deflection_plot.png`: `delta_y` curves

Use `wake_width_summary.csv` to fill the `W` column in `measurements.csv`, or
replace it with a manual measurement if the automatic bracket is not physically
reasonable.

If the automatic wake bracket spans too many unrelated streamlines, constrain
the measurement band:

```bash
python analyze_video_streamlines.py IMG_5622.MOV \
  --frame-step 1 \
  --max-frames 5 \
  --roi 1250,120,670,760 \
  --wake-x 1700 \
  --wake-y-range 350,760 \
  --detector ridge \
  --ridge-threshold 5 \
  --speed-analysis \
  --speed-frame-gap 1 \
  --speed-step 1 \
  --overlay-video video_output_smoke/frame_overlay.mp4 \
  --speed-video video_output_smoke/speed_overlay.mp4 \
  --out-dir video_output_smoke
```

Speed is reported in `px/s` by default. To convert it, provide a calibrated
length per pixel:

```bash
python analyze_video_streamlines.py IMG_5622.MOV \
  --roi 1250,120,670,760 \
  --speed-analysis \
  --length-per-px 0.00025 \
  --speed-unit m/s
```

## 5. Dimensionless Similarity Analysis

After filling `measurements.csv`, run:

```bash
python analyze_similarity.py measurements.csv --out similarity_summary.csv --plot video_output/dimensionless_comparison.png
```

The script calculates:

- `d_over_D = d / D`
- `r_out_over_D = r_out / D`
- `r_in_over_D = r_in / D`
- `Re = rho * V * D / mu`
- `W_over_D = W / D`
- large-vs-small percent differences when both rows are available

## 6. Report Interpretation

Use this logic in the report:

1. If `d/D`, `r_out/D`, and `r_in/D` are close, the two models are
   geometrically similar.
2. If `Re_large` and `Re_small` are close, the tests are dynamically similar.
3. If `W_large/D_large` and `W_small/D_small` are close, the observed wake
   scales similarly.
4. If all three are close, the small model can reasonably represent the large
   model / real-object flow pattern.
5. If `Re` is not close, explain that the small-model result is only a
   qualitative flow visualization unless the wind speed is adjusted.

## 7. Optional CFD Extension

CFD is optional and should support the report rather than replace the wind
tunnel evidence. A useful first version is 2D incompressible flow around the
large and small cross-sections.

Minimum CFD outputs:

- Streamlines
- Velocity contour
- Pressure contour
- Wake width `W`
- Reynolds number `Re`

Optional advanced output:

- Transient wake motion
- Turbulence model comparison
- Drag coefficient `C_D`
