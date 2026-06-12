#!/usr/bin/env python3
"""Build a dense report-style streamline map from separate curved and horizontal ROIs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from extract_streamlines import (
    extract_polylines,
    filter_polylines,
    make_streamline_mask,
    parse_angles,
    parse_roi,
    remove_small_components,
    report_polyline_points,
    smooth_and_downsample,
    stitch_polylines,
    zhang_suen_thinning,
)


def extract_region(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    detector: str,
    ridge_threshold: int,
    line_threshold: float,
    line_angles: list[float],
    min_length: float,
    min_horizontal_span: int,
    stitch_gap: float,
    stitch_y_tolerance: float,
    close_width: int,
    smooth_window: int,
) -> list[list[tuple[float, float]]]:
    x0, y0, w, h = roi
    work = image[y0 : y0 + h, x0 : x0 + w]
    mask = make_streamline_mask(
        work,
        detector=detector,
        green_score_threshold=35.0,
        ridge_threshold=ridge_threshold,
        ridge_height=31,
        line_threshold=line_threshold,
        line_length=35,
        line_angles=line_angles,
        min_value=15,
        saturation_threshold=8,
        blur=3,
        close_width=close_width,
    )
    mask = remove_small_components(mask, min_area=6)
    skeleton = zhang_suen_thinning(mask)
    raw = extract_polylines(skeleton, min_points=6)
    lines = filter_polylines(raw, min_length=min_length, min_horizontal_span=min_horizontal_span, horizontal_ratio=0)
    lines = stitch_polylines(
        lines,
        max_gap=stitch_gap,
        y_tolerance=stitch_y_tolerance,
        overlap_tolerance=55,
        iterations=4,
    )
    sampled = []
    for line in lines:
        smoothed = smooth_and_downsample(line, every=2, smooth_window=smooth_window)
        if len(smoothed) >= 2:
            sampled.append([(x + x0, y + y0) for x, y in smoothed])
    return sampled


def extend_horizontal_lines(
    lines: list[list[tuple[float, float]]],
    *,
    x_start: float,
    x_end: float,
    y_tolerance: float,
    point_step: float,
) -> list[list[tuple[float, float]]]:
    """Convert detected right-side fragments into full horizontal free-stream lines."""
    if not lines:
        return []

    y_values = sorted(float(np.median([p[1] for p in line])) for line in lines if len(line) >= 2)
    clusters: list[list[float]] = []
    for y in y_values:
        if not clusters or abs(np.median(clusters[-1]) - y) > y_tolerance:
            clusters.append([y])
        else:
            clusters[-1].append(y)

    xs = np.arange(x_start, x_end + point_step, point_step, dtype=np.float32)
    extended = []
    for cluster in clusters:
        y = float(np.median(cluster))
        extended.append([(float(x), y) for x in xs])
    return extended


def draw_report(image: np.ndarray, lines: list[list[tuple[float, float]]], out_path: Path, thickness: int) -> None:
    canvas = image.copy()
    dim = np.full_like(canvas, 20)
    canvas = cv2.addWeighted(canvas, 0.65, dim, 0.35, 0)
    for line in lines:
        local = [(x, y) for x, y in line]
        pts = report_polyline_points(local, 0, 0, spacing=2.0, smooth_window=13)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def draw_overlay(image: np.ndarray, lines: list[list[tuple[float, float]]], out_path: Path) -> None:
    overlay = image.copy()
    palette = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0)]
    for idx, line in enumerate(lines, start=1):
        pts = np.array([(int(round(x)), int(round(y))) for x, y in line], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(overlay, [pts], False, palette[idx % len(palette)], 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), overlay)


def write_csv(lines: list[list[tuple[float, float]]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["line_id", "point_index", "x_px", "y_px"])
        writer.writeheader()
        for line_id, line in enumerate(lines, start=1):
            for point_index, (x, y) in enumerate(line):
                writer.writerow(
                    {
                        "line_id": line_id,
                        "point_index": point_index,
                        "x_px": round(x, 3),
                        "y_px": round(y, 3),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a dense streamline report map with separate ROIs.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("image1_report_output"))
    parser.add_argument("--curved-roi", type=parse_roi, default=parse_roi("760,170,600,730"))
    parser.add_argument("--right-roi", type=parse_roi, default=parse_roi("1210,170,496,730"))
    parser.add_argument("--right-horizontal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--right-y-tolerance", type=float, default=4.0)
    parser.add_argument("--right-point-step", type=float, default=6.0)
    parser.add_argument("--report-thickness", type=int, default=1)
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    angles = parse_angles("-50,-40,-30,-20,-12,-6,0,6,12,20,30,40,50")
    curved = extract_region(
        image,
        args.curved_roi,
        detector="ridge_line",
        ridge_threshold=1,
        line_threshold=1.25,
        line_angles=angles,
        min_length=8,
        min_horizontal_span=3,
        stitch_gap=220,
        stitch_y_tolerance=14,
        close_width=7,
        smooth_window=17,
    )
    right = extract_region(
        image,
        args.right_roi,
        detector="ridge",
        ridge_threshold=1,
        line_threshold=99,
        line_angles=[0],
        min_length=18,
        min_horizontal_span=25,
        stitch_gap=260,
        stitch_y_tolerance=5,
        close_width=17,
        smooth_window=21,
    )
    if args.right_horizontal:
        right_x, _, right_w, _ = args.right_roi
        right = extend_horizontal_lines(
            right,
            x_start=right_x,
            x_end=image.shape[1] - 1,
            y_tolerance=args.right_y_tolerance,
            point_step=args.right_point_step,
        )

    lines = curved + right
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(lines, args.out_dir / "streamlines.csv")
    draw_overlay(image, lines, args.out_dir / "overlay.png")
    draw_report(image, lines, args.out_dir / "streamline_map.png", args.report_thickness)
    print(f"Curved region lines: {len(curved)}")
    print(f"Right horizontal lines: {len(right)}")
    print(f"Total lines: {len(lines)}")
    print(f"Wrote: {args.out_dir / 'streamline_map.png'}")


if __name__ == "__main__":
    main()
