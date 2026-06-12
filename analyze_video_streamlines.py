#!/usr/bin/env python3
"""Analyze wind-tunnel streamline video frames and produce report-ready outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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
    smooth_and_downsample,
    stitch_polylines,
    zhang_suen_thinning,
)


PALETTE = [
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (0, 180, 255),
    (180, 0, 255),
]


def print_progress(label: str, current: int, total: int, *, width: int = 32) -> None:
    if total <= 0:
        return
    current = min(current, total)
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = 100 * current / total
    sys.stderr.write(f"\r{label} [{bar}] {current}/{total} ({pct:5.1f}%)")
    if current >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def parse_roi_y_range(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    parts = [float(v.strip()) for v in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--wake-y-range must be y_min,y_max")
    y_min, y_max = parts
    if y_max <= y_min:
        raise argparse.ArgumentTypeError("--wake-y-range y_max must be greater than y_min")
    return y_min, y_max


def process_frame(frame: np.ndarray, args: argparse.Namespace) -> list[list[tuple[float, float]]]:
    work = frame
    x0, y0 = 0, 0
    if args.roi:
        x0, y0, w, h = args.roi
        work = frame[y0 : y0 + h, x0 : x0 + w]

    mask = make_streamline_mask(
        work,
        detector=args.detector,
        green_score_threshold=args.green_score,
        ridge_threshold=args.ridge_threshold,
        ridge_height=args.ridge_height,
        line_threshold=args.line_threshold,
        line_length=args.line_length,
        line_angles=args.line_angles,
        min_value=args.min_value,
        saturation_threshold=args.min_saturation,
        blur=args.blur,
        close_width=args.close_width,
    )
    mask = remove_small_components(mask, args.min_area, args.max_area, args.max_fill_ratio)
    skeleton = zhang_suen_thinning(mask)
    raw_lines = extract_polylines(skeleton, args.min_points)
    lines = filter_polylines(
        raw_lines,
        min_length=args.min_length,
        min_horizontal_span=args.min_horizontal_span,
        horizontal_ratio=args.horizontal_ratio,
    )
    lines = stitch_polylines(
        lines,
        max_gap=args.stitch_gap,
        y_tolerance=args.stitch_y_tolerance,
        overlap_tolerance=args.stitch_overlap,
        iterations=args.stitch_iterations,
    )

    sampled = []
    for line in lines:
        if len(line) < 2:
            continue
        smoothed = smooth_and_downsample(line, every=args.sample_every, smooth_window=args.smooth_window)
        if len(smoothed) < 2:
            continue
        sampled.append([(x + x0, y + y0) for x, y in smoothed])
    return sampled


def line_summary(line: list[tuple[float, float]]) -> dict[str, float]:
    if not line:
        return {"x_min": 0.0, "x_max": 0.0, "y_median": 0.0, "y_mean": 0.0, "points": 0.0}
    xs = np.array([p[0] for p in line], dtype=np.float32)
    ys = np.array([p[1] for p in line], dtype=np.float32)
    return {
        "x_min": float(xs.min()),
        "x_max": float(xs.max()),
        "y_median": float(np.median(ys)),
        "y_mean": float(ys.mean()),
        "points": float(len(line)),
    }


def draw_frame_overlay(frame: np.ndarray, lines: list[list[tuple[float, float]]], frame_index: int, time_sec: float) -> np.ndarray:
    overlay = frame.copy()
    for line_id, line in enumerate(lines, start=1):
        if len(line) < 2:
            continue
        pts = np.array([(int(round(x)), int(round(y))) for x, y in line], dtype=np.int32)
        color = PALETTE[line_id % len(PALETTE)]
        cv2.polylines(overlay, [pts], False, color, 2)
        mid = tuple(pts[len(pts) // 2])
        cv2.putText(overlay, str(line_id), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        overlay,
        f"frame={frame_index}  t={time_sec:.3f}s",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def assign_tracks(detections: list[dict], y_tolerance: float) -> list[dict]:
    ordered = sorted(detections, key=lambda d: (d["summary"]["y_median"], d["summary"]["x_min"]))
    clusters: list[list[dict]] = []
    cluster_medians: list[float] = []

    for detection in ordered:
        y = detection["summary"]["y_median"]
        best_idx = None
        best_gap = None
        for idx, median_y in enumerate(cluster_medians):
            gap = abs(y - median_y)
            if gap <= y_tolerance and (best_gap is None or gap < best_gap):
                best_idx = idx
                best_gap = gap
        if best_idx is None:
            clusters.append([detection])
            cluster_medians.append(y)
        else:
            clusters[best_idx].append(detection)
            cluster_medians[best_idx] = float(np.median([d["summary"]["y_median"] for d in clusters[best_idx]]))

    clusters.sort(key=lambda cluster: np.median([d["summary"]["y_median"] for d in cluster]))
    for track_id, cluster in enumerate(clusters, start=1):
        for detection in cluster:
            detection["track_id"] = track_id
    return detections


def interpolate_line(line: list[tuple[float, float]], grid: np.ndarray) -> np.ndarray:
    by_x: dict[float, list[float]] = defaultdict(list)
    for x, y in line:
        by_x[float(x)].append(float(y))
    xs_list = sorted(by_x.keys())
    xs = np.array(xs_list, dtype=np.float32)
    ys = np.array([np.mean(by_x[x]) for x in xs_list], dtype=np.float32)
    if len(xs) < 2:
        return np.full_like(grid, np.nan, dtype=np.float32)
    values = np.interp(grid, xs, ys).astype(np.float32)
    values[(grid < xs.min()) | (grid > xs.max())] = np.nan
    return values


def build_mean_streamlines(
    detections: list[dict],
    x_step: int,
    min_track_frames: int,
) -> tuple[list[dict], list[dict]]:
    by_track: dict[int, list[dict]] = defaultdict(list)
    for detection in detections:
        by_track[int(detection["track_id"])].append(detection)

    point_rows = []
    track_summaries = []
    for track_id, items in sorted(by_track.items()):
        if len({item["frame_index"] for item in items}) < min_track_frames:
            continue
        x_min = min(item["summary"]["x_min"] for item in items)
        x_max = max(item["summary"]["x_max"] for item in items)
        if x_max - x_min < x_step:
            continue
        grid = np.arange(x_min, x_max + 0.1, x_step, dtype=np.float32)
        values = np.vstack([interpolate_line(item["line"], grid) for item in items])
        covered_columns = np.any(~np.isnan(values), axis=0)
        if not np.any(covered_columns):
            continue
        grid = grid[covered_columns]
        values = values[:, covered_columns]
        mean_y = np.nanmean(values, axis=0)
        std_y = np.nanstd(values, axis=0)
        n_frames = np.sum(~np.isnan(values), axis=0)

        valid = n_frames > 0
        if not np.any(valid):
            continue
        point_index = 0
        for idx, x in enumerate(grid):
            if not valid[idx]:
                continue
            point_rows.append(
                {
                    "track_id": track_id,
                    "point_index": point_index,
                    "x_px": float(x),
                    "mean_y_px": float(mean_y[idx]),
                    "std_y_px": float(std_y[idx]),
                    "n_frames": int(n_frames[idx]),
                }
            )
            point_index += 1
        ys = [row["mean_y_px"] for row in point_rows if row["track_id"] == track_id]
        xs = [row["x_px"] for row in point_rows if row["track_id"] == track_id]
        track_summaries.append(
            {
                "track_id": track_id,
                "x_min": min(xs),
                "x_max": max(xs),
                "y_median": float(np.median(ys)),
                "frames_seen": len({item["frame_index"] for item in items}),
            }
        )
    return point_rows, track_summaries


def write_frame_streamlines(path: Path, detections: list[dict]) -> None:
    fieldnames = [
        "frame_index",
        "time_sec",
        "track_id",
        "line_id_in_frame",
        "point_index",
        "x_px",
        "y_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for detection in sorted(detections, key=lambda d: (d["frame_index"], d["track_id"], d["line_id_in_frame"])):
            for point_index, (x, y) in enumerate(detection["line"]):
                writer.writerow(
                    {
                        "frame_index": detection["frame_index"],
                        "time_sec": f"{detection['time_sec']:.6f}",
                        "track_id": detection["track_id"],
                        "line_id_in_frame": detection["line_id_in_frame"],
                        "point_index": point_index,
                        "x_px": f"{x:.3f}",
                        "y_px": f"{y:.3f}",
                    }
                )


def write_mean_streamlines(path: Path, rows: list[dict]) -> None:
    fieldnames = ["track_id", "point_index", "x_px", "mean_y_px", "std_y_px", "n_frames"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "track_id": row["track_id"],
                    "point_index": row["point_index"],
                    "x_px": f"{row['x_px']:.3f}",
                    "mean_y_px": f"{row['mean_y_px']:.3f}",
                    "std_y_px": f"{row['std_y_px']:.3f}",
                    "n_frames": row["n_frames"],
                }
            )


def write_mean_json(path: Path, rows: list[dict]) -> None:
    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[int(row["track_id"])].append(row)
    payload = [
        {
            "track_id": track_id,
            "points": [
                {
                    "x_px": round(point["x_px"], 3),
                    "mean_y_px": round(point["mean_y_px"], 3),
                    "std_y_px": round(point["std_y_px"], 3),
                    "n_frames": point["n_frames"],
                }
                for point in points
            ],
        }
        for track_id, points in sorted(by_track.items())
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def draw_mean_overlay(frame: np.ndarray, rows: list[dict], path: Path) -> None:
    overlay = frame.copy()
    by_track: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        by_track[int(row["track_id"])].append((row["x_px"], row["mean_y_px"]))
    for track_id, points in sorted(by_track.items()):
        if len(points) < 2:
            continue
        pts = np.array([(round(x), round(y)) for x, y in points], dtype=np.int32)
        color = PALETTE[track_id % len(PALETTE)]
        cv2.polylines(overlay, [pts], False, color, 2)
        mid = tuple(pts[len(pts) // 2])
        cv2.putText(overlay, str(track_id), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), overlay)


def compute_wake_width(
    rows: list[dict],
    wake_x: float | None,
    frame_width: int,
    roi: tuple[int, int, int, int] | None,
    x_tolerance: float,
    wake_y_range: tuple[float, float] | None,
    percentile_low: float,
    percentile_high: float,
) -> dict[str, float | int | str]:
    if wake_x is None:
        if roi:
            wake_x = roi[0] + roi[2] * 0.75
        else:
            wake_x = frame_width * 0.75

    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[int(row["track_id"])].append(row)

    y_values = []
    used_tracks = []
    for track_id, points in by_track.items():
        close = [p for p in points if abs(p["x_px"] - wake_x) <= x_tolerance]
        if not close:
            continue
        best = min(close, key=lambda p: abs(p["x_px"] - wake_x))
        if wake_y_range and not (wake_y_range[0] <= best["mean_y_px"] <= wake_y_range[1]):
            continue
        y_values.append(best["mean_y_px"])
        used_tracks.append(track_id)

    if len(y_values) < 2:
        return {
            "wake_x_px": float(wake_x),
            "W_px": "",
            "y_top_px": "",
            "y_bottom_px": "",
            "tracks_used": len(y_values),
            "note": "Not enough streamlines crossed wake_x; adjust --wake-x or ROI.",
        }

    y_top = float(np.percentile(y_values, percentile_low))
    y_bottom = float(np.percentile(y_values, percentile_high))
    return {
        "wake_x_px": float(wake_x),
        "W_px": float(y_bottom - y_top),
        "y_top_px": float(y_top),
        "y_bottom_px": float(y_bottom),
        "tracks_used": len(used_tracks),
        "note": "Automatic wake width from selected mean-streamline spread at wake_x; verify against overlay.",
    }


def write_wake_summary(path: Path, wake: dict[str, float | int | str]) -> None:
    fieldnames = ["wake_x_px", "W_px", "y_top_px", "y_bottom_px", "tracks_used", "note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(wake)


def draw_wake_overlay(frame: np.ndarray, mean_rows: list[dict], wake: dict[str, float | int | str], path: Path) -> None:
    overlay = frame.copy()
    draw_mean_overlay(frame, mean_rows, path)
    overlay = cv2.imread(str(path))
    if not wake.get("W_px"):
        cv2.imwrite(str(path), overlay)
        return
    x = int(round(float(wake["wake_x_px"])))
    y_top = int(round(float(wake["y_top_px"])))
    y_bottom = int(round(float(wake["y_bottom_px"])))
    cv2.line(overlay, (x, y_top), (x, y_bottom), (255, 255, 255), 3)
    cv2.line(overlay, (x - 30, y_top), (x + 30, y_top), (255, 255, 255), 3)
    cv2.line(overlay, (x - 30, y_bottom), (x + 30, y_bottom), (255, 255, 255), 3)
    cv2.putText(
        overlay,
        f"W={float(wake['W_px']):.1f}px",
        (x + 12, (y_top + y_bottom) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), overlay)


def compute_deflection(rows: list[dict], ref_fraction: float) -> tuple[list[dict], list[dict]]:
    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[int(row["track_id"])].append(row)

    curve_rows = []
    summary_rows = []
    for track_id, points in sorted(by_track.items()):
        points = sorted(points, key=lambda p: p["x_px"])
        x_min = points[0]["x_px"]
        x_max = points[-1]["x_px"]
        ref_limit = x_min + (x_max - x_min) * ref_fraction
        ref_points = [p for p in points if p["x_px"] <= ref_limit]
        if not ref_points:
            ref_points = points[: max(1, len(points) // 5)]
        y_ref = float(np.mean([p["mean_y_px"] for p in ref_points]))

        deltas = []
        for point in points:
            delta = point["mean_y_px"] - y_ref
            deltas.append(delta)
            curve_rows.append(
                {
                    "track_id": track_id,
                    "x_px": point["x_px"],
                    "mean_y_px": point["mean_y_px"],
                    "y_ref_px": y_ref,
                    "delta_y_px": delta,
                    "std_y_px": point["std_y_px"],
                    "n_frames": point["n_frames"],
                }
            )
        max_idx = int(np.argmax(np.abs(deltas)))
        summary_rows.append(
            {
                "track_id": track_id,
                "y_ref_px": y_ref,
                "max_abs_delta_y_px": float(abs(deltas[max_idx])),
                "x_at_max_delta_px": points[max_idx]["x_px"],
                "mean_abs_delta_y_px": float(np.mean(np.abs(deltas))),
            }
        )
    return curve_rows, summary_rows


def write_deflection(path: Path, curve_path: Path, curves: list[dict], summaries: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["track_id", "y_ref_px", "max_abs_delta_y_px", "x_at_max_delta_px", "mean_abs_delta_y_px"],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: f"{value:.3f}" if isinstance(value, float) else value for key, value in row.items()})

    with curve_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["track_id", "x_px", "mean_y_px", "y_ref_px", "delta_y_px", "std_y_px", "n_frames"],
        )
        writer.writeheader()
        for row in curves:
            writer.writerow({key: f"{value:.3f}" if isinstance(value, float) else value for key, value in row.items()})


def draw_deflection_plot(curves: list[dict], path: Path) -> None:
    width, height = 1300, 760
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Streamline Deflection: delta_y(x)", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    if not curves:
        cv2.imwrite(str(path), img)
        return

    xs = np.array([row["x_px"] for row in curves], dtype=np.float32)
    ds = np.array([row["delta_y_px"] for row in curves], dtype=np.float32)
    x_min, x_max = float(xs.min()), float(xs.max())
    d_abs = max(float(np.max(np.abs(ds))), 1.0)

    left, top, plot_w, plot_h = 80, 100, 1140, 560
    center_y = top + plot_h // 2
    cv2.rectangle(img, (left, top), (left + plot_w, top + plot_h), (0, 0, 0), 1)
    cv2.line(img, (left, center_y), (left + plot_w, center_y), (190, 190, 190), 1)

    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in curves:
        by_track[int(row["track_id"])].append(row)

    for track_id, points in sorted(by_track.items()):
        if len(points) < 2:
            continue
        pts = []
        for point in sorted(points, key=lambda p: p["x_px"]):
            x = left + int((point["x_px"] - x_min) / max(x_max - x_min, 1) * plot_w)
            y = center_y - int(point["delta_y_px"] / d_abs * (plot_h * 0.45))
            pts.append((x, y))
        color = PALETTE[track_id % len(PALETTE)]
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 1)

    cv2.putText(img, f"x: {x_min:.0f}..{x_max:.0f}px", (left, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, f"delta_y range: +/-{d_abs:.1f}px", (left + 360, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


def crop_frame(frame: np.ndarray, roi: tuple[int, int, int, int] | None) -> tuple[np.ndarray, int, int]:
    if not roi:
        return frame, 0, 0
    x, y, w, h = roi
    return frame[y : y + h, x : x + w], x, y


def speed_mask(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    mask = make_streamline_mask(
        frame,
        detector=args.detector,
        green_score_threshold=args.green_score,
        ridge_threshold=args.ridge_threshold,
        ridge_height=args.ridge_height,
        line_threshold=args.line_threshold,
        line_length=args.line_length,
        line_angles=args.line_angles,
        min_value=args.min_value,
        saturation_threshold=args.min_saturation,
        blur=args.blur,
        close_width=args.close_width,
    )
    mask = remove_small_components(mask, args.min_area, args.max_area, args.max_fill_ratio)
    if args.speed_y_range:
        y_min, y_max = args.speed_y_range
        band = np.zeros_like(mask)
        band[int(y_min) : int(y_max) + 1, :] = 255
        mask = cv2.bitwise_and(mask, band)
    return mask


def compute_speed_analysis(args: argparse.Namespace) -> tuple[list[dict], dict[str, float | int | str], np.ndarray | None]:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        return [], {"note": "Could not open video for speed analysis."}, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(args.start_sec * fps))
    end_frame = total_frames if args.end_sec is None else min(total_frames, int(args.end_sec * fps))
    gap = max(args.speed_frame_gap, 1)
    step = max(args.speed_step, gap)
    samples: list[dict] = []
    overlay_frame = None
    pair_indices = list(range(start_frame, max(start_frame, end_frame - gap), step))
    if args.max_frames:
        pair_indices = pair_indices[: args.max_frames]

    processed_pairs = 0
    for pair_i, frame_index in enumerate(pair_indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok1, frame1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index + gap)
        ok2, frame2 = cap.read()
        if not (ok1 and ok2):
            continue
        processed_pairs += 1

        roi1, x0, y0 = crop_frame(frame1, args.roi)
        roi2, _, _ = crop_frame(frame2, args.roi)
        if overlay_frame is None:
            overlay_frame = frame1.copy()

        gray1 = cv2.GaussianBlur(cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        gray2 = cv2.GaussianBlur(cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=args.speed_window,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mask = speed_mask(roi1, args) > 0
        if args.speed_min_green_pixels > 0 and int(mask.sum()) < args.speed_min_green_pixels:
            continue

        h, w = mask.shape
        cell = max(args.speed_grid, 8)
        dt = gap / fps
        for cy in range(cell // 2, h, cell):
            for cx in range(cell // 2, w, cell):
                y1 = max(0, cy - cell // 2)
                y2 = min(h, cy + cell // 2)
                x1 = max(0, cx - cell // 2)
                x2 = min(w, cx + cell // 2)
                cell_mask = mask[y1:y2, x1:x2]
                if cell_mask.mean() < args.speed_mask_fraction:
                    continue
                vectors = flow[y1:y2, x1:x2][cell_mask]
                if len(vectors) == 0:
                    continue
                u = float(np.median(vectors[:, 0]))
                v = float(np.median(vectors[:, 1]))
                row = {
                    "frame_index": frame_index,
                    "time_sec": frame_index / fps,
                    "x_px": x0 + cx,
                    "y_px": y0 + cy,
                    "u_px_s": u / dt,
                    "v_px_s": v / dt,
                    "raw_u_px_s": u / dt,
                    "raw_v_px_s": v / dt,
                }
                samples.append(row)
        if args.progress:
            print_progress("Speed pairs", pair_i, len(pair_indices))

    cap.release()
    samples, correction_note = orient_and_filter_speed_samples(samples, args)
    if not samples:
        return [], {"note": "No valid optical-flow speed samples found; adjust ROI, direction, or angle filter."}, overlay_frame

    speeds = np.array([float(row["speed_px_s"]) for row in samples], dtype=np.float32)
    streamwise = np.array([float(row["streamwise_px_s"]) for row in samples], dtype=np.float32)
    summary: dict[str, float | int | str] = {
        "samples": len(samples),
        "frame_gap": gap,
        "dt_sec": gap / fps,
        "median_speed_px_s": float(np.median(speeds)),
        "mean_speed_px_s": float(np.mean(speeds)),
        "median_streamwise_px_s": float(np.median(streamwise)),
        "mean_streamwise_px_s": float(np.mean(streamwise)),
        "p10_speed_px_s": float(np.percentile(speeds, 10)),
        "p90_speed_px_s": float(np.percentile(speeds, 90)),
        "flow_direction": args.speed_flow_direction,
        "direction_correction": correction_note,
        "length_per_px": args.length_per_px if args.length_per_px else "",
        "speed_unit": args.speed_unit if args.length_per_px else "px/s",
        "note": "Dense optical flow on green streamline pixels, oriented to the expected physical free-stream direction.",
    }
    if args.length_per_px:
        summary["median_speed_units_s"] = float(np.median(speeds) * args.length_per_px)
        summary["mean_speed_units_s"] = float(np.mean(speeds) * args.length_per_px)
        summary["median_streamwise_units_s"] = float(np.median(streamwise) * args.length_per_px)
        summary["mean_streamwise_units_s"] = float(np.mean(streamwise) * args.length_per_px)

    if overlay_frame is not None:
        scale = args.speed_vector_scale
        for row in samples[: args.speed_overlay_vectors]:
            x = float(row["x_px"])
            y = float(row["y_px"])
            u = float(row["u_px_s"])
            v = float(row["v_px_s"])
            arrow_len = float(np.hypot(u * scale, v * scale))
            if args.speed_overlay_max_arrow_px > 0 and arrow_len > args.speed_overlay_max_arrow_px:
                shrink = args.speed_overlay_max_arrow_px / arrow_len
                u *= shrink
                v *= shrink
            end = (int(round(x + u * scale)), int(round(y + v * scale)))
            cv2.arrowedLine(overlay_frame, (int(x), int(y)), end, (255, 255, 255), 1, tipLength=0.3)
        label = f"median streamwise={summary['median_streamwise_px_s']:.1f} px/s"
        if args.length_per_px:
            label += f" ({summary['median_streamwise_units_s']:.3g} {args.speed_unit})"
        cv2.putText(overlay_frame, label, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return samples, summary, overlay_frame


def orient_and_filter_speed_samples(samples: list[dict], args: argparse.Namespace) -> tuple[list[dict], str]:
    if not samples:
        return [], "no_samples"

    raw_u = np.array([float(row["u_px_s"]) for row in samples], dtype=np.float32)
    expected_sign = 0
    if args.speed_flow_direction == "left":
        expected_sign = -1
    elif args.speed_flow_direction == "right":
        expected_sign = 1
    elif args.speed_flow_direction == "auto":
        expected_sign = -1 if float(np.median(raw_u)) <= 0 else 1

    correction = "raw"
    if expected_sign:
        median_u = float(np.median(raw_u))
        if median_u * expected_sign < 0:
            for row in samples:
                row["u_px_s"] = -float(row["u_px_s"])
                row["v_px_s"] = -float(row["v_px_s"])
            correction = "flipped_180deg_to_match_expected_direction"
        else:
            correction = "kept_original_direction"

    filtered = []
    for row in samples:
        u = float(row["u_px_s"])
        v = float(row["v_px_s"])
        if expected_sign < 0:
            streamwise = -u
            crossflow = v
        elif expected_sign > 0:
            streamwise = u
            crossflow = v
        else:
            streamwise = float(np.hypot(u, v))
            crossflow = v

        if args.speed_filter_wrong_direction and expected_sign and streamwise < 0:
            continue
        angle = math.degrees(math.atan2(abs(crossflow), max(abs(streamwise), 1e-9)))
        if args.speed_max_angle_deg < 180 and angle > args.speed_max_angle_deg:
            continue

        row["streamwise_px_s"] = streamwise
        row["crossflow_px_s"] = crossflow
        row["flow_angle_deg"] = angle
        row["speed_px_s"] = float(np.hypot(u, v))
        row["speed_units_s"] = row["speed_px_s"] * args.length_per_px if args.length_per_px else ""
        row["streamwise_units_s"] = streamwise * args.length_per_px if args.length_per_px else ""
        filtered.append(row)

    return filtered, f"{correction}; kept {len(filtered)}/{len(samples)} samples"


def write_speed_outputs(out_dir: Path, samples: list[dict], summary: dict[str, float | int | str], overlay: np.ndarray | None) -> None:
    summary_fields = [
        "samples",
        "frame_gap",
        "dt_sec",
        "median_speed_px_s",
        "mean_speed_px_s",
        "median_streamwise_px_s",
        "mean_streamwise_px_s",
        "p10_speed_px_s",
        "p90_speed_px_s",
        "flow_direction",
        "direction_correction",
        "length_per_px",
        "speed_unit",
        "median_speed_units_s",
        "mean_speed_units_s",
        "median_streamwise_units_s",
        "mean_streamwise_units_s",
        "note",
    ]
    with (out_dir / "speed_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow({field: summary.get(field, "") for field in summary_fields})

    sample_fields = [
        "frame_index",
        "time_sec",
        "x_px",
        "y_px",
        "raw_u_px_s",
        "raw_v_px_s",
        "u_px_s",
        "v_px_s",
        "streamwise_px_s",
        "crossflow_px_s",
        "flow_angle_deg",
        "speed_px_s",
        "speed_units_s",
        "streamwise_units_s",
    ]
    with (out_dir / "speed_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fields)
        writer.writeheader()
        for row in samples:
            writer.writerow(
                {
                    field: f"{row[field]:.6g}" if isinstance(row.get(field), float) else row.get(field, "")
                    for field in sample_fields
                }
            )
    if overlay is not None:
        cv2.imwrite(str(out_dir / "speed_overlay.png"), overlay)


def draw_speed_vectors(frame: np.ndarray, rows: list[dict], args: argparse.Namespace, label: str) -> np.ndarray:
    overlay = frame.copy()
    scale = args.speed_vector_scale
    for row in rows[: args.speed_overlay_vectors]:
        x = float(row["x_px"])
        y = float(row["y_px"])
        u = float(row["u_px_s"])
        v = float(row["v_px_s"])
        arrow_len = float(np.hypot(u * scale, v * scale))
        if args.speed_overlay_max_arrow_px > 0 and arrow_len > args.speed_overlay_max_arrow_px:
            shrink = args.speed_overlay_max_arrow_px / arrow_len
            u *= shrink
            v *= shrink
        end = (int(round(x + u * scale)), int(round(y + v * scale)))
        cv2.arrowedLine(overlay, (int(x), int(y)), end, (255, 255, 255), 1, tipLength=0.3)
    cv2.putText(overlay, label, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def write_speed_video(video_path: Path, out_path: Path, samples: list[dict], args: argparse.Namespace, fps: float) -> None:
    if not samples:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = args.speed_video_fps or max(fps / max(args.speed_step, 1), 1.0)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in samples:
        by_frame[int(row["frame_index"])].append(row)

    for frame_index in sorted(by_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        rows = by_frame[frame_index]
        streamwise = np.array([float(row["streamwise_px_s"]) for row in rows], dtype=np.float32)
        label = f"frame={frame_index} median streamwise={float(np.median(streamwise)):.1f} px/s"
        writer.write(draw_speed_vectors(frame, rows, args, label))
    writer.release()
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze wind-tunnel streamline video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("video_output"))
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--max-frames", type=int, help="Process at most this many frames/pairs; useful for smoke tests")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Show progress bars")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--roi", type=parse_roi, help="Optional crop as x,y,w,h")
    parser.add_argument("--detector", choices=["hybrid", "color", "ridge", "line", "ridge_line"], default="ridge")
    parser.add_argument("--green-score", type=float, default=35.0)
    parser.add_argument("--ridge-threshold", type=int, default=5)
    parser.add_argument("--ridge-height", type=int, default=31)
    parser.add_argument("--line-threshold", type=float, default=3.5)
    parser.add_argument("--line-length", type=int, default=31)
    parser.add_argument("--line-angles", type=parse_angles, default=parse_angles("-35,-25,-15,-8,0,8,15,25,35"))
    parser.add_argument("--min-value", type=int, default=20)
    parser.add_argument("--min-saturation", type=int, default=15)
    parser.add_argument("--blur", type=int, default=3)
    parser.add_argument("--close-width", type=int, default=9)
    parser.add_argument("--min-area", type=int, default=12)
    parser.add_argument("--max-area", type=int, default=0)
    parser.add_argument("--max-fill-ratio", type=float, default=1.0)
    parser.add_argument("--min-points", type=int, default=12)
    parser.add_argument("--min-length", type=float, default=12.0)
    parser.add_argument("--min-horizontal-span", type=int, default=8)
    parser.add_argument("--horizontal-ratio", type=float, default=0.0)
    parser.add_argument("--stitch-gap", type=float, default=160.0)
    parser.add_argument("--stitch-y-tolerance", type=float, default=7.0)
    parser.add_argument("--stitch-overlap", type=float, default=20.0)
    parser.add_argument("--stitch-iterations", type=int, default=4)
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--track-y-tolerance", type=float, default=5.0)
    parser.add_argument("--mean-x-step", type=int, default=5)
    parser.add_argument("--min-track-frames", type=int, default=2)
    parser.add_argument("--wake-x", type=float, help="Pixel x coordinate for wake width measurement")
    parser.add_argument("--wake-x-tolerance", type=float, default=12.0)
    parser.add_argument("--wake-y-range", type=parse_roi_y_range, help="Optional y_min,y_max range for wake width measurement")
    parser.add_argument("--wake-percentile-low", type=float, default=5.0)
    parser.add_argument("--wake-percentile-high", type=float, default=95.0)
    parser.add_argument("--deflection-ref-fraction", type=float, default=0.15)
    parser.add_argument("--speed-analysis", action="store_true", help="Estimate apparent velocity using dense optical flow")
    parser.add_argument("--speed-frame-gap", type=int, default=1, help="Frame separation for optical-flow velocity")
    parser.add_argument("--speed-step", type=int, default=1, help="Frame stride between optical-flow samples")
    parser.add_argument("--speed-grid", type=int, default=48, help="Grid cell size for speed samples")
    parser.add_argument("--speed-window", type=int, default=25, help="Farneback optical-flow window size")
    parser.add_argument("--speed-mask-fraction", type=float, default=0.05, help="Minimum green-pixel fraction per speed cell")
    parser.add_argument("--speed-min-green-pixels", type=int, default=0, help="Skip frame pairs with fewer streamline pixels than this")
    parser.add_argument("--speed-y-range", type=parse_roi_y_range, help="Optional y_min,y_max inside ROI for speed sampling")
    parser.add_argument("--speed-flow-direction", choices=["left", "right", "auto", "raw"], default="auto", help="Expected physical free-stream direction")
    parser.add_argument("--speed-filter-wrong-direction", action=argparse.BooleanOptionalAction, default=True, help="Drop vectors opposite to expected flow direction")
    parser.add_argument("--speed-max-angle-deg", type=float, default=75.0, help="Drop vectors with cross-flow angle larger than this; set 180 to disable")
    parser.add_argument("--length-per-px", type=float, default=0.0, help="Physical length per pixel for speed conversion")
    parser.add_argument("--speed-unit", default="length/s", help="Label for converted speed, e.g. m/s or mm/s")
    parser.add_argument("--speed-vector-scale", type=float, default=2.0, help="Display scale for velocity arrows")
    parser.add_argument("--speed-overlay-max-arrow-px", type=float, default=45.0, help="Cap displayed arrow length without changing CSV values")
    parser.add_argument("--speed-overlay-vectors", type=int, default=250, help="Maximum arrows drawn on speed overlay")
    parser.add_argument("--speed-video", type=Path, help="Write per-speed-sample-frame velocity arrow video")
    parser.add_argument("--speed-video-fps", type=float, help="FPS for speed video; defaults to input fps / speed_step")
    parser.add_argument("--overlay-video", type=Path, help="Write per-processed-frame streamline overlay video")
    parser.add_argument("--overlay-video-fps", type=float, help="FPS for overlay video; defaults to input fps / frame_step")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    start_frame = max(0, int(args.start_sec * fps))
    end_frame = total_frames if args.end_sec is None else min(total_frames, int(args.end_sec * fps))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detections = []
    representative_frame = None
    processed_frames = 0
    overlay_writer = None
    if args.overlay_video:
        args.overlay_video.parent.mkdir(parents=True, exist_ok=True)
        out_fps = args.overlay_video_fps or max(fps / max(args.frame_step, 1), 1.0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        overlay_writer = cv2.VideoWriter(str(args.overlay_video), fourcc, out_fps, (frame_width, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if not overlay_writer.isOpened():
            overlay_writer = None

    frame_indices = list(range(start_frame, end_frame, max(args.frame_step, 1)))
    if args.max_frames:
        frame_indices = frame_indices[: args.max_frames]

    for loop_i, frame_index in enumerate(frame_indices, start=1):
        if args.max_frames and processed_frames >= args.max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        if representative_frame is None:
            representative_frame = frame.copy()
        lines = process_frame(frame, args)
        processed_frames += 1
        if overlay_writer is not None:
            overlay_writer.write(draw_frame_overlay(frame, lines, frame_index, frame_index / fps))
        for line_id, line in enumerate(lines, start=1):
            if len(line) < 2:
                continue
            detections.append(
                {
                    "frame_index": frame_index,
                    "time_sec": frame_index / fps,
                    "line_id_in_frame": line_id,
                    "line": line,
                    "summary": line_summary(line),
                }
            )
        if args.progress:
            print_progress("Streamline frames", loop_i, len(frame_indices))

    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()
    if representative_frame is None:
        raise SystemExit("No frames were processed.")

    detections = assign_tracks(detections, args.track_y_tolerance)
    mean_rows, track_summaries = build_mean_streamlines(detections, args.mean_x_step, args.min_track_frames)
    deflection_curves, deflection_summaries = compute_deflection(mean_rows, args.deflection_ref_fraction)
    wake = compute_wake_width(
        mean_rows,
        args.wake_x,
        frame_width,
        args.roi,
        args.wake_x_tolerance,
        args.wake_y_range,
        args.wake_percentile_low,
        args.wake_percentile_high,
    )

    write_frame_streamlines(args.out_dir / "frame_streamlines.csv", detections)
    write_mean_streamlines(args.out_dir / "mean_streamlines.csv", mean_rows)
    write_mean_json(args.out_dir / "mean_streamlines.json", mean_rows)
    write_wake_summary(args.out_dir / "wake_width_summary.csv", wake)
    write_deflection(
        args.out_dir / "deflection_summary.csv",
        args.out_dir / "deflection_curves.csv",
        deflection_curves,
        deflection_summaries,
    )
    draw_mean_overlay(representative_frame, mean_rows, args.out_dir / "mean_overlay.png")
    draw_wake_overlay(representative_frame, mean_rows, wake, args.out_dir / "wake_width_overlay.png")
    draw_deflection_plot(deflection_curves, args.out_dir / "deflection_plot.png")
    speed_summary = {"note": "Speed analysis not requested."}
    if args.speed_analysis:
        speed_samples, speed_summary, speed_overlay = compute_speed_analysis(args)
        write_speed_outputs(args.out_dir, speed_samples, speed_summary, speed_overlay)
        if args.speed_video:
            write_speed_video(args.video, args.speed_video, speed_samples, args, fps)

    metadata = {
        "video": str(args.video),
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "max_frames": args.max_frames or "",
        "overlay_video": str(args.overlay_video) if args.overlay_video else "",
        "speed_video": str(args.speed_video) if args.speed_video else "",
        "detections": len(detections),
        "tracks": len(track_summaries),
        "wake": wake,
        "speed": speed_summary,
    }
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Processed {processed_frames} frames; detected {len(detections)} frame-level lines.")
    print(f"Wrote {len(track_summaries)} mean streamline tracks to {args.out_dir / 'mean_streamlines.csv'}")
    print(f"Wrote wake summary to {args.out_dir / 'wake_width_summary.csv'}")
    if args.speed_analysis:
        print(f"Wrote speed summary to {args.out_dir / 'speed_summary.csv'}")


if __name__ == "__main__":
    main()
