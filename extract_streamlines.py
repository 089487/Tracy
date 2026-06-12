#!/usr/bin/env python3
"""
Extract wind-tunnel laser/smoke streamline centerline coordinates from an image.

Example:
    python3 extract_streamlines.py input.jpg --out-dir output

The script writes:
    output/streamlines.csv      one row per sampled point
    output/streamlines.json     grouped polyline coordinates
    output/mask.png             detected green-line mask
    output/skeleton.png         one-pixel centerline image
    output/overlay.png          detected lines drawn on the source image
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


NEIGHBORS_8 = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]


def parse_roi(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [int(v.strip()) for v in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must be x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("--roi width and height must be positive")
    return x, y, w, h


def parse_angles(text: str) -> list[float]:
    try:
        return [float(v.strip()) for v in text.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--line-angles must be comma-separated numbers") from exc


def make_green_mask(
    image_bgr: np.ndarray,
    green_score_threshold: float,
    min_value: int,
    saturation_threshold: int,
    blur: int,
    close_width: int,
) -> np.ndarray:
    if blur > 1:
        if blur % 2 == 0:
            blur += 1
        image_bgr = cv2.GaussianBlur(image_bgr, (blur, blur), 0)

    b, g, r = cv2.split(image_bgr.astype(np.float32))
    green_score = g - 0.5 * (r + b)
    score_mask = green_score >= green_score_threshold

    hsv = cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsv_mask = (h >= 35) & (h <= 95) & (s >= saturation_threshold) & (v >= min_value)

    mask = (score_mask & hsv_mask).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if close_width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def make_ridge_mask(
    image_bgr: np.ndarray,
    ridge_threshold: int,
    ridge_height: int,
    min_value: int,
    saturation_threshold: int,
    close_width: int,
) -> np.ndarray:
    """Detect thin bright horizontal ridges in the green channel."""
    if ridge_height % 2 == 0:
        ridge_height += 1
    ridge_height = max(ridge_height, 3)

    green = image_bgr[:, :, 1]
    hsv = cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # A vertical opening estimates the local green background while suppressing
    # thin horizontal bright lines. Subtracting it highlights streamlines even
    # when the whole right side of the image is already green.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ridge_height))
    background = cv2.morphologyEx(green, cv2.MORPH_OPEN, vertical_kernel)
    ridge = cv2.subtract(green, background)

    greenish = (h >= 35) & (h <= 95) & (s >= saturation_threshold) & (v >= min_value)
    mask = ((ridge >= ridge_threshold) & greenish).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if close_width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def make_line_kernel(length: int, angle_deg: float) -> np.ndarray:
    size = length if length % 2 == 1 else length + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    theta = np.deg2rad(angle_deg)
    dx = np.cos(theta)
    dy = np.sin(theta)
    for t in np.linspace(-(length - 1) / 2, (length - 1) / 2, length):
        x = int(round(center + t * dx))
        y = int(round(center + t * dy))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    kernel_sum = float(kernel.sum())
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def make_multiorientation_line_mask(
    image_bgr: np.ndarray,
    line_threshold: float,
    line_length: int,
    line_angles: list[float],
    min_value: int,
    saturation_threshold: int,
    close_width: int,
) -> np.ndarray:
    green = image_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    enhanced_f = enhanced.astype(np.float32)
    background = cv2.GaussianBlur(enhanced_f, (0, 0), sigmaX=max(line_length / 2, 3))
    local = enhanced_f - background

    response = np.zeros_like(local, dtype=np.float32)
    for angle in line_angles:
        kernel = make_line_kernel(line_length, angle)
        filtered = cv2.filter2D(local, cv2.CV_32F, kernel)
        response = np.maximum(response, filtered)

    hsv = cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    greenish = (h >= 35) & (h <= 95) & (s >= saturation_threshold) & (v >= min_value)
    mask = ((response >= line_threshold) & greenish).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if close_width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_width, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def compute_ridge_response(
    image_bgr: np.ndarray,
    ridge_height: int,
    min_value: int,
    saturation_threshold: int,
    smooth_x: float,
    smooth_y: float,
    contrast_norm: bool = True,
    min_headroom: float = 48.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a float ridge-strength map and a boolean greenish-validity map.

    With contrast_norm enabled, the response is divided by the local intensity
    headroom (255 - background). On a bright green background the lines can
    only be slightly brighter than their surroundings, so the same physical
    streamline produces a much weaker raw ridge response than on a dark
    background; normalizing by headroom makes one threshold work across both.
    """
    if ridge_height % 2 == 0:
        ridge_height += 1
    ridge_height = max(ridge_height, 3)

    green = image_bgr[:, :, 1]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ridge_height))
    background = cv2.morphologyEx(green, cv2.MORPH_OPEN, vertical_kernel)
    response = cv2.subtract(green, background).astype(np.float32)
    if contrast_norm:
        background_blur = cv2.GaussianBlur(background.astype(np.float32), (0, 0), 15)
        headroom = np.clip(255.0 - background_blur, min_headroom, None)
        response = response * (255.0 / headroom)
    if smooth_x > 0 or smooth_y > 0:
        response = cv2.GaussianBlur(
            response,
            (0, 0),
            sigmaX=max(smooth_x, 0.01),
            sigmaY=max(smooth_y, 0.01),
        )

    hsv = cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    valid = (h >= 35) & (h <= 95) & (s >= saturation_threshold) & (v >= min_value)
    return response, valid


def find_subpixel_column_peaks(
    column: np.ndarray,
    valid_column: np.ndarray,
    threshold: float,
    min_spacing: int,
) -> list[float]:
    """Find subpixel y positions of local maxima in one image column."""
    r = np.where(valid_column, column, 0.0)
    interior = np.zeros_like(r, dtype=bool)
    interior[1:-1] = (r[1:-1] >= threshold) & (r[1:-1] >= r[:-2]) & (r[1:-1] > r[2:])
    candidates = np.where(interior)[0]
    if candidates.size == 0:
        return []

    order = candidates[np.argsort(-r[candidates])]
    taken = np.zeros(len(r), dtype=bool)
    selected = []
    for idx in order:
        lo = max(0, idx - min_spacing)
        if taken[lo : idx + min_spacing + 1].any():
            continue
        taken[idx] = True
        selected.append(int(idx))

    peaks = []
    for y in selected:
        denom = r[y - 1] - 2.0 * r[y] + r[y + 1]
        delta = 0.0 if abs(denom) < 1e-6 else 0.5 * (r[y - 1] - r[y + 1]) / denom
        delta = float(np.clip(delta, -0.5, 0.5))
        peaks.append(y + delta)
    peaks.sort()
    return peaks


def trace_ridge_tracks(
    response: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    min_spacing: int,
    max_gap: int,
    y_tolerance: float,
    gap_tolerance_growth: float = 0.12,
    slope_alpha: float = 0.3,
    slope_max: float = 2.5,
) -> list[list[tuple[float, float]]]:
    """Link per-column subpixel peaks into streamline tracks, left to right.

    Each active track predicts its next y from a smoothed local slope, so the
    tracker follows curved lines and does not jump to a neighboring line even
    where lines are tightly packed.
    """
    height, width = response.shape
    active: list[dict] = []
    finished: list[list[tuple[float, float]]] = []

    for x in range(width):
        peaks = find_subpixel_column_peaks(response[:, x], valid[:, x], threshold, min_spacing)

        candidates = []
        for ti, track in enumerate(active):
            gap = x - track["last_x"]
            predicted = track["last_y"] + track["slope"] * gap
            tolerance = y_tolerance + gap_tolerance_growth * gap
            for pi, peak_y in enumerate(peaks):
                distance = abs(peak_y - predicted)
                if distance <= tolerance:
                    candidates.append((distance, ti, pi))
        candidates.sort(key=lambda c: c[0])

        used_tracks: set[int] = set()
        used_peaks: set[int] = set()
        for distance, ti, pi in candidates:
            if ti in used_tracks or pi in used_peaks:
                continue
            used_tracks.add(ti)
            used_peaks.add(pi)
            track = active[ti]
            peak_y = peaks[pi]
            gap = x - track["last_x"]
            instant_slope = float(np.clip((peak_y - track["last_y"]) / gap, -slope_max, slope_max))
            track["slope"] = (1.0 - slope_alpha) * track["slope"] + slope_alpha * instant_slope
            track["points"].append((float(x), peak_y))
            track["last_x"] = x
            track["last_y"] = peak_y

        still_active = []
        for track in active:
            if x - track["last_x"] > max_gap:
                finished.append(track["points"])
            else:
                still_active.append(track)
        active = still_active

        for pi, peak_y in enumerate(peaks):
            if pi in used_peaks:
                continue
            active.append({"points": [(float(x), peak_y)], "last_x": x, "last_y": peak_y, "slope": 0.0})

    finished.extend(track["points"] for track in active)
    return finished


def rasterize_polylines(shape: tuple[int, int], lines: list[list[tuple[float, float]]]) -> np.ndarray:
    img = np.zeros(shape, dtype=np.uint8)
    for line in lines:
        pts = np.array([(int(round(x)), int(round(y))) for x, y in line], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], isClosed=False, color=255, thickness=1)
    return img


def make_streamline_mask(
    image_bgr: np.ndarray,
    detector: str,
    green_score_threshold: float,
    ridge_threshold: int,
    ridge_height: int,
    line_threshold: float,
    line_length: int,
    line_angles: list[float],
    min_value: int,
    saturation_threshold: int,
    blur: int,
    close_width: int,
) -> np.ndarray:
    color_mask = None
    ridge_mask = None
    line_mask = None
    if detector in {"color", "hybrid"}:
        color_mask = make_green_mask(
            image_bgr,
            green_score_threshold=green_score_threshold,
            min_value=min_value,
            saturation_threshold=saturation_threshold,
            blur=blur,
            close_width=close_width,
        )
    if detector in {"ridge", "hybrid", "ridge_line"}:
        ridge_mask = make_ridge_mask(
            image_bgr,
            ridge_threshold=ridge_threshold,
            ridge_height=ridge_height,
            min_value=min_value,
            saturation_threshold=saturation_threshold,
            close_width=close_width,
        )
    if detector in {"line", "hybrid", "ridge_line"}:
        line_mask = make_multiorientation_line_mask(
            image_bgr,
            line_threshold=line_threshold,
            line_length=line_length,
            line_angles=line_angles,
            min_value=min_value,
            saturation_threshold=saturation_threshold,
            close_width=close_width,
        )

    masks = [mask for mask in (color_mask, ridge_mask, line_mask) if mask is not None]
    if not masks:
        raise ValueError(f"Unsupported detector: {detector}")
    combined = masks[0]
    for mask in masks[1:]:
        combined = cv2.bitwise_or(combined, mask)
    return combined


def remove_small_components(
    mask: np.ndarray,
    min_area: int,
    max_area: int = 0,
    max_fill_ratio: float = 1.0,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        fill_ratio = area / max(width * height, 1)
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        if max_fill_ratio < 1.0 and fill_ratio > max_fill_ratio:
            continue
        cleaned[labels == label] = 255
    return cleaned


def zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel skeleton using Zhang-Suen thinning."""
    img = (mask > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]

            neighbor_count = sum(neighbors)
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            if step == 0:
                side_condition = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                side_condition = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            delete = (
                (img == 1)
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & side_condition
            )
            if np.any(delete):
                img[delete] = 0
                changed = True
    return (img * 255).astype(np.uint8)


def skeleton_neighbors(point: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    x, y = point
    return [(x + dx, y + dy) for dx, dy in NEIGHBORS_8 if (x + dx, y + dy) in pixels]


def trace_skeleton_component(component_mask: np.ndarray, x_offset: int, y_offset: int) -> list[list[tuple[int, int]]]:
    ys, xs = np.where(component_mask > 0)
    pixels = {(int(x + x_offset), int(y + y_offset)) for x, y in zip(xs, ys)}
    if not pixels:
        return []

    degree = {p: len(skeleton_neighbors(p, pixels)) for p in pixels}
    nodes = {p for p, deg in degree.items() if deg != 2}
    visited_edges: set[frozenset[tuple[int, int]]] = set()
    polylines: list[list[tuple[int, int]]] = []

    def walk(start: tuple[int, int], first: tuple[int, int]) -> list[tuple[int, int]]:
        line = [start, first]
        prev, current = start, first
        visited_edges.add(frozenset((prev, current)))
        while current not in nodes:
            choices = [p for p in skeleton_neighbors(current, pixels) if p != prev]
            if not choices:
                break
            nxt = choices[0]
            edge = frozenset((current, nxt))
            if edge in visited_edges:
                break
            line.append(nxt)
            visited_edges.add(edge)
            prev, current = current, nxt
        return line

    starts = nodes if nodes else {next(iter(pixels))}
    for start in starts:
        for nxt in skeleton_neighbors(start, pixels):
            edge = frozenset((start, nxt))
            if edge not in visited_edges:
                polylines.append(walk(start, nxt))

    if not nodes:
        remaining = [
            (p, n)
            for p in pixels
            for n in skeleton_neighbors(p, pixels)
            if frozenset((p, n)) not in visited_edges
        ]
        if remaining:
            polylines.append(walk(*remaining[0]))

    return polylines


def extract_polylines(skeleton: np.ndarray, min_points: int) -> list[list[tuple[int, int]]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    all_lines: list[list[tuple[int, int]]] = []
    for label in range(1, count):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        component = (labels[y : y + h, x : x + w] == label).astype(np.uint8)
        for line in trace_skeleton_component(component, x, y):
            if len(line) >= min_points:
                all_lines.append(line)
    return all_lines


def line_length(line: list[tuple[int, int]]) -> float:
    return sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(line, line[1:]))


def filter_polylines(
    lines: list[list[tuple[int, int]]],
    min_length: float,
    min_horizontal_span: int,
    horizontal_ratio: float,
) -> list[list[tuple[int, int]]]:
    kept = []
    for line in lines:
        xs = [p[0] for p in line]
        ys = [p[1] for p in line]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        if line_length(line) < min_length:
            continue
        if dx < min_horizontal_span:
            continue
        if horizontal_ratio > 0 and dx / max(dy, 1) < horizontal_ratio:
            continue
        kept.append(line)
    kept.sort(key=lambda pts: (np.median([p[1] for p in pts]), min(p[0] for p in pts)))
    return kept


def normalize_left_to_right(line: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not line:
        return line
    if line[0][0] <= line[-1][0]:
        return line
    return list(reversed(line))


def merge_line_points(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> list[tuple[int, int]]:
    points = normalize_left_to_right(left) + normalize_left_to_right(right)
    seen = set()
    deduped = []
    for point in sorted(points, key=lambda p: (p[0], p[1])):
        if point not in seen:
            deduped.append(point)
            seen.add(point)
    return deduped


def stitch_polylines(
    lines: list[list[tuple[int, int]]],
    max_gap: float,
    y_tolerance: float,
    overlap_tolerance: float,
    iterations: int,
) -> list[list[tuple[int, int]]]:
    if max_gap <= 0 or y_tolerance < 0:
        return [normalize_left_to_right(line) for line in lines]

    stitched = [normalize_left_to_right(line) for line in lines]
    for _ in range(max(iterations, 1)):
        used = [False] * len(stitched)
        merged: list[list[tuple[int, int]]] = []
        changed = False

        order = sorted(
            range(len(stitched)),
            key=lambda i: (np.median([p[1] for p in stitched[i]]), min(p[0] for p in stitched[i])),
        )

        for i in order:
            if used[i]:
                continue
            current = stitched[i]
            used[i] = True
            grew = True
            while grew:
                grew = False
                current = normalize_left_to_right(current)
                current_end = current[-1]
                current_y = np.median([p[1] for p in current[-min(len(current), 9) :]])

                best_j = None
                best_score = None
                for j in order:
                    if used[j]:
                        continue
                    candidate = normalize_left_to_right(stitched[j])
                    candidate_start = candidate[0]
                    candidate_y = np.median([p[1] for p in candidate[: min(len(candidate), 9)]])

                    x_gap = candidate_start[0] - current_end[0]
                    y_gap = abs(candidate_y - current_y)
                    if x_gap < -overlap_tolerance or x_gap > max_gap:
                        continue
                    if y_gap > y_tolerance:
                        continue

                    score = y_gap * 10 + abs(x_gap)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_j = j

                if best_j is not None:
                    current = merge_line_points(current, stitched[best_j])
                    used[best_j] = True
                    grew = True
                    changed = True
            merged.append(current)

        stitched = merged
        if not changed:
            break

    stitched.sort(key=lambda pts: (np.median([p[1] for p in pts]), min(p[0] for p in pts)))
    return stitched


def smooth_and_downsample(line: list[tuple[int, int]], every: int, smooth_window: int) -> list[tuple[float, float]]:
    arr = np.array(line, dtype=np.float32)
    if smooth_window > 1 and len(arr) >= smooth_window:
        if smooth_window % 2 == 0:
            smooth_window += 1
        pad = smooth_window // 2
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        arr[:, 0] = np.convolve(padded[:, 0], kernel, mode="valid")
        arr[:, 1] = np.convolve(padded[:, 1], kernel, mode="valid")
    return [(float(x), float(y)) for x, y in arr[:: max(every, 1)]]


def report_polyline_points(
    line: list[tuple[float, float]],
    x_offset: int,
    y_offset: int,
    spacing: float = 3.0,
    smooth_window: int = 25,
) -> np.ndarray:
    points = np.array([(x + x_offset, y + y_offset) for x, y in normalize_left_to_right(line)], dtype=np.float32)
    if len(points) < 2:
        return points.astype(np.int32)

    order = np.argsort(points[:, 0])
    points = points[order]
    unique_x = []
    mean_y = []
    for x in np.unique(points[:, 0]):
        ys = points[points[:, 0] == x, 1]
        unique_x.append(float(x))
        mean_y.append(float(np.mean(ys)))

    xs = np.array(unique_x, dtype=np.float32)
    ys = np.array(mean_y, dtype=np.float32)
    if len(xs) < 2 or xs[-1] - xs[0] < spacing:
        return points.astype(np.int32)

    grid = np.arange(xs[0], xs[-1] + spacing, spacing, dtype=np.float32)
    sampled_y = np.interp(grid, xs, ys).astype(np.float32)
    if smooth_window > 1 and len(sampled_y) >= 5:
        smooth_window = min(smooth_window, len(sampled_y) if len(sampled_y) % 2 == 1 else len(sampled_y) - 1)
        if smooth_window >= 3:
            pad = smooth_window // 2
            padded = np.pad(sampled_y, pad, mode="edge")
            kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
            sampled_y = np.convolve(padded, kernel, mode="valid")

    return np.column_stack([grid, sampled_y]).round().astype(np.int32)


def write_outputs(
    out_dir: Path,
    source: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
    lines: list[list[tuple[float, float]]],
    roi_offset: tuple[int, int],
    scales: tuple[float, float],
    origins: tuple[float, float],
    report_spacing: float,
    report_smooth_window: int,
    report_thickness: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "mask.png"), mask)
    cv2.imwrite(str(out_dir / "skeleton.png"), skeleton)

    x0, y0 = roi_offset
    sx, sy = scales
    ox, oy = origins

    with (out_dir / "streamlines.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["line_id", "point_index", "x_px", "y_px", "x", "y"],
        )
        writer.writeheader()
        for line_id, line in enumerate(lines, start=1):
            for point_index, (x, y) in enumerate(line):
                x_px = x + x0
                y_px = y + y0
                writer.writerow(
                    {
                        "line_id": line_id,
                        "point_index": point_index,
                        "x_px": round(x_px, 3),
                        "y_px": round(y_px, 3),
                        "x": round(ox + x_px * sx, 6),
                        "y": round(oy + y_px * sy, 6),
                    }
                )

    payload = []
    for line_id, line in enumerate(lines, start=1):
        points = []
        for x, y in line:
            x_px = x + x0
            y_px = y + y0
            points.append(
                {
                    "x_px": round(x_px, 3),
                    "y_px": round(y_px, 3),
                    "x": round(ox + x_px * sx, 6),
                    "y": round(oy + y_px * sy, 6),
                }
            )
        payload.append({"line_id": line_id, "points": points})
    (out_dir / "streamlines.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    overlay = source.copy()
    palette = [
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
    ]
    for line_id, line in enumerate(lines):
        pts = np.array([(int(round(x + x0)), int(round(y + y0))) for x, y in line], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(overlay, [pts], isClosed=False, color=palette[line_id % len(palette)], thickness=2)
            cv2.putText(
                overlay,
                str(line_id + 1),
                tuple(pts[len(pts) // 2]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)

    report = source.copy()
    dim = np.full_like(report, 20)
    report = cv2.addWeighted(report, 0.65, dim, 0.35, 0)
    for line in lines:
        pts = report_polyline_points(line, x0, y0, spacing=report_spacing, smooth_window=report_smooth_window)
        if len(pts) >= 2:
            cv2.polylines(report, [pts], isClosed=False, color=(0, 0, 0), thickness=report_thickness, lineType=cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "streamline_map.png"), report)


def detect_streamline_polylines(
    work: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[list[tuple[float, float]]], np.ndarray, np.ndarray]:
    """Run the full detection pipeline on one BGR image (already ROI-cropped).

    Shared by the single-image CLI and the video pipeline so both use the same
    tracer. Returns (lines, mask, skeleton); lines are filtered and stitched
    but not yet smoothed/downsampled.
    """
    if getattr(args, "tracer", "peak") == "peak":
        response, valid = compute_ridge_response(
            work,
            ridge_height=args.ridge_height,
            min_value=args.min_value,
            saturation_threshold=args.min_saturation,
            smooth_x=args.ridge_smooth_x,
            smooth_y=args.ridge_smooth_y,
            contrast_norm=args.contrast_norm,
            min_headroom=args.min_headroom,
        )
        mask = ((response >= args.ridge_threshold) & valid).astype(np.uint8) * 255
        raw_lines = trace_ridge_tracks(
            response,
            valid,
            threshold=args.ridge_threshold,
            min_spacing=args.peak_min_spacing,
            max_gap=args.peak_max_gap,
            y_tolerance=args.peak_y_tolerance,
        )
        raw_lines = [line for line in raw_lines if len(line) >= args.min_points]
        skeleton = rasterize_polylines(work.shape[:2], raw_lines)
    else:
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
    return lines, mask, skeleton


def add_detection_arguments(parser: argparse.ArgumentParser) -> None:
    """Register tracer-specific options shared by the image and video CLIs."""
    parser.add_argument(
        "--tracer",
        choices=["peak", "skeleton"],
        default="peak",
        help="peak tracks subpixel per-column ridge maxima (precise centerlines); skeleton is the legacy mask-thinning path",
    )
    parser.add_argument("--peak-min-spacing", type=int, default=3, help="Minimum vertical pixel spacing between peaks in one column")
    parser.add_argument("--peak-max-gap", type=int, default=40, help="Columns a track may go unmatched before it is closed")
    parser.add_argument("--peak-y-tolerance", type=float, default=2.0, help="Base vertical matching tolerance for peak tracking")
    parser.add_argument("--ridge-smooth-x", type=float, default=2.0, help="Gaussian sigma along x applied to the ridge response")
    parser.add_argument("--ridge-smooth-y", type=float, default=1.0, help="Gaussian sigma along y applied to the ridge response")
    parser.add_argument(
        "--contrast-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize ridge response by local brightness headroom so bright-background regions keep their lines",
    )
    parser.add_argument("--min-headroom", type=float, default=48.0, help="Lower clamp for 255-background during contrast normalization")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract green wind-tunnel streamline coordinates.")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--out-dir", type=Path, default=Path("streamline_output"))
    parser.add_argument("--roi", type=parse_roi, help="Optional crop as x,y,w,h in pixel coordinates")
    parser.add_argument(
        "--detector",
        choices=["hybrid", "color", "ridge", "line", "ridge_line"],
        default="ridge",
        help="hybrid combines color, horizontal ridge, and multi-angle line detection; ridge_line omits color blobs",
    )
    add_detection_arguments(parser)
    parser.add_argument("--green-score", type=float, default=35.0, help="Minimum G - 0.5*(R+B)")
    parser.add_argument("--ridge-threshold", type=int, default=5, help="Minimum local green-channel ridge contrast")
    parser.add_argument("--ridge-height", type=int, default=31, help="Vertical window used to estimate local background")
    parser.add_argument("--line-threshold", type=float, default=3.5, help="Minimum multi-angle local line response")
    parser.add_argument("--line-length", type=int, default=31, help="Matched-filter line kernel length")
    parser.add_argument(
        "--line-angles",
        type=parse_angles,
        default=parse_angles("-35,-25,-15,-8,0,8,15,25,35"),
        help="Comma-separated line angles in degrees for curved/oblique streamline fragments",
    )
    parser.add_argument("--min-value", type=int, default=20, help="Minimum HSV value/brightness")
    parser.add_argument("--min-saturation", type=int, default=15, help="Minimum HSV saturation")
    parser.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel size")
    parser.add_argument("--close-width", type=int, default=9, help="Horizontal closing width for broken lines")
    parser.add_argument("--min-area", type=int, default=12, help="Drop tiny mask components")
    parser.add_argument("--max-area", type=int, default=0, help="Drop mask components larger than this; 0 disables")
    parser.add_argument("--max-fill-ratio", type=float, default=1.0, help="Drop solid-looking components; 1 disables")
    parser.add_argument("--min-points", type=int, default=12, help="Drop skeleton fragments with too few pixels")
    parser.add_argument("--min-length", type=float, default=12.0, help="Drop polylines shorter than this many pixels")
    parser.add_argument("--min-horizontal-span", type=int, default=8, help="Drop lines with less horizontal span")
    parser.add_argument("--horizontal-ratio", type=float, default=0.0, help="Require dx/dy at least this; set 0 to disable")
    parser.add_argument("--stitch-gap", type=float, default=160.0, help="Connect line fragments with endpoint x gaps up to this many pixels")
    parser.add_argument("--stitch-y-tolerance", type=float, default=7.0, help="Connect fragments whose endpoint y values differ by at most this many pixels")
    parser.add_argument("--stitch-overlap", type=float, default=20.0, help="Allow this many pixels of x overlap when stitching fragments")
    parser.add_argument("--stitch-iterations", type=int, default=4, help="Maximum repeated stitching passes")
    parser.add_argument("--sample-every", type=int, default=3, help="Write every Nth point on each line")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving-average window before output")
    parser.add_argument("--x-scale", type=float, default=1.0, help="Real-world units per pixel in x")
    parser.add_argument("--y-scale", type=float, default=1.0, help="Real-world units per pixel in y")
    parser.add_argument("--x-origin", type=float, default=0.0, help="Real-world x at image x=0")
    parser.add_argument("--y-origin", type=float, default=0.0, help="Real-world y at image y=0")
    parser.add_argument("--report-spacing", type=float, default=3.0, help="Pixel spacing for smoothed report-view streamlines")
    parser.add_argument("--report-smooth-window", type=int, default=25, help="Smoothing window for report-view streamlines")
    parser.add_argument("--report-thickness", type=int, default=2, help="Line thickness for streamline_map.png")
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    roi_offset = (0, 0)
    work = image
    if args.roi:
        x, y, w, h = args.roi
        roi_offset = (x, y)
        work = image[y : y + h, x : x + w]

    lines, mask, skeleton = detect_streamline_polylines(work, args)
    sampled = [
        smooth_and_downsample(line, every=args.sample_every, smooth_window=args.smooth_window)
        for line in lines
    ]

    write_outputs(
        args.out_dir,
        source=image,
        mask=mask,
        skeleton=skeleton,
        lines=sampled,
        roi_offset=roi_offset,
        scales=(args.x_scale, args.y_scale),
        origins=(args.x_origin, args.y_origin),
        report_spacing=args.report_spacing,
        report_smooth_window=args.report_smooth_window,
        report_thickness=args.report_thickness,
    )

    total_points = sum(len(line) for line in sampled)
    print(f"Extracted {len(sampled)} streamlines and {total_points} sampled points.")
    print(f"Wrote: {args.out_dir / 'streamlines.csv'}")
    print(f"Check: {args.out_dir / 'overlay.png'}")
    print(f"Report view: {args.out_dir / 'streamline_map.png'}")


if __name__ == "__main__":
    main()
