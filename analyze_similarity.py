#!/usr/bin/env python3
"""Compute dimensionless large/small-model similarity metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_FIELDS = [
    "D",
    "d",
    "r_out",
    "r_in",
    "V",
    "rho",
    "mu",
    "W",
    "d_over_D",
    "r_out_over_D",
    "r_in_over_D",
    "Re",
    "W_over_D",
]


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return float(value)


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def percent_difference(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    baseline = (abs(a) + abs(b)) / 2
    if baseline == 0:
        return 0.0
    return abs(a - b) / baseline * 100


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8g}"


def load_measurements(path: Path) -> list[dict[str, float | str | None]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float | str | None] = {"model": row.get("model", "").strip()}
            for field in ["D", "d", "r_out", "r_in", "V", "rho", "mu", "W"]:
                parsed[field] = parse_float(row.get(field))
            rows.append(add_metrics(parsed))
    return rows


def add_metrics(row: dict[str, float | str | None]) -> dict[str, float | str | None]:
    d = row["d"]
    diameter = row["D"]
    r_out = row["r_out"]
    r_in = row["r_in"]
    velocity = row["V"]
    rho = row["rho"]
    mu = row["mu"]
    wake = row["W"]

    row["d_over_D"] = safe_div(d, diameter)
    row["r_out_over_D"] = safe_div(r_out, diameter)
    row["r_in_over_D"] = safe_div(r_in, diameter)
    if None not in (rho, velocity, diameter, mu) and mu != 0:
        row["Re"] = float(rho) * float(velocity) * float(diameter) / float(mu)
    else:
        row["Re"] = None
    row["W_over_D"] = safe_div(wake, diameter)
    return row


def build_summary_rows(rows: list[dict[str, float | str | None]]) -> list[dict[str, str]]:
    out = []
    by_model = {str(row["model"]).lower(): row for row in rows}
    large = by_model.get("large")
    small = by_model.get("small")

    for row in rows:
        output = {"row_type": "model", "model": str(row["model"])}
        for field in METRIC_FIELDS:
            output[field] = fmt(row.get(field)) if field != "model" else str(row.get(field, ""))
        output["large_small_percent_difference"] = ""
        out.append(output)

    if large and small:
        for field in ["d_over_D", "r_out_over_D", "r_in_over_D", "Re", "W_over_D"]:
            out.append(
                {
                    "row_type": "large_small_percent_difference",
                    "model": field,
                    **{metric: "" for metric in METRIC_FIELDS},
                    "large_small_percent_difference": fmt(
                        percent_difference(
                            large.get(field),  # type: ignore[arg-type]
                            small.get(field),  # type: ignore[arg-type]
                        )
                    ),
                }
            )
    return out


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path(".") else None
    fieldnames = ["row_type", "model", *METRIC_FIELDS, "large_small_percent_difference"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_plot(path: Path, rows: list[dict[str, float | str | None]]) -> None:
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError:
        print("Skipping plot because cv2/numpy are not installed.")
        return

    metrics = ["d_over_D", "r_out_over_D", "r_in_over_D", "Re", "W_over_D"]
    models = [str(row["model"]) for row in rows if row.get("model")]
    values = {
        str(row["model"]): [row.get(metric) if isinstance(row.get(metric), float) else None for metric in metrics]
        for row in rows
        if row.get("model")
    }

    width, height = 1200, 700
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Dimensionless Similarity Comparison", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)

    x0, y0 = 90, 610
    plot_w, plot_h = 1020, 460
    cv2.rectangle(img, (x0, y0 - plot_h), (x0 + plot_w, y0), (0, 0, 0), 1)

    for i, metric in enumerate(metrics):
        metric_values = [values[model][i] for model in models]
        present = [v for v in metric_values if v is not None]
        if not present:
            continue
        scale = max(present)
        if scale == 0:
            scale = 1
        group_x = x0 + 70 + i * 190
        for j, model in enumerate(models[:2]):
            value = values[model][i]
            if value is None:
                continue
            bar_h = int((value / scale) * 160)
            color = (50, 120, 240) if model.lower() == "large" else (60, 180, 80)
            x = group_x + j * 42
            cv2.rectangle(img, (x, y0 - bar_h), (x + 32, y0), color, -1)
            cv2.putText(img, model[:5], (x - 8, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            cv2.putText(img, f"{value:.3g}", (x - 18, y0 - bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(img, metric, (group_x - 35, y0 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dimensionless wind-tunnel similarity metrics.")
    parser.add_argument("measurements", type=Path, help="CSV with model,D,d,r_out,r_in,V,rho,mu,W")
    parser.add_argument("--out", type=Path, default=Path("similarity_summary.csv"))
    parser.add_argument("--plot", type=Path, default=Path("video_output/dimensionless_comparison.png"))
    args = parser.parse_args()

    rows = load_measurements(args.measurements)
    summary_rows = build_summary_rows(rows)
    write_summary(args.out, summary_rows)
    write_comparison_plot(args.plot, rows)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.plot} if cv2/numpy were available")


if __name__ == "__main__":
    main()
