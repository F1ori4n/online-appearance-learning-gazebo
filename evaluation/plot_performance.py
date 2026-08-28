#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot performance time series for one run")
    parser.add_argument("run", type=Path, help="Run directory")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    # Resolve Input path
    run_dir = args.run.expanduser().resolve()
    csv_path = run_dir / "performance.csv"

    # Resolve Output dir
    output_dir = (args.output_dir.expanduser().resolve()
                  if args.output_dir is not None else run_dir / "performance_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise FileNotFoundError(f"No performance data found for run: {run_dir}")

    times = [float(row["wall_elapsed_sec"]) for row in rows]
    generated = []

    # Plot FPS over time
    def plot_metric(key: str, ylabel: str, title: str, filename: str) -> None:
        if key not in rows[0]:
            return
        values = [float(row[key]) for row in rows]
        plt.figure(figsize=(10, 5))
        plt.plot(times, values, label=key)
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.title(f"{title}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=170)
        plt.close()
        generated.append(filename)

    plot_metric(
        "rolling_fps",
        "Frames/s",
        "Perception Throughput",
        "fps_over_time.png",
    )

    # Plot Pipeline Latency
    plt.figure(figsize=(10, 5))
    plt.plot(times, [float(r["tracking_ms"]) for r in rows], label="YOLO + Tracking")
    plt.plot(times, [float(r["target_update_ms"]) for r in rows], label="Target Update (incl. ReID)")
    plt.xlabel("Time [s]")
    plt.ylabel("Latency [ms]")
    plt.title("Perception Pipeline Latency")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pipeline_latency_over_time.png", dpi=170)
    plt.close()
    generated.append("pipeline_latency_over_time.png")

    # Plot ReID Calls per interval
    if "reid_calls_since_last_sample" in rows[0]:
        plt.figure(figsize=(10, 5))
        plt.plot(times, [int(r["reid_calls_since_last_sample"]) for r in rows])
        plt.xlabel("Time [s]")
        plt.ylabel("Calls")
        plt.title("ReID Calls per Measurement Interval")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "reid_calls_over_time.png", dpi=170)
        plt.close()
        generated.append("reid_calls_over_time.png")

    print(f"Created {len(generated)} plots in {output_dir}")
    for name in generated:
        print(f"  {name}")


if __name__ == "__main__":
    main()
