#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def resolve_csv(path: Path) -> Path:
    """
    Resolve the input path to a robot_path.csv file.
    Accepts either a run directory or a direct path to the CSV.
    """
    if path.is_dir():
        path = path / "robot_path.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_path(path: Path) -> tuple[list[float], list[float]]:
    """
    Extract the x and y odometry coordinates from the CSV.
    The logger writes the robot's physical position (odometry) over time.
    """
    xs: list[float] = []
    ys: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # The CSV might have empty rows; check before parsing.
            if row.get("robot_x") and row.get("robot_y"):
                xs.append(float(row["robot_x"]))
                ys.append(float(row["robot_y"]))
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare robot paths (trajectories) from timed experiments."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Run directories or robot_path.csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("robot_paths.png"),
        help="Output image path (default: robot_paths.png)",
    )
    args = parser.parse_args()

    # Iterate over all provided runs and plot their paths.
    for raw in args.runs:
        path = resolve_csv(raw)
        xs, ys = load_path(path)
        if not xs:
            print(f"Skipping empty path: {path}")
            continue

        # Use the run directory name as the label
        label = path.parent.name
        plt.plot(xs, ys, label=label)

        # Mark the start (circle) and end (cross) of the trajectory.
        plt.scatter([xs[0]], [ys[0]], marker="o")
        plt.scatter([xs[-1]], [ys[-1]], marker="x")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Robot trajectories")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()