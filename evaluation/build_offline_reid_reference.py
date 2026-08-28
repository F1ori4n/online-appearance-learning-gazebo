#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

# Ensure the project root is in the Python path so we can import the local
# FeatureExtractor when running this script directly from the command line.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.reid.extractor import FeatureExtractor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalise(embedding: np.ndarray) -> np.ndarray:
    """Normalize an embedding"""
    array = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array if norm <= 1e-12 else array / norm


def image_paths(directory: Path) -> list[Path]:
    """Recursively collect all images inside a directory"""
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a model-specific offline ReID reference from independent "
            "query crops arranged as QUERY_ROOT/<actor>/*.jpg."
        )
    )
    parser.add_argument("query_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="osnet_x1_0")
    parser.add_argument("--max-per-actor", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.query_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Query root does not exist: {root}")

    # Each subfolder corresponds to a different person (actor).
    actor_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not actor_dirs:
        raise SystemExit(
            f"No actor subdirectories found below {root}; expected e.g. remy/ and brian/."
        )

    # Determine the device (GPU if available, otherwise CPU).
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    extractor = FeatureExtractor(
        device=device,
        model_name=args.model,
    )

    # Collect embeddings, labels (actor names), and file paths for the reference.
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    files: list[str] = []

    for actor_dir in actor_dirs:
        paths = image_paths(actor_dir)
        if args.max_per_actor > 0:
            paths = paths[: args.max_per_actor]
        if not paths:
            print(f"Warning: no images for actor {actor_dir.name}")
            continue

        for path in paths:
            crop = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if crop is None or crop.size == 0:
                print(f"Warning: cannot read {path}")
                continue
            # Extract the embedding from the crop and normalize it.
            embeddings.append(normalise(extractor.extract(crop)))
            labels.append(actor_dir.name)
            files.append(str(path.relative_to(root)))

        print(f"Embedded actor={actor_dir.name}: {sum(x == actor_dir.name for x in labels)} images")

    if not embeddings:
        raise SystemExit("No valid query images could be embedded.")

    # Save the reference embeddings to a compressed .npz file for later offline evaluation.
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=np.stack(embeddings).astype(np.float32),
        labels=np.asarray(labels),
        files=np.asarray(files),
        model_name=np.asarray(args.model),
        weight_source=np.asarray(getattr(extractor, "weight_source", "library")),
        query_root=np.asarray(str(root)),
    )
    print(f"Saved offline reference: {output} ({len(embeddings)} embeddings)")


if __name__ == "__main__":
    main()
