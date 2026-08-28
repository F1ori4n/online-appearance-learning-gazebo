#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def normalise_rows(values: np.ndarray) -> np.ndarray:
    """Normalize all embedding rows to unit length (L2 norm)."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return array / norms


def scalar_text(value) -> str:
    """Helper to safely extract a scalar string from a numpy array."""
    array = np.asarray(value)
    return str(array.item()) if array.ndim == 0 else str(array)


def describe(scores: np.ndarray, threshold: float) -> dict:
    """Calculate statistical metrics for a given set of scores."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return {
            "count": 0, "mean": None, "median": None, "min": None,
            "max": None, "p05": None, "p95": None,
            "accepted_count": 0, "accepted_rate": None,
        }
    accepted = scores >= threshold
    return {
        "count": int(scores.size),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "p05": float(np.percentile(scores, 5)),
        "p95": float(np.percentile(scores, 95)),
        "accepted_count": int(np.sum(accepted)),
        "accepted_rate": float(np.mean(accepted)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the final ReID memory of one run against a model-specific "
            "offline actor reference."
        )
    )
    parser.add_argument("run_memory", type=Path, help="Path to the run's reid_memory.npz")
    parser.add_argument("offline_reference", type=Path, help="Path to the offline reference .npz")
    parser.add_argument("--target-actor", nargs='+', required=True,
                        help="Names of the target actors to evaluate against.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the stored ReID threshold. Default: use stored value.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_actors = args.target_actor
    memory_path = args.run_memory.expanduser().resolve()
    reference_path = args.offline_reference.expanduser().resolve()

    if not memory_path.is_file():
        raise SystemExit(f"Run memory does not exist: {memory_path}")
    if not reference_path.is_file():
        raise SystemExit(f"Offline reference does not exist: {reference_path}")

    # Load the run memory
    with np.load(memory_path, allow_pickle=False) as memory_data:
        memory_embeddings = normalise_rows(memory_data["embeddings"])
        entry_kind = np.asarray(memory_data["entry_kind"]).astype(str)
        mode = scalar_text(memory_data["mode"])
        model_name = scalar_text(memory_data["model_name"])
        memory_weight_source = scalar_text(memory_data["weight_source"])
        stored_threshold = float(np.asarray(memory_data["reassign_threshold"]).item())

    # Load the offline reference
    with np.load(reference_path, allow_pickle=False) as reference_data:
        query_embeddings = normalise_rows(reference_data["embeddings"])
        labels = np.asarray(reference_data["labels"]).astype(str)
        files = np.asarray(reference_data["files"]).astype(str)
        reference_model = scalar_text(reference_data["model_name"])
        reference_weight_source = scalar_text(reference_data["weight_source"])

    # Verify compatibility
    if model_name != reference_model:
        raise SystemExit(f"Model mismatch: run memory uses {model_name}, offline reference uses {reference_model}.")
    if memory_embeddings.shape[1] != query_embeddings.shape[1]:
        raise SystemExit("Embedding dimension mismatch between run memory and offline reference.")

    threshold = stored_threshold if args.threshold is None else float(args.threshold)

    # Calculate similarity (dot product on normalized vectors)
    similarity_matrix = query_embeddings @ memory_embeddings.T
    best_indices = np.argmax(similarity_matrix, axis=1)
    best_scores = similarity_matrix[np.arange(similarity_matrix.shape[0]), best_indices]

    # Target vs. Distractor metrics
    actor_metrics = {
        actor: describe(best_scores[labels == actor], threshold)
        for actor in sorted(set(labels.tolist()))
    }
    target_mask = np.isin(labels, target_actors)
    distractor_mask = ~target_mask
    target_scores = best_scores[target_mask]
    distractor_scores = best_scores[distractor_mask]

    # Target Accept Rate
    target_accept_rate = (
        float(np.mean(target_scores >= threshold)) if target_scores.size else None
    )
    # False Accept Rate
    false_accept_rate = (
        float(np.mean(distractor_scores >= threshold))
        if distractor_scores.size
        else None
    )
    # Balanced Accuracy
    balanced_accuracy = None
    if target_accept_rate is not None and false_accept_rate is not None:
        balanced_accuracy = 0.5 * (target_accept_rate + (1.0 - false_accept_rate))

    # Save results
    result = {
        "run_memory": str(memory_path),
        "offline_reference": str(reference_path),
        "mode": mode,
        "model_name": model_name,
        "memory_weight_source": memory_weight_source,
        "reference_weight_source": reference_weight_source,
        "target_actor": target_actors,
        "threshold": threshold,
        "memory": {
            "entries": int(memory_embeddings.shape[0]),
            "anchors": int(np.sum(entry_kind == "anchor")),
            "adaptive": int(np.sum(entry_kind == "adaptive")),
        },
        "reference": {
            "entries": int(query_embeddings.shape[0]),
            "actors": {
                actor: int(np.sum(labels == actor))
                for actor in sorted(set(labels.tolist()))
            },
        },
        "actor_metrics": actor_metrics,
        "target_accept_rate": target_accept_rate,
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": (
            None if target_accept_rate is None else 1.0 - target_accept_rate
        ),
        "balanced_accuracy": balanced_accuracy,
    }

    run_dir = memory_path.parent
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else run_dir / "offline_reid_metrics.json"
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else run_dir / "offline_reid_scores.csv"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file", "actor", "is_target", "best_score",
                "accepted", "best_memory_index", "best_memory_kind",
            ],
        )
        writer.writeheader()
        for index, score in enumerate(best_scores):
            memory_index = int(best_indices[index])
            writer.writerow(
                {
                    "file": files[index],
                    "actor": labels[index],
                    "is_target": bool(np.isin(labels[index], target_actors)),
                    "best_score": f"{float(score):.8f}",
                    "accepted": bool(score >= threshold),
                    "best_memory_index": memory_index,
                    "best_memory_kind": entry_kind[memory_index],
                }
            )

    target_count = sum(result["reference"]["actors"].get(actor, 0) for actor in target_actors)
    print(
        f"Offline ReID evaluation completed: "
        f"reference={result['reference']['entries']} images, "
        f"{target_actors}=total {target_count}"
    )
    print(f"Saved JSON: {output_json}")
    print(f"Saved CSV:  {output_csv}")


if __name__ == "__main__":
    main()