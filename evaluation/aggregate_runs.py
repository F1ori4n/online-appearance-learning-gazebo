#!/usr/bin/env python3
"""
Aggregate all results from the 5 runs of every (Scenario, Mode, Model) cell

This script combines:
1. General run metrics from summary.json and performance_summary.json
2. ReID quality metrics from offline_reid_metrics.json
3. Per-actor detection rates from offline_reid_scores.csv

Usage:
    python3 aggregate_all_results.py results [output_prefix]
"""
from __future__ import annotations
import os, sys, json, glob, statistics, csv

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT_PREFIX = sys.argv[2] if len(sys.argv) > 2 else "aggregated"

ONLINE_FIELDS = ["path_length_m", "displacement_m"]
REID_FIELDS = ["target_accept_rate", "false_accept_rate", "false_reject_rate", "balanced_accuracy"]
PERF_FIELDS = ["mean_fps", "callback_total_ms", "tracking_ms", "target_update_ms", "total_reid_calls"]
MEMORY_FIELDS = ["entries", "anchors", "adaptive"]
# Memory update statistics (from perception_events.csv)
MEMORY_UPDATE_FIELDS = ["accepted_updates", "rejected_updates", "rejected_redundant",
                        "rejected_low_similarity", "rejected_low_quality", "rejected_cooldown"]

ALL_FIELDS = ONLINE_FIELDS + REID_FIELDS + PERF_FIELDS
CSV_MEMORY_FIELDS = ["memory_" + f for f in MEMORY_FIELDS]

ACTORS = ["remy", "remy_depot", "remyblue", "remyblue_depot",
          "brian", "brian_depot", "brianblue", "brianblue_depot"]
MODES = ("calibration_reid", "online_reid", "tracker_only")


def parse_cell(folder_path):
    """Return (mode, model, run_idx) from a run folder name, or None."""
    metadata_path = os.path.join(folder_path, "metadata.txt")
    if not os.path.isfile(metadata_path):
        return None

    data = {}
    try:
        with open(metadata_path, encoding="utf-8") as fh:
            for line in fh:
                if ":" in line:
                    key, value = line.split(":", 1)
                    data[key.strip()] = value.strip()
    except OSError:
        return None

    mode = data.get("Mode", "")
    model = data.get("ReID model", "")

    if mode == "tracker_only":
        model = "none"

    return mode, model, 0


def load_json(path):
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"  [warn] could not parse {path}: {exc}")
        return {}


def get_mean(d, key):
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("mean")
    return v


def fmt_mean_sd(vals):
    nums = [v for v in vals if isinstance(v, (int, float))]
    if not nums:
        return "--", "--"
    if len(nums) == 1:
        return f"{nums[0]:.4f}", "--"
    mu = statistics.mean(nums)
    sd = statistics.pstdev(nums)
    return f"{mu:.4f}", f"{sd:.4f}"


def read_actor_counts(scores_path):
    counts = {}
    with open(scores_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            actor = row.get("actor")
            if actor is None:
                continue
            entry = counts.setdefault(actor, {"accepted": 0, "crops": 0})
            entry["crops"] += 1
            if row.get("accepted", "").strip().lower() == "true":
                entry["accepted"] += 1
    return counts

def read_memory_update_stats(events_path):
    """Count Memory Update events from the perception_events.csv"""
    counts = {f: 0 for f in MEMORY_UPDATE_FIELDS}
    if not os.path.isfile(events_path):
        return counts
    try:
        with open(events_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                event = row.get("event", "")
                if "MEMORY_UPDATE_ACCEPTED" in event:
                    counts["accepted_updates"] += 1
                if "MEMORY_UPDATE_REJECTED" in event:
                    counts["rejected_updates"] += 1
                    if "REDUNDANT" in event:
                        counts["rejected_redundant"] += 1
                    elif "LOW_SIMILARITY" in event:
                        counts["rejected_low_similarity"] += 1
                    elif "LOW_QUALITY" in event:
                        counts["rejected_low_quality"] += 1
                    elif "COOLDOWN" in event:
                        counts["rejected_cooldown"] += 1
    except Exception as exc:
        print(f"  [warn] could not parse {events_path}: {exc}")
    return counts

# Collect all runs
cells = {}
found_any = False

for summary in glob.glob(os.path.join(ROOT, "*", "*", "summary.json"), recursive=True):
    run_dir = os.path.dirname(summary)
    scenario = os.path.basename(os.path.dirname(run_dir))

    parsed = parse_cell(run_dir)
    if parsed is None:
        print(f"  [skip] unrecognised run folder: {run_dir}")
        continue
    mode, model, idx = parsed
    found_any = True

    # Main metrics
    rec = {}
    reid = load_json(os.path.join(run_dir, "offline_reid_metrics.json"))
    rec.update(reid)
    rec["target_actor"] = reid.get("target_actor", "N/A")
    rec.update(load_json(os.path.join(run_dir, "summary.json")))
    perf = load_json(os.path.join(run_dir, "performance_summary.json"))
    events_path = os.path.join(run_dir, "perception_events.csv")
    rec.update(read_memory_update_stats(events_path))
    for f in PERF_FIELDS:
        v = perf.get(f)
        if isinstance(v, dict):
            rec[f] = v.get("mean")
        elif v is not None:
            rec[f] = v

    # Memory composition
    mem = rec.get("memory", {}) if reid else {}
    for f in MEMORY_FIELDS:
        rec["memory_" + f] = mem.get(f)

    # actor detection stats
    scores_path = os.path.join(run_dir, "offline_reid_scores.csv")
    if os.path.isfile(scores_path):
        rec["actor_counts"] = read_actor_counts(scores_path)
    else:
        rec["actor_counts"] = {}

    cells.setdefault((scenario, mode, model), []).append(rec)

if not found_any:
    print(f"No runs found under {ROOT!r} (expected results/<Scenario>/<mode>_<model>_run_<n>/...)")
    sys.exit(1)

# Aggregate main metrics and write CSV
rows = []
for (scenario, mode, model), recs in sorted(cells.items()):
    row = {"scenario": scenario, "mode": mode, "model": model, "target_actor": recs[0].get("target_actor", "N/A"),
           "n": len(recs)}
    for f in ALL_FIELDS:
        vals = [get_mean(r, f) for r in recs]
        mu, sd = fmt_mean_sd(vals)
        row[f + "_mean"] = mu
        row[f + "_sd"] = sd
    for f in CSV_MEMORY_FIELDS:
        vals = [r.get(f) for r in recs]
        mu, sd = fmt_mean_sd(vals)
        row[f + "_mean"] = mu
        row[f + "_sd"] = sd
    for f in MEMORY_UPDATE_FIELDS:
        vals = [get_mean(r, f) for r in recs]
        mu, sd = fmt_mean_sd(vals)
        row[f + "_mean"] = mu
        row[f + "_sd"] = sd
    rows.append(row)

# Write combined CSV
if rows:
    with open(f"{OUT_PREFIX}_results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {OUT_PREFIX}_results.csv")

# Aggregate actor detection rates and write CSV
detection_rows = []
for (scenario, mode, model), recs in sorted(cells.items()):
    # Only consider cells that have actor_counts data
    if not any("actor_counts" in r and r["actor_counts"] for r in recs):
        continue
    wide = {"scenario": scenario, "mode": mode, "model": model, "n": len(recs)}
    for actor in ACTORS:
        present = [r["actor_counts"].get(actor) for r in recs if actor in r.get("actor_counts", {})]
        if not present:
            continue
        accepted = [p["accepted"] for p in present]
        crops = [p["crops"] for p in present]
        crops_mean = statistics.mean(crops)
        rates = [a / c for a, c in zip(accepted, crops)]
        acc_mean = statistics.mean(accepted)
        acc_sd = statistics.pstdev(accepted) if len(accepted) > 1 else 0.0
        rate_mean = statistics.mean(rates)
        rate_sd = statistics.pstdev(rates) if len(rates) > 1 else 0.0
        wide[f"accepted_{actor}_mean"] = round(acc_mean, 4)
        wide[f"accepted_{actor}_sd"] = round(acc_sd, 4)
        wide[f"rate_{actor}"] = round(rate_mean, 4)
    detection_rows.append(wide)

if detection_rows:
    with open(f"{OUT_PREFIX}_detection_rates.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(detection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detection_rows)
    print(f"Wrote {len(detection_rows)} rows -> {OUT_PREFIX}_detection_rates.csv")
