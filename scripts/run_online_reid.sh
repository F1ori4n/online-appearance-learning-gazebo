#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"

# ROS setup
source "/opt/ros/jazzy/setup.bash"

export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

# Python environment
VENV="${VENV:-$HOME/j5bth-auueu/Thesis/Code/.venv}"
if [[ -x "$VENV/bin/python3" ]]; then
  PYTHON_BIN="$VENV/bin/python3"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

# Experiment Configuration standard values are overwritten by start_follow_me.sh
MODE="${MODE:-online_reid}"
REID_MODEL="${REID_MODEL:-osnet_x1_0}"
YOLO_MODEL="${YOLO_MODEL:-yolov8s.pt}"

# RUN_TAG for naming the folders correctly
case "$MODE" in
  tracker_only)
    REID_MODEL="none"
    DEFAULT_RUN_TAG="tracker_only"
    ;;
  online_reid|calibration_reid)
    DEFAULT_RUN_TAG="${MODE}_${REID_MODEL}"
    ;;
  *)
    echo "ERROR: unsupported MODE='$MODE'." >&2
    echo "Allowed: tracker_only, online_reid, calibration_reid" >&2
    exit 2
    ;;
esac

RUN_TAG="${RUN_TAG:-$DEFAULT_RUN_TAG}"
LOG_PATH="${LOG_PATH:-logs/${RUN_TAG}.csv}" # Event log csv
PERFORMANCE_LOG_PATH="${PERFORMANCE_LOG_PATH:-}" # performance data csv
PERFORMANCE_SUMMARY_PATH="${PERFORMANCE_SUMMARY_PATH:-}" # performance data summary json
RUN_DIR="${RUN_DIR:-$(dirname "$LOG_PATH")}"

# Reid and memory initialisation configuration
GALLERY_MAX_SIZE="${GALLERY_MAX_SIZE:-30}"
CALIBRATION_START_TIME_SEC="${CALIBRATION_START_TIME_SEC:-18.0}"
ONLINE_UPDATES_START_TIME_SEC="${ONLINE_UPDATES_START_TIME_SEC:-20.0}"
CALIBRATION_EMBEDDINGS="${CALIBRATION_EMBEDDINGS:-15}"

# path to save reid memory to
RUN_MEMORY_PATH="${RUN_MEMORY_PATH:-$RUN_DIR/reid_memory.npz}"

mkdir -p "$(dirname "$LOG_PATH")" "$RUN_DIR"

# ROS 2 parameters for perception node
ARGS=(
  --ros-args
  -p "use_sim_time:=true"
  -p "mode:=${MODE}"
  -p "model_path:=${YOLO_MODEL}"
  -p "log_path:=${LOG_PATH}"
  -p "reid_model:=${REID_MODEL}"
  -p "gallery_max_size:=${GALLERY_MAX_SIZE}"
  -p "calibration_start_time_sec:=${CALIBRATION_START_TIME_SEC}"
  -p "online_updates_start_time_sec:=${ONLINE_UPDATES_START_TIME_SEC}"
  -p "calibration_embeddings:=${CALIBRATION_EMBEDDINGS}"
  -p "run_memory_path:=${RUN_MEMORY_PATH}"
  -p "performance_sample_every_n_frames:=30"

  # Reassignment parameters
  -p "reassign_threshold:=0.80"
  -p "reassign_margin:=0.08"
  -p "reassign_consecutive_frames:=3"

  # Tracking confidence
  -p "min_yolo_conf:=0.55"
  -p "min_crop_height:=90"

  # ReID interval settings
  -p "reid_stable_interval_sec:=0.5"
  -p "reid_low_conf_interval_sec:=0.20"
  -p "identity_check_threshold:=0.72"

  # Memory update policy parameters
  -p "memory_update_threshold:=0.84" # Tuned with online_reid + OSNet x1_0 in world_2. 0.84 was selected based on having a really low run-to-run variation in similarity scores.
  -p "memory_update_max_similarity:=0.97" # Tuned with online_reid + OSNet x1_0 in world_2. 0.97 was selected based on having a low run-to-run variation then 0.95 in similarity scores.
  -p "memory_update_cooldown_sec:=0.5"
  -p "memory_update_min_track_age_sec:=1.0"
  -p "memory_update_min_yolo_conf:=0.75"
  -p "memory_update_min_crop_height:=160"
)

# only add when path set (from start_follow_me.sh)
[[ -n "$PERFORMANCE_LOG_PATH" ]] && ARGS+=( -p "performance_log_path:=${PERFORMANCE_LOG_PATH}" )
[[ -n "$PERFORMANCE_SUMMARY_PATH" ]] && ARGS+=( -p "performance_summary_path:=${PERFORMANCE_SUMMARY_PATH}" )

echo "Starting perception / YOLO:"
echo "  mode                 = $MODE"
echo "  reid_model           = $REID_MODEL"
echo "  event_log            = $LOG_PATH"
echo "  performance_log      = ${PERFORMANCE_LOG_PATH:-disabled}"
echo "  performance_json     = ${PERFORMANCE_SUMMARY_PATH:-derived/disabled}"
echo "  gallery_max_size     = ${GALLERY_MAX_SIZE}"
echo "  calibration_window   = ${CALIBRATION_START_TIME_SEC}-${ONLINE_UPDATES_START_TIME_SEC}s sim time"
echo "  calibration_anchors  = ${CALIBRATION_EMBEDDINGS}"
echo "  event_crop_dir       = ${EVENT_CROP_DIR}"
echo "  run_memory           = ${RUN_MEMORY_PATH}"

# Start perception node with set parameters
exec "$PYTHON_BIN" nodes/perception_node.py "${ARGS[@]}"
