#!/usr/bin/env bash
set -Eeo pipefail

# Controlled ReID crop capture for the offline benchmark only for bench worlds.
# Usage: ACTOR=Remy WORLD_NAME=abstract ./scripts/collect_reid_crops.sh
# Worlds options: abstract, depot
# Actor options: Brian, Brian-blue, Remy, Remy-blue
# Afterward, move pictures to `evaluation/reid_dataset/query` in a folder with the name of the actor to create a reference.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"

VENV="$CODE_DIR/.venv"
WORLD_DIR="$CODE_DIR/gazebo/worlds/bench"

# World and actor configuration
WORLD_NAME="${WORLD_NAME:-abstract}" 
ACTOR="${ACTOR:-Remy}"

# Capture timing parameters
CAPTURE_START_SEC=19.0
CAPTURE_END_SEC=24.8
STOP_SEC=25.5

# Image processing
SAVE_INTERVAL_SEC=0.08
MIN_YOLO_CONF=0.70
MIN_CROP_HEIGHT=160
SAVE_CONTEXT=false

# Paths
PYTHON_BIN="$VENV/bin/python3"
COLLECTOR="$CODE_DIR/nodes/reid_crop_collector.py"
WORLD_TEMPLATE="$WORLD_DIR/$WORLD_NAME.sdf"


GENERATED_WORLD_NAME="${WORLD_NAME}_${ACTOR}_generated"
WORLD_FILE="$WORLD_DIR/$GENERATED_WORLD_NAME.sdf"

#inject actor and world name into template world file
sed -e "s/{{ACTOR}}/$ACTOR/g" -e "s/{{WORLD_NAME}}/$GENERATED_WORLD_NAME/g" "$WORLD_TEMPLATE" > "$WORLD_FILE"

# output and log directories
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$CODE_DIR/evaluation/reid_dataset/raw/$ACTOR/$STAMP"
LOG_DIR="$CODE_DIR/logs/calibration_capture/$STAMP"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

GAZEBO_LOG="$LOG_DIR/gazebo.log"
COLLECTOR_LOG="$LOG_DIR/collector.log"

source /opt/ros/jazzy/setup.bash

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$COLLECTOR" ]]; then
  echo "ERROR: collector not found: $COLLECTOR" >&2
  exit 1
fi
if [[ ! -f "$WORLD_FILE" ]]; then
  echo "ERROR: world not found: $WORLD_FILE" >&2
  exit 1
fi
if ros2 topic list 2>/dev/null | grep -Fxq /clock; then
  echo "ERROR: /clock already exists. Close the old Gazebo simulation first." >&2
  exit 1
fi

PGIDS=()
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  set +e

  echo "Stopping calibration processes ..."
  for pgid in "${PGIDS[@]}"; do
    kill -INT -- "-$pgid" 2>/dev/null || true
  done
  sleep 3
  for pgid in "${PGIDS[@]}"; do
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done
  sleep 1
  for pgid in "${PGIDS[@]}"; do
    kill -KILL -- "-$pgid" 2>/dev/null || true
  done

  # Stop leftover Gazebo processes
  pkill -TERM -f "turtlebot4_gz.launch.new.py.*world:=$GENERATED_WORLD_NAME" 2>/dev/null || true
  pkill -TERM -f "gz sim.*$GENERATED_WORLD_NAME" 2>/dev/null || true

  echo "Crops: $OUTPUT_DIR"
  echo "Logs:  $LOG_DIR"
  exit "$code"
}
trap cleanup EXIT INT TERM

wait_for_topic() {
  local topic="$1"
  local timeout_sec="${2:-60}"
  local watched_pid="${3:-}"
  local watched_log="${4:-}"
  local start=$SECONDS

  echo "Waiting for $topic ..."
  until ros2 topic list 2>/dev/null | grep -Fxq "$topic"; do
    if [[ -n "$watched_pid" ]] && ! kill -0 "$watched_pid" 2>/dev/null; then
      echo "ERROR: simulation exited before $topic appeared." >&2
      [[ -f "$watched_log" ]] && tail -n 100 "$watched_log" >&2 || true
      return 1
    fi
    if (( SECONDS - start >= timeout_sec )); then
      echo "ERROR: timed out waiting for $topic." >&2
      [[ -f "$watched_log" ]] && tail -n 100 "$watched_log" >&2 || true
      return 1
    fi
    sleep 1
  done
}

echo "World:  $WORLD_NAME"
echo "Actor:  $ACTOR"

# Start reid_crop_collector.py node
echo "Starting crop collector node"
(
  cd "$CODE_DIR"
  exec setsid env \
    PATH="$VENV/bin:$PATH" \
    PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}" \
    "$PYTHON_BIN" "$COLLECTOR" --ros-args \
      -p "model_path:=yolov8s.pt" \
      -p "tracker_config:=bytetrack.yaml" \
      -p "output_dir:=$OUTPUT_DIR" \
      -p "capture_start_sec:=$CAPTURE_START_SEC" \
      -p "capture_end_sec:=$CAPTURE_END_SEC" \
      -p "stop_sec:=$STOP_SEC" \
      -p "save_interval_sec:=$SAVE_INTERVAL_SEC" \
      -p "min_yolo_conf:=$MIN_YOLO_CONF" \
      -p "min_crop_height:=$MIN_CROP_HEIGHT" \
      -p "save_context:=$SAVE_CONTEXT"
 ) >"$COLLECTOR_LOG" 2>&1 &
COLLECTOR_PID=$!
PGIDS+=("$COLLECTOR_PID")

# Start Gazebo simulation with modified world file
echo "Starting TurtleBot4 simulation '$GENERATED_WORLD_NAME' ..."
(
  cd "$WORLD_DIR"
  exec setsid ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.new.py \
    model:=lite \
    world:="$GENERATED_WORLD_NAME" \
    spawn_dock:=false \
    slam:=false \
    nav2:=false \
    rviz:=false \
    x:=-2 \
    y:=0 \
    yaw:=0
) >"$GAZEBO_LOG" 2>&1 &
GAZEBO_PID=$!
PGIDS+=("$GAZEBO_PID")

# Wait for simulation to started
wait_for_topic /clock 60 "$GAZEBO_PID" "$GAZEBO_LOG"
wait_for_topic /oakd/rgb/preview/image_raw 60 "$GAZEBO_PID" "$GAZEBO_LOG"

# check if collector is running
if ! kill -0 "$COLLECTOR_PID" 2>/dev/null; then
  echo "ERROR: collector exited during startup." >&2
  tail -n 120 "$COLLECTOR_LOG" >&2 || true
  exit 1
fi

echo "Simulation is running. Capturing from ${CAPTURE_START_SEC}s to ${CAPTURE_END_SEC}s."

# Wait for the collector to finish
set +e
wait "$COLLECTOR_PID"
COLLECTOR_RC=$?
set -e

if (( COLLECTOR_RC != 0 )); then
  echo "ERROR: collector exited with code $COLLECTOR_RC." >&2
  tail -n 160 "$COLLECTOR_LOG" >&2 || true
  exit "$COLLECTOR_RC"
fi

# Count saved crops
CROP_COUNT="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'crop_*.jpg' ! -name '*_context.jpg' | wc -l)"

echo "Crop collection completed: $CROP_COUNT crops saved."

# shutdown simulation
echo "Stopping Gazebo ..."
kill -INT -- "-$GAZEBO_PID" 2>/dev/null || true
sleep 3
exit 0

