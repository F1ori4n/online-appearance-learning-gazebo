#!/usr/bin/env bash
set -Eeo pipefail

#--------------------------------------------------
# Script to run the complete Follow me Experiment
# Usage: MODE=calibration_reid REID_MODEL=osnet_x0_5 RUN_TAG=run_1 ./start_follow_me.sh
# REID_MODEL
#   osnet / osnet_x1_0 / osnet_x0_75 / osnet_x0_5 / osnet_x0_25 / resnet50
#   none (or an empty value) = YOLO + ByteTrack, without ReID
# MODE options per MODEL:
#   online_reid / calibration_reid
# The script dose:
# 1. Cleanup old processes
# 2. Launches Gazebo simulation and spawns Turtlebot 4 lite
# 3. Launches Perception node (YOLO + ByteTrack + ReID) based on set parameters
# 4. Launches Follower Node which controls the robot
# 5. Launches Path logger node
# 6. Automatic evaluates and plots results
#--------------------------------------------------

# Configuration
REID_MODEL="${REID_MODEL:-osnet_x1_0}"
MODE="${MODE:-online_reid}"
RUN_TAG="${RUN_TAG:-no_tag}"

# Experiment Duration in seconds:
# World_0 = 60 sec all others 210 sec
DURATION_SEC=210.0
WORLD_NAME="world_3"
SCENARIO="Scenario_D"

# ReID Memory settings
GALLERY_MAX_SIZE=30
CALIBRATION_START_TIME_SEC=18.0
ONLINE_UPDATES_START_TIME_SEC=20.0

# Number of calibration anchors based on mode
if [[ "$MODE" == "online_reid" ]]; then
  CALIBRATION_EMBEDDINGS="${CALIBRATION_EMBEDDINGS:-1}"
else
  CALIBRATION_EMBEDDINGS="${CALIBRATION_EMBEDDINGS:-15}"
fi

# Path setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$CODE_DIR/.venv"
WORLD_DIR="$CODE_DIR/gazebo/worlds"

# Input normalization
case "$MODE" in
  none|tracker_only)
    REID_MODEL="none"
    MODE="tracker_only"
    ;;
  online_reid|calibration_reid)
    case "$REID_MODEL" in
      osnet_x1_0|osnet_x0_75|osnet_x0_5|osnet_x0_25|resnet50) ;;
      *)
        echo "ERROR: unsupported REID_MODEL='$REID_MODEL' for MODE='$MODE'." >&2
        echo "Allowed: osnet_x1_0, osnet_x0_75, osnet_x0_5, osnet_x0_25, resnet50" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "ERROR: unsupported MODE='$MODE'." >&2
    echo "Allowed: [none:=, tracker_only], online_reid, calibration_reid" >&2
    exit 2
    ;;
esac

# Calibration mode requires a special world (world_1c, world_2c, world_3c)
# Where the person spins in a circle once for calibration
if [[ "$MODE" == "calibration_reid" && "$WORLD_NAME" != *c ]]; then
    WORLD_NAME="${WORLD_NAME}c"
fi

# Run directory setup
RUN_DIR="${CODE_DIR}/results/${SCENARIO}/${MODE}_${REID_MODEL}_${RUN_TAG}"
if [ -d "$RUN_DIR" ]; then
  echo "ERROR: Run directory already exists: ${RUN_DIR}" >&2
  echo "Remove it manually or use a different RUN_TAG." >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
mkdir -p "${RUN_DIR}/logs"

# ROS and Gazebo cleanup because they like to do weird stuff!
source /opt/ros/jazzy/setup.bash

echo "Cleaning up old Gazebo/ROS processes ..."
# Kill its trice to make sure
for _ in {1..3}; do
    pkill -f "turtlebot4_gz.launch.new.py" 2>/dev/null || true
    pkill -f "gz sim" 2>/dev/null || true
    pkill -f "gzserver" 2>/dev/null || true
    pkill -f "gzclient" 2>/dev/null || true
    pkill -f "ros2" 2>/dev/null || true   # hilft ggf. bei hängenden ROS-Nodes
    sleep 1
done

# Wait till /clock is gone max 30s
WAIT_COUNT=0
MAX_WAIT=30
while ros2 topic list 2>/dev/null | grep -Fxq /clock; do
    if (( WAIT_COUNT >= MAX_WAIT )); then
        echo "ERROR: /clock still exists after ${MAX_WAIT}s cleanup. Force killing all Gazebo processes." >&2
        pkill -KILL -f "gz sim" 2>/dev/null || true
        pkill -KILL -f "gzserver" 2>/dev/null || true
        pkill -KILL -f "gzclient" 2>/dev/null || true
        ros2 daemon stop 2>/dev/null || true
        sleep 2
        # Check again if its finally gone
        if ros2 topic list 2>/dev/null | grep -Fxq /clock; then
            echo "ERROR: /clock still present after force cleanup. Manual intervention required." >&2
            exit 1
        fi
        break
    fi
    echo "  Waiting for /clock to disappear (${WAIT_COUNT}s) ..."
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    # Try again maybe something started again
    pkill -f "turtlebot4_gz.launch.new.py" 2>/dev/null || true
    pkill -f "gz sim" 2>/dev/null || true
    pkill -f "gzserver" 2>/dev/null || true
done
echo "Cleanup complete. /clock is gone."

# Helpers wait for topics
# This is used to make sure the simulation is fully up before starting other nodes
wait_for_topic() {
  local topic="$1"
  local timeout_sec="${2:-60}"
  local watched_pid="${3:-}"
  local watched_log="${4:-}"
  local start=$SECONDS

  echo "Waiting for ${topic} ..."
  until ros2 topic list 2>/dev/null | grep -Fxq "$topic"; do
    if [[ -n "$watched_pid" ]] && ! kill -0 "$watched_pid" 2>/dev/null; then
      echo "ERROR: process exited before ${topic} appeared." >&2
      [[ -f "$watched_log" ]] && tail -n 80 "$watched_log" >&2 || true
      return 1
    fi
    if (( SECONDS - start >= timeout_sec )); then
      echo "ERROR: timed out waiting for ${topic}" >&2
      [[ -f "$watched_log" ]] && tail -n 80 "$watched_log" >&2 || true
      return 1
    fi
    sleep 1
  done
}

# Safe cleanup on exit
# This ensures all started process groups are killed if script is aborted.
# Might not always work that is why startup cleanup exists for automatic testing
PGIDS=()
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  echo "Stopping experiment processes ..."

  for pgid in "${PGIDS[@]}"; do
    kill -INT -- "-${pgid}" 2>/dev/null || true
  done
  sleep 3
  for pgid in "${PGIDS[@]}"; do
    kill -TERM -- "-${pgid}" 2>/dev/null || true
  done
  sleep 5
  for pgid in "${PGIDS[@]}"; do
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  done

  # Terminate all Gazebo/ROS-processes
  pkill -TERM -f "turtlebot4_gz.launch.new.py.*world:=${WORLD_NAME}" 2>/dev/null || true
  pkill -TERM -f "gz sim.*${WORLD_NAME}" 2>/dev/null || true
  pkill -TERM -f "gzserver" 2>/dev/null || true
  pkill -TERM -f "gzclient" 2>/dev/null || true   # Sometimes it helps to kill it twice
  pkill -TERM -f "ros2" 2>/dev/null || true

  sleep 3
  pkill -KILL -f "gzclient" 2>/dev/null || true
  pkill -KILL -f "gz sim" 2>/dev/null || true
  pkill -KILL -f "gzserver" 2>/dev/null || true
  pkill -KILL -f "gzclient" 2>/dev/null || true

  ros2 daemon stop 2>/dev/null || true

  echo "Results: ${RUN_DIR}"
  exit "$code"
}
trap cleanup EXIT INT TERM

echo "Configuration: mode=${MODE}, reid_model=${REID_MODEL}, world=${WORLD_NAME}, fixed_duration=${DURATION_SEC}s"

# 1. Launch Simulation
echo "1/4 Starting TurtleBot4 simulation ..."
(
  cd "$WORLD_DIR"
  exec setsid ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.new.py \
    model:=lite \
    world:="$WORLD_NAME" \
    spawn_dock:=false \
    slam:=false \
    nav2:=false \
    rviz:=false \
    x:=-6 \
    y:=6.5 \
    yaw:=0
) >"${RUN_DIR}/logs/gazebo.log" 2>&1 &
GAZEBO_PID=$!
PGIDS+=("$GAZEBO_PID")

# Wait for the core topics to appear
wait_for_topic /clock 45 "$GAZEBO_PID"
wait_for_topic /odom 45 "$GAZEBO_PID"
wait_for_topic /scan 45 "$GAZEBO_PID"
wait_for_topic /oakd/rgb/preview/image_raw 45 "$GAZEBO_PID"

# Enable full safety override so the robot can reach top speed (0.46)
echo "Waiting for /motion_control and enabling full safety override ..."
SAFETY_START=$SECONDS
until ros2 param set /motion_control safety_override full >/dev/null 2>&1; do
  if (( SECONDS - SAFETY_START >= 45 )); then
    echo "ERROR: timed out setting /motion_control safety_override." >&2
    exit 1
  fi
  sleep 1
done

# 2. Start perception node
echo "2/4 Starting YOLO / tracking / ReID ..."
(
  cd "$CODE_DIR"
  exec setsid env PATH="${VENV}/bin:${PATH}" \
    MODE="${MODE}" \
    REID_MODEL="${REID_MODEL}" \
    GALLERY_MAX_SIZE="${GALLERY_MAX_SIZE}" \
    CALIBRATION_START_TIME_SEC="${CALIBRATION_START_TIME_SEC}" \
    ONLINE_UPDATES_START_TIME_SEC="${ONLINE_UPDATES_START_TIME_SEC}" \
    CALIBRATION_EMBEDDINGS="${CALIBRATION_EMBEDDINGS}" \
    RUN_TAG="${RUN_TAG}" \
    RUN_DIR="${RUN_DIR}" \
    LOG_PATH="${RUN_DIR}/perception_events.csv" \
    PERFORMANCE_LOG_PATH="${RUN_DIR}/performance.csv" \
    PERFORMANCE_SUMMARY_PATH="${RUN_DIR}/performance_summary.json" \
    "${CODE_DIR}/scripts/run_online_reid.sh"
  )  >"${RUN_DIR}/logs/perception.log" 2>&1 &
PERCEPTION_PID=$!
PGIDS+=("$PERCEPTION_PID")

# Wait until the perception node is fully up and publishes the control signal
wait_for_topic /target_person/control 300 "$PERCEPTION_PID"
wait_for_topic /target_person/visible 20 "$PERCEPTION_PID"

# 3. Start follower node
echo "3/4 Starting follower ..."
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${VENV}/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

setsid "$PYTHON_BIN" "${CODE_DIR}/nodes/cmd_vel_follow_node.py" \
  >"${RUN_DIR}/logs/follower.log" 2>&1 &
FOLLOWER_PID=$!
PGIDS+=("$FOLLOWER_PID")

# Check if follower started
sleep 2
if ! kill -0 "$FOLLOWER_PID" 2>/dev/null; then
  echo "ERROR: follower exited during startup." >&2
  tail -n 80 "${RUN_DIR}/logs/follower.log" >&2 || true
  exit 1
fi

# Write metadata for the current run
cat >"${RUN_DIR}/metadata.txt" <<EOF
=============================================
  Run Metadata
=============================================
Scenario              : ${SCENARIO}
World                 : ${WORLD_NAME}
Duration (sec)        : ${DURATION_SEC}
Run tag               : ${RUN_TAG}
Mode                  : ${MODE}
ReID model            : ${REID_MODEL}
Calibration start (s) : ${CALIBRATION_START_TIME_SEC}
Online updates start  : ${ONLINE_UPDATES_START_TIME_SEC}
Calibration embeddings: ${CALIBRATION_EMBEDDINGS}
GALLERY_MAX_SIZE      : ${GALLERY_MAX_SIZE}
Robot start           : x = -6.0, y = 6.5, yaw = 0.0
=============================================
EOF

# 4. Start path logger
echo "4/4 Starting path logger for ${DURATION_SEC}s ..."
setsid "${VENV}/bin/python3" "${CODE_DIR}/nodes/robot_path_logger.py" --ros-args \
  -p use_sim_time:=true \
  -p "output_dir:=${RUN_DIR}" \
  -p "end_sim_time_sec:=0.0" \
  -p "duration_sec:=${DURATION_SEC}" \
  -p "scenario:=${SCENARIO}" \
  -p "mode:=${MODE}" \
  -p "reid_model:=${REID_MODEL}" \
  >"${RUN_DIR}/logs/path_logger.log" 2>&1 &
LOGGER_PID=$!
PGIDS+=("$LOGGER_PID")

# Wait for the experiment to finish (duration passed)
wait "$LOGGER_PID"
echo "Timed experiment completed."

# Finalize
# Stop the consumers first so their CSV files are flushed and the final
# performance_summary.json is written before post-processing starts.
echo "Finalizing perception and follower logs ..."
kill -INT -- "-${FOLLOWER_PID}" 2>/dev/null || true
kill -INT -- "-${PERCEPTION_PID}" 2>/dev/null || true

wait_for_process_exit() {
  local pid="$1"
  local timeout_sec="${2:-20}"
  local start=$SECONDS
  local state=""

  while kill -0 "$pid" 2>/dev/null; do
    state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
    [[ "$state" == Z* ]] && return 0
    if (( SECONDS - start >= timeout_sec )); then
      return 1
    fi
    sleep 0.25
  done
  return 0
}

if ! wait_for_process_exit "$PERCEPTION_PID" 20; then
  echo "WARNING: perception did not stop after SIGINT; sending SIGTERM."
  kill -TERM -- "-${PERCEPTION_PID}" 2>/dev/null || true
  wait_for_process_exit "$PERCEPTION_PID" 5 || true
fi

if ! wait_for_process_exit "$FOLLOWER_PID" 10; then
  echo "WARNING: follower did not stop after SIGINT; sending SIGTERM."
  kill -TERM -- "-${FOLLOWER_PID}" 2>/dev/null || true
  wait_for_process_exit "$FOLLOWER_PID" 5 || true
fi

wait "$PERCEPTION_PID" 2>/dev/null || true
wait "$FOLLOWER_PID" 2>/dev/null || true

# Give the filesystem a moment after the logger has closed its files.
sleep 1

# Post processing
# Evaluate the final run memory against the independent actor-query reference.
if [[ "$MODE" != "tracker_only" ]]; then
  OFFLINE_REFERENCE="${CODE_DIR}/evaluation/reid_references/${REID_MODEL}.npz"
  EVALUATOR="${CODE_DIR}/evaluation/evaluate_run_memory.py"

  if [[ -f "${RUN_DIR}/reid_memory.npz" \
        && -f "$OFFLINE_REFERENCE" \
        && -f "$EVALUATOR" ]]; then
    echo "Evaluating final ReID memory against offline actor queries ..."

    PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}" \
    "${VENV}/bin/python3" "$EVALUATOR" \
      "${RUN_DIR}/reid_memory.npz" \
      "$OFFLINE_REFERENCE" \
      --target-actor remy remy_depot remyblue remyblue_depot\
      --output-json "${RUN_DIR}/offline_reid_metrics.json" \
      --output-csv "${RUN_DIR}/offline_reid_scores.csv" \
      || echo "WARNING: offline ReID evaluation failed."
  else
    echo "WARNING: offline ReID evaluation skipped."
    [[ ! -f "${RUN_DIR}/reid_memory.npz" ]] \
      && echo "  missing: ${RUN_DIR}/reid_memory.npz"
    [[ ! -f "$OFFLINE_REFERENCE" ]] \
      && echo "  missing: $OFFLINE_REFERENCE"
    [[ ! -f "$EVALUATOR" ]] \
      && echo "  missing: $EVALUATOR"
  fi
fi

# Plot the robots path
PLOTTER="${CODE_DIR}/evaluation/plot_robot_paths.py"
if [[ -f "$PLOTTER" && -f "${RUN_DIR}/robot_path.csv" ]]; then
  echo "Creating robot path plot ..."
  env MPLBACKEND=Agg "${VENV}/bin/python3" "$PLOTTER" "$RUN_DIR" \
    --output "$RUN_DIR/robot_path.png" \
    || echo "WARNING: robot path plot could not be created."
else
  echo "WARNING: robot path plotter or robot_path.csv is missing."
fi

# Plot performance time series
PERFORMANCE_PLOTTER="${CODE_DIR}/evaluation/plot_performance.py"
if [[ -f "$PERFORMANCE_PLOTTER" && -f "${RUN_DIR}/performance.csv" ]]; then
  echo "Creating performance time-series plots ..."
  env MPLBACKEND=Agg "${VENV}/bin/python3" "$PERFORMANCE_PLOTTER" "$RUN_DIR" \
    --output-dir "$RUN_DIR/performance_plots" \
    || echo "WARNING: performance plots could not be created."
else
  echo "WARNING: performance plotter or performance.csv is missing."
fi

# Print output directory
echo
echo "Run completed. Run directory:"
echo "  ${RUN_DIR}"
