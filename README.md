# Online Appearance Learning for a Vision Based Follow Me Robot in a Gazebo Simulation

This repository contains the implementation of a person-following robot system using ROS2, YOLO, ByteTrack, and Re-Identification (ReID). 
The system is designed to operate in a Gazebo simulation with a TurtleBot4.

## Overview

The system is organized into three main layers:
1. **Perception:** YOLO detection, ByteTrack tracking, feature extraction, and the Target Manager (state machine for ReID).
2. **Decision & Control:** The `cmd_vel_follow_node` translates the visual signals (offset, confidence, bounding box height) into robot velocities, including obstacle avoidance and breadcrumb recovery.
3. **Evaluation:** Automated logging, offline ReID benchmarks, and performance analysis.

## Dependencies
- ROS2 Jazzy
- Gazebo (TurtleBot4 simulation package)
- Python 3.12

The project was developed and tested with the following core packages:
- `ultralytics` 8.4.45 (YOLO Detection)
- `torch` 2.9.1 + `torchvision` 0.24.0
- `torchreid` 0.2.5 (Re-Identification)
- `opencv-python` 4.10.0
- `numpy` 1.26.4
- `matplotlib` 3.10.9 (Plotting)
- ROS2 / `rclpy` 7.1.11 + `cv-bridge` 4.1.0 (System dependency)

Note: `torch` and `torchvision` are compiled with ROCm (AMD GPU).

### System Modifications & Patches

This project relies on several custom modifications to the standard TurtleBot4 simulation to ensure optimal performance and correct camera orientation. All modified system files are backed up in the `patches/` folder.

- `create3.urdf.xacro`
  Target path: `/opt/ros/jazzy/share/irobot_create_description/urdf/`  
  Modified the rendering engine from `ogre` to `ogre2` to improve simulation performance.

- `oakd.urdf.xacro`  
  Target path: `/opt/ros/jazzy/share/turtlebot4_description/urdf/sensors`  
  Changes resolution to 640x480.

- `turtlebot4.urdf.xacro`  
  Target path: `/opt/ros/jazzy/share/turtlebot4_description/urdf/lite/`  
  Added an `rpy` offset to the OAK-D camera so it points slightly upwards.

- `turtlebot4_gz.launch.new.py`  
  Target path: `/opt/ros/jazzy/share/turtlebot4_gz_bringup/launch/`  
  Modified main launch file to load it without dock.

- `turtlebot4_spawn.launch.py`  
  Target path: `/opt/ros/jazzy/share/turtlebot4_gz_bringup/launch/`  
  Modified spawn configuration to load it without dock.

Note: The simulation script `scripts/start_follow_me.sh` automatically enables the full safety override (`ros2 param set /motion_control safety_override full`) to disable robot safety limits during the experiments.

## Project Structure
```text
.
├── app/                    # Core Python modules (Target Manager, ReID, Trackers)
├── nodes/                  # ROS2 Node entry points (Perception, Follower, Path Logger)
├── scripts/                # Bash scripts to launch experiments (start_follow_me.sh)
├── evaluation/             # Offline references, evaluation, and plotting scripts
├── gazebo/                 # Simulation worlds (world_2, world_2c, etc.)
└── results/                # Output directories for all experiments
``` 
---
## Experiment Modes
 `tracker_only`
- YOLO + ByteTrack
- No ReID memory.

`online_reid`
- Single crop used as the initial anchor.
- Adaptive updates start at t=20s.

`calibration_reid`
- 15 anchors collected between t=18s and t=20s. 
- Adaptive updates afterwards.

`online_reid`
- maximales Gallery-Budget: 30
---
### Simulation Environment and Assets
The worlds(world_1*, world_2*, world_3*) share the [Depot map made by Open Robotics](https://app.gazebosim.org/OpenRobotics/fuel/models/Depot), which is a closed room with warehouse shelves, pallets, and many other obstacles. Each world has a normal version and a calibrated version, which only differs in that the person spins once in a circle for calibration. The actor models used, Remy and Brian are from [Mixamo](https://www.mixamo.com/), downloaded with a basic walking animation. Mixamo characters and animations are royalty-free for personal, commercial projects.

---
## Setup
### Generate offline References (rebuild these files whenever the query dataset changes):
Expected dataset structure
```text
evaluation/reid_dataset/query/
├── remy/
│   └── *.jpg
├── brian/
│   └── *.jpg
└── ...
```
```bash
mkdir -p evaluation/reid_references
for model in osnet_x1_0 osnet_x0_75 osnet_x0_5 osnet_x0_25 resnet50; do
  PYTHONPATH="$PWD:${PYTHONPATH:-}" \
  .venv/bin/python3 evaluation/build_offline_reid_reference.py \
    evaluation/reid_dataset/query \
    --model "$model" \
    --output "evaluation/reid_references/${model}.npz"
done
```
### To run a single experiment:
```bash
MODE=online_reid REID_MODEL=osnet_x1_0 RUN_TAG=test_run ./scripts/start_follow_me.sh
```
Note: worlds and scneario name can be changed in `start_follow_me.sh`.

### To run 5 experiments per model and mode:
```bash
for model in osnet_x1_0 osnet_x0_75 osnet_x0_5 osnet_x0_25 resnet50; do
  for mode in online_reid calibration_reid; do
    for run in 1 2 3 4 5; do
      MODE="$mode" REID_MODEL="$model" RUN_TAG="run_${run}" ./scripts/start_follow_me.sh
    done
  done
done
```
### For the tracker-only baseline:
```bash
for run in 1 2 3 4 5; do
  MODE=none RUN_TAG="run_${run}" ./scripts/start_follow_me.sh
done
```
---
## Output Files
After a run, the following files are generated in `results/<Scenario>/<mode>_<model>_<RUN_TAG>/`:

- `metadata.txt`: Experiment parameters (World, Mode, Model, Duration, Calibration settings).
- `perception_events.csv`: Events log (ReID checks, Memory updates, Target loss, Reassignments).
- `robot_path.csv`: Kinematic data (Timestamps, Position, Path Length, Velocity commands).
- `robot_path.png`: Visualization of the robot's trajectory.
- `summary.json`: Path summary (Duration, Path Length, Start/End Pose).
- `performance.csv`: Performance time-series (FPS, Tracking latency, Target update latency, ReID calls).
- `performance_summary.json`: Aggregated performance metrics (Mean FPS, Latencies).
- `performance_plots/`: Plots of throughput and latency over time.
- `logs/`: Console logs (gazebo.log, perception.log, follower.log, path_logger.log).
- `reid_memory.npz` (ReID modes only): Final memory gallery (Embeddings, Anchors, Adaptive entries).
- `offline_reid_metrics.json` (ReID modes only): Benchmark results (Target/False Accept Rate, Balanced Accuracy).
- `offline_reid_scores.csv` (ReID modes only): Per-query-image scores and acceptance.

---
## Aggregate Results over All Runs
Use aggregate_all_results.py to compute mean ± std for all metrics (Path, ReID, Performance) across the 5 runs.
```bash
.venv/bin/python3 evaluation/aggregate_runs.py results results/my_results
```


### Plotting Robot path
```bash
.venv/bin/python3 evaluation/plot_robot_paths.py \
  results/Scenario_/online_reid_osnet_x1_0_run1 \
  results/Scenario_/online_reid_osnet_x0_5_run1 \
  --output path_compare.png
```

---
## Note on the Use of Generative AI
Generative AI tools were used to support the programming processes for this code. Specifically, ChatGPT (OpenAI, GPT-5.5) was used to assist with code structuring, improving my code, debugging and to and to quickly test and iterate over new ideas during the development process. 
All AI-generated outputs were carefully reviewed, revised and validated prior to inclusion.