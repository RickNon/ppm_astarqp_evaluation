# Experiments 2 and 3: Path planning evaluation

This directory contains the planner implementations shared by the simulated Experiment 2 and the real-environment Experiment 3:
- `ppm_planning/`: A* QP (base), A* QP (wide-space), A* QP (full), and BIT* on PPM.
- `ros2_ws/`: Nav2 Hybrid-A* and Theta* benchmark workspace.
- `path_analysis/`: aggregation and overview figures for Experiment 2.

Experiment-specific behavior is selected by config files, so there is no duplicated `experiments/exp3/` source tree. 
Run all Python and ROS benchmark commands from the repository root unless a section explicitly changes directory for the ROS workspace build.

## Contents

- [Experiments 2 and 3: Path planning evaluation](#experiments-2-and-3-path-planning-evaluation)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
    - [Python planners and analysis](#python-planners-and-analysis)
    - [ROS 2 planners](#ros-2-planners)
  - [Planner methods](#planner-methods)
  - [Workflow (Experiment 2)](#workflow-experiment-2)
  - [PPM planning](#ppm-planning)
  - [ROS 2 and Nav2 planning](#ros-2-and-nav2-planning)
    - [Build the workspace](#build-the-workspace)
    - [Run one condition](#run-one-condition)
  - [Experiment 2 path analysis](#experiment-2-path-analysis)
  - [Experiment 3](#experiment-3)
    - [PPM planners](#ppm-planners)
    - [Nav2 planners](#nav2-planners)
  - [Timing policy](#timing-policy)

## Prerequisites

### Python planners and analysis

Use Python 3.11 or later. 
Install the PPM planner dependencies with:
```bash
python -m pip install -r experiments/exp2/ppm_planning/requirements.txt
python -m pip install ompl
```

OMPL is used only by `bitstar_ppm`, but every checked-in PPM planner config includes that method. 

Install the additional plotting dependency before running the Experiment 2 analysis:
```bash
python -m pip install -r experiments/exp2/path_analysis/requirements.txt
```

### ROS 2 planners

Hybrid-A* and Theta* require a separate ROS 2 and Nav2 environment. 
The tested environment is Ubuntu 22.04, ROS 2 Humble, and Navigation2 1.1.20.

## Planner methods

| Config method | Reported method | Implementation |
| --- | --- | --- |
| `astar_qp_base` | A*QP (base) | Distance-based A* followed by the boundary QP |
| `astar_qp_wide_space` | A*QP (wide-space) | Wide-space-aware A* followed by the base boundary QP |
| `astar_qp_full` | A*QP (full) | Wide-space-aware A* followed by the smoothing QP |
| `bitstar_ppm` | BIT* | OMPL BIT* constrained to PPM free space |
| `nav2_hybrid_astar` | Hybrid-A* | Nav2 Smac Hybrid-A* on the occupancy grid |
| `nav2_theta_star` | Theta* | Nav2 Theta* on the occupancy grid |

## Workflow (Experiment 2)

Experiment 2 has 12 conditions: 
four environments (`01_sparse`, `02_dense`, `03_maze_wide`, and `04_maze_narrow`) at target coverage levels 50, 80, and 90.

For each condition:
1. Ensure that the frozen target-coverage PPM, Nav2 occupancy grid, and `data/synthetic/worlds/<environment>.world` exist. Regenerate missing map inputs with the [Experiment 1 target-coverage PPM](../exp1/README.md#target-coverage-ppm-generation) and [multi-resolution map](../exp1/README.md#multi-resolution-map-and-nav2-grid-generation) procedures.
2. Run the condition's PPM planner config. It generates one deterministic `start_goal_pairs.csv` shared by all six planners.
3. Run Hybrid-A* and Theta* with their matching condition configs; both read that same CSV.
4. After all conditions are complete, generate the Experiment 2 overview figures.

The Experiment 2 configs are located under:
```text
configs/planners/
├── ppm/exp2_planner_sim/
└── ros2/exp2_planner_sim/
```

## PPM planning

Run one simulated condition from the repository root:
```bash
python experiments/exp2/ppm_planning/main.py \
  --config configs/planners/ppm/exp2_planner_sim/01_sparse_coverage_50.yml
```

Repeat with the matching PPM config for all 12 environment/coverage conditions. 
Each config runs all four PPM methods and writes:
```text
results/exp2_planner_sim/<environment>/coverage_<percent>/
├── start_goal_pairs.csv
├── astar_qp_base/
│   ├── results.csv
│   └── paths/<count>_path.csv
├── astar_qp_wide_space/
│   ├── results.csv
│   └── paths/<count>_path.csv
├── astar_qp_full/
│   ├── results.csv
│   └── paths/<count>_path.csv
└── bitstar_ppm/
    ├── results.csv
    └──paths/<count>_path.csv
```

Path CSVs are written only for successful trials. When plotting is enabled, cross-method trial figures are written under the config's `plot.output_dir`.

## ROS 2 and Nav2 planning

### Build the workspace

Install a compatible ROS 2 distribution with Nav2, then build the benchmark package:

```bash
source /opt/ros/humble/setup.bash
cd experiments/exp2/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

`src/smac_benchmark` contains the planner-only launch file, fixed costmap and inflation settings, planner parameter files, single-query and batch runners, and PPM-compatible result conversion. 

### Run one condition

After building, open two terminals at the repository root and source both ROS 2 and this workspace in each terminal.

Terminal 1 launches the map and planner servers:
```bash
source /opt/ros/humble/setup.bash
source experiments/exp2/ros2_ws/install/setup.bash
ros2 launch smac_benchmark smac_planner_only.launch.py \
  config:="path/to/configs/planners/ros2/exp2_planner_sim/01_sparse_coverage_50_hybrid_astar.yml"
```

Terminal 2 runs all shared start-goal pairs:
```bash
source /opt/ros/humble/setup.bash
source experiments/exp2/ros2_ws/install/setup.bash
ros2 run smac_benchmark batch_benchmark \
  --config "path/to/configs/planners/ros2/exp2_planner_sim/01_sparse_coverage_50_hybrid_astar.yml"
```

Both terminals must use the same config. 
Repeat the two-terminal run with the matching Theta* config, then repeat both planners for all 12 conditions.

Each batch writes under the config's `io.output_dir`:
```text
<condition>/
├── start_goal_pairs.csv
└── <nav2_hybrid_astar|nav2_theta_star>/
    ├── config_used.yml
    ├── start_goal_pairs.csv
    ├── run_summary.csv
    ├── results.csv
    ├── paths/
    └── raw/
```

`results.csv` uses the same schema as the PPM planners. 
Individual raw PNGs are also produced when `save_individual_plots` is enabled.

## Experiment 2 path analysis

`path_analysis/plot_path_analysis.py` expects this layout for each condition:
```text
results/exp2_planner_sim/<environment>/coverage_<percent>/<method>/results.csv
```

All six method directories must be present with equal trial counts and the columns `success`, `time`, `mean_abs_curvature`, and `mean_clearance`. 
The legacy aliases `hybrid_astar` and `theta_star` are also accepted for the two Nav2 methods.

Generate the overview figures with:
```bash
python experiments/exp2/path_analysis/plot_path_analysis.py
```

The default output directory is `results/exp2_planner_sim/figures/`. 
The script writes success-rate, planning-time, mean-absolute-curvature, and mean-clearance overviews in PNG and EPS formats. 


## Experiment 3

Experiment 3 reuses the same executables with real-environment inputs under `data/real/`, configs under `configs/planners/*/exp3_planner_real/`, and outputs under `results/exp3_planner_real/`.

### PPM planners

Run the three real PPM variants:
```bash
python experiments/exp2/ppm_planning/main.py \
  --config configs/planners/ppm/exp3_planner_real/ku_outdoor_1of1.yml

python experiments/exp2/ppm_planning/main.py \
  --config configs/planners/ppm/exp3_planner_real/ku_outdoor_1of3.yml

python experiments/exp2/ppm_planning/main.py \
  --config configs/planners/ppm/exp3_planner_real/ku_outdoor_adjusted.yml
```

All three configs read `data/real/start_goal_pairs/ku_outdoor.csv`. 
Their `io.output_dir` values are `results/exp3_planner_real/ppm_1of1`, `ppm_1of3`, and `ppm_adjusted`; 
each method writes below that variant directory like experiment 2. 
Figures are written separately to each config's `plot.output_dir`.

Since SDF file for real-world planning is unavailable,
these configs do not require an SDF `world_file`. 
Planning runs normally, but obstacle-based clearance columns remain empty. 
Do not interpret those empty values as zero.

### Nav2 planners

Same as experiment 2.
Use the two-terminal launch and batch procedure with:

```text
configs/planners/ros2/exp3_planner_real/ku_outdoor_hybrid_astar.yml
configs/planners/ros2/exp3_planner_real/ku_outdoor_theta_star.yml
```

Both configs use `data/real/occupancy_grids/ku_outdoor.yaml` and `data/real/start_goal_pairs/ku_outdoor.csv`. 
Hybrid-A* uses `planner_server_hybrid_astar.yaml`; Theta* uses `planner_server_theta_star.yaml`. 
Their outputs remain isolated by method below `results/exp3_planner_real/`.

## Timing policy

PPM planner time excludes PPM CSV loading, base-graph loading, validity-checker construction, metric calculation, file output, and plotting.
A* QP timing includes polygon-graph construction, A* search, and QP. 
BIT* timing includes OMPL setup, solve, solution simplification, and final PPM segment validation.

For Nav2, `planning_time_sec` measures the selected `/compute_path_to_pose` request from goal submission through result receipt. 
Nav2 startup and costmap initialization are excluded. 
Hybrid-A* may try multiple yaw candidates; the stored value is the selected successful request time, not the wall time of the complete yaw search.
