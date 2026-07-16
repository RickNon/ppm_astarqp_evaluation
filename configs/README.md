# Experiment configurations

This directory contains the configurations used for the reported experiments.

## Layout

- `worlds/`: the four synthetic-world generation configurations.
- `ppm_sampling/`: PPM construction with independent-random and scrambled-Halton sampled observation points, and target-coverage reconstruction.
- `multi_resolution_map/`: one multi-resolution grid (quadtree) and target-coverage Nav2 map configuration for each synthetic environment.
- `planners/ppm/exp2_planner_sim/`: 12 simulated PPM-planner conditions.
- `planners/ppm/exp3_planner_real/`: three real-environment PPM variants.
- `planners/ros2/exp2_planner_sim/`: 24 simulated Nav2 conditions: Hybrid-A* and Theta* for four environments and three coverages.
- `planners/ros2/exp3_planner_real/`: the two real-environment experimental sets for Nav2.

## Naming convention in configs

Public result directories use `independent_random` and `halton_random`. 
The values passed to the sampler remain `random` and `halton`.

The synthetic environment names are `01_sparse`, `02_dense`, `03_maze_wide`, and `04_maze_narrow`.

Experiment 2 PPM configs consume the frozen target-coverage PPM files under `results/exp1_map/<environment>/coverage_<percent>/`. Experiment 1 target-coverage reconstruction configs retain their sampling-method-specific output under `independent_random/` or `halton_random/`; if the frozen Experiment 2 inputs are regenerated, record or automate the copy into the Experiment 2 input location.

Each Experiment 2 Nav2 config reuses the condition-level `start_goal_pairs.csv` produced by the PPM planner run. 
Experiment 3 uses the fixed shared file `data/real/start_goal_pairs/ku_outdoor.csv`.
