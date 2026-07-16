# Results

Recorded outputs are separated by experiment.

## Layout

- `exp1_map/<environment>/independent_random/`: PPM construction with independent-random sampling.
- `exp1_map/<environment>/halton_random/`: PPM construction with scrambled-Halton sampling.
- `exp1_map/<environment>/multi_resolution_map/`: multi-resolution analysis tables, logs, and figures.
- `exp1_map/<environment>/coverage_<percent>/`: frozen target-coverage PPM inputs used by Experiment 2.
- `exp1_map/figures/`: Experiment 1 comparison figures.
- `exp2_planner_sim/<environment>/coverage_<percent>/`: shared start-goal pairs and six planner result directories.
- `exp2_planner_sim/figures/`: Experiment 2 comparison figures.
- `exp3_planner_real/`: Experiment 3 outputs, separated by PPM input variant and Nav2 method.

PPM planner method directories are `astar_qp_base`, `astar_qp_wide_space`, `astar_qp_full`, and `bitstar_ppm`. 
Nav2 method directories are `nav2_hybrid_astar` and `nav2_theta_star`.

Nav2 PGM/YAML inputs for synthetic environments are stored under `data/synthetic/occupancy_grids/`.
