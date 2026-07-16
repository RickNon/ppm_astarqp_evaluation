# Experiment implementations

This directory contains the experiment-oriented implementation used by the paper.
The output of each experiment is recorded under `results/`.

## Layout

```text
experiments/
├── exp1/
│   ├── world_generation/
│   ├── map_generation/
│   │   ├── ppm/
│   │   └── multi_resolution_map/
│   └── map_analysis/
└── exp2/
    ├── ppm_planning/
    ├── path_analysis/
    └── ros2_ws/
```

The `exp2` planner implementations are shared by Experiments 2 and 3.
Experiment-specific behavior is selected by the config file, so an `exp3` source-code copy is unnecessary.

## PPM submodule

The PPM generator depends on the submodule at:

```text
experiments/exp1/map_generation/ppm/modules/Polygonal-Proximity-Map
```

Do not forget to initialize it from the repository root:

```bash
git submodule update --init --recursive
```

## Running experiments

- [Experiment 1](exp1/README.md)
- [Experiments 2 and 3](exp2/README.md)

Run repository commands from the repository root.
