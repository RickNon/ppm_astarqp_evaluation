# Experiment 1: map construction

Experiment 1 evaluates PPM construction and a quadtree-based multi-resolution baseline in four fixed synthetic environments:

- `01_sparse`
- `02_dense`
- `03_maze_wide`
- `04_maze_narrow`

The implementation is split across `world_generation/`, `map_generation/`, and `map_analysis/`, but this file is the single entry point for all Experiment 1 procedures. 
Run every command below from the repository root.

## Contents

- [Experiment 1: map construction](#experiment-1-map-construction)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Reproduction workflow](#reproduction-workflow)
  - [World generation](#world-generation)
  - [PPM generation](#ppm-generation)
  - [Target-coverage PPM generation](#target-coverage-ppm-generation)
  - [Multi-resolution map and Nav2 grid generation](#multi-resolution-map-and-nav2-grid-generation)
  - [Coverage analysis](#coverage-analysis)

## Prerequisites

Use Python 3.11 or later. Initialize the PPM submodule and install the shared Experiment 1 dependencies:

```bash
git submodule update --init --recursive
python -m pip install -r experiments/exp1/map_generation/requirements.txt
```

Configurations are stored under `configs/worlds/`, `configs/ppm_sampling/`, and `configs/multi_resolution_map/`. Fixed worlds and occupancy grids are stored under `data/synthetic/`, and generated results are written under `results/exp1_map/`.

The public PPM result directories and their internal sampler values are:

| Result directory | `sampling_method` | Sampling behavior |
| --- | --- | --- |
| `independent_random` | `random` | Independent uniform-random candidates |
| `halton_random` | `halton` | Scrambled Halton candidates |

## Reproduction workflow

1. Use the committed worlds in `data/synthetic/worlds/`, or regenerate diagnostic copies with the [world generators](#world-generation).
2. Generate the independent-random and scrambled-Halton PPM coverage histories for each environment.
3. Reconstruct the independent-random PPMs at the configured target coverage thresholds.
4. Run the multi-resolution baseline and export its target-coverage Nav2 maps.
5. Generate the three-way coverage comparison figures.

The generated world files in `results/exp1_map/world_generation/` do not automatically replace the fixed inputs in `data/synthetic/worlds/`.

## World generation

`world_generation/random_world/` generates spatial density fields, samples Sobol candidates, selects obstacle centers, and exports random-obstacle SDF worlds. Generate the sparse and dense environments with:

```bash
python experiments/exp1/world_generation/random_world/src/main.py \
  --config configs/worlds/01_sparse_world_generation.yaml \
  --output-dir results/exp1_map/world_generation/01_sparse

python experiments/exp1/world_generation/random_world/src/main.py \
  --config configs/worlds/02_dense_world_generation.yaml \
  --output-dir results/exp1_map/world_generation/02_dense
```

`world_generation/maze_world/` generates a seeded perfect-maze topology, converts it to wall geometry, and exports an SDF world and preview.

Generate the two maze environments with:

```bash
python experiments/exp1/world_generation/maze_world/src/main.py \
  --config configs/worlds/03_maze_wide_world_generation.yaml

python experiments/exp1/world_generation/maze_world/src/main.py \
  --config configs/worlds/04_maze_narrow_world_generation.yaml
```

Only the four final experiment configurations are included. Their geometry and random-sequence parameters are fixed for reproducibility.

## PPM generation

The PPM pipeline samples candidate sensor positions, rejects candidates inside obstacles or existing PPM free cells, records covered free-space area, and exports the sensor and proximity-point data.

Run one independent-random condition with:

```bash
python experiments/exp1/map_generation/ppm/generate_random_ppm.py \
  --config configs/ppm_sampling/01_sparse_independent_random.yml
```

Run the corresponding scrambled-Halton condition with:

```bash
python experiments/exp1/map_generation/ppm/generate_random_ppm.py \
  --config configs/ppm_sampling/01_sparse_halton_random.yml
```

Equivalent sampling configs exist for all four environments. Each run writes its primary outputs to:

```text
results/exp1_map/<environment>/<independent_random|halton_random>/<run>/
├── ppm_area.csv
├── ppm_sensor.csv
├── ppm_prox.csv
└── ppm_area_plot.png
```

The published evaluation uses five repeated runs for both sampling methods: seeds 1 through 5 correspond to output folders `01` through `05`. The checked-in configs show seed 1 and folder `01`; 
change both values together for the other runs. 
The `--out-dir` override redirects outputs but does not change the seed.

The configs also produce PPM overlay plots at the sensor counts in `log_output_numbers` and at the final sampling state.

## Target-coverage PPM generation

After generating an independent-random area CSV, reconstruct the first PPM that reaches each configured coverage threshold:

```bash
python experiments/exp1/map_generation/ppm/generate_target_coverage_ppm.py \
  --config configs/ppm_sampling/target_coverage_01_sparse_independent.yml
```

There is one target-coverage config per environment, and all four use the independent-random run. 
No Halton target-coverage configs are included. The default targets are 0.5, 0.8, and 0.9, producing:

```text
results/exp1_map/<environment>/
├── coverage_50/
│   ├── ppm_sensor.csv
│   ├── ppm_prox.csv
│   ├── ppm_target_plot.png
│   └── ppm_target_plot.eps
├── coverage_80/
└── coverage_90/
```

Each threshold selects the first sensor count whose recorded coverage reaches or exceeds the target, so a `coverage_50` directory does not imply an achieved ratio of exactly 0.50. 
The world and all sensor and sampling settings in the target config, including range, field of view, method, scrambling, and seed, must match those used to create its source `ppm_area.csv`.

Use `--coverage-ratio`, `--area-csv`, or `--out-dir` to override the configured targets, source CSV, or output directory during validation.

## Multi-resolution map and Nav2 grid generation

The quadtree baseline parses an SDF world, classifies cells against its obstacles, records confirmed-free-area growth at multiple minimum resolutions, and exports target-coverage Nav2 maps for Experiment 2.

Run all four final configurations:

```bash
python experiments/exp1/map_generation/multi_resolution_map/src/main.py \
  --config configs/multi_resolution_map/01_sparse.yml

python experiments/exp1/map_generation/multi_resolution_map/src/main.py \
  --config configs/multi_resolution_map/02_dense.yml

python experiments/exp1/map_generation/multi_resolution_map/src/main.py \
  --config configs/multi_resolution_map/03_maze_wide.yml

python experiments/exp1/map_generation/multi_resolution_map/src/main.py \
  --config configs/multi_resolution_map/04_maze_narrow.yml
```

Paths inside these configs are resolved relative to the config file.
The published `uniform_grid` settings use:

| Setting | Published value | Effect |
| --- | --- | --- |
| `enable` | `true` | Records coverage snapshots and exports Nav2 maps |
| `resolution` | `0.05` | Decide resolution of grids |
| `coverage_targets` | `[0.5, 0.8, 0.9]` | Exports the first frontier reaching each threshold |
| `treat_mixed_as_occupied` | `true` | Conservatively marks unresolved cells as occupied |
| `nav2.mode` | `trinary` | Uses Nav2 trinary-map metadata |
| `nav2.negate` | `0` | Keeps the exported PGM value convention |
| `nav2.occupied_thresh` | `0.65` | Sets the Nav2 occupied threshold |
| `nav2.free_thresh` | `0.196` | Sets the Nav2 free threshold |

Coverage is confirmed `FREE` leaf area divided by the true free-space area. 
The builder records a frontier only after expanding all splittable cells at the current depth, 
then selects the first recorded frontier reaching each target. 
The resulting achieved coverage can therefore exceed its directory label.

The exporter treats unresolved `MIXED` leaves and the four outer walls as occupied, reverses PGM rows to match the Nav2 coordinate convention, and writes 0 for occupied pixels and 254 for free pixels. 
Nav2 artifacts are separate from the Experiment 1 analysis outputs:

```text
data/synthetic/occupancy_grids/<environment>/
├── coverage_50/
│   ├── <environment>_coverage_50_r_0p05.pgm
│   └── <environment>_coverage_50_r_0p05.yaml
├── coverage_80/
└── coverage_90/
```

Use `--nav2-output-dir <directory>` to redirect one environment's maps during validation and avoid overwriting checked-in artifacts. 
Setting `uniform_grid.enable` to `false` skips snapshot recording and Nav2 export. 

Each condition writes its quadtree analysis under:

```text
results/exp1_map/<environment>/multi_resolution_map/
```

Depending on the config switches, outputs include `leaf_cells.csv`, `quadtree_summary.json`, `log.txt`, quadtree PNG/EPS figures, and resolution-sweep CSVs and figures. 
The coverage comparison uses `free_area_vs_resolution_0p05.csv`.

## Coverage analysis

`map_analysis/plot_coverage_ratio.py` compares independent-random PPM, scrambled-Halton PPM, and the multi-resolution baseline. 
For each environment, it expects:
```text
results/exp1_map/<environment>/independent_random/<run>/ppm_area.csv
results/exp1_map/<environment>/halton_random/<run>/ppm_area.csv
results/exp1_map/<environment>/multi_resolution_map/free_area_vs_resolution_0p05.csv
```

Repeated PPM runs are averaged by sensor count. 
Generate the figures with:
```bash
python experiments/exp1/map_analysis/plot_coverage_ratio.py
```

The script writes PNG and EPS figures to `results/exp1_map/figures/`. 
It skips an environment if any required series is missing. 
Optional arguments are `--ppm-output-dir`, `--multi-output-dir`, `--output-dir`, and `--folders`.
