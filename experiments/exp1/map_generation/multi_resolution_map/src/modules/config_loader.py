import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QuadtreeConfig:
    min_resolution: float
    max_depth: int
    treat_mixed_as_occupied_at_min_resolution: bool


@dataclass
class Nav2MapConfig:
    output_dir: Path | None
    mode: str
    negate: int
    occupied_thresh: float
    free_thresh: float


@dataclass
class UniformGridConfig:
    enable: bool
    resolution: float
    export_csv: bool
    export_visualization: bool
    coverage_targets: list[float]
    treat_mixed_as_occupied: bool
    nav2: Nav2MapConfig


@dataclass
class ResolutionSweepConfig:
    enable: bool
    min_resolutions: list[float]
    export_csv: bool
    export_plot: bool


@dataclass
class AnalysisConfig:
    resolution_sweep: ResolutionSweepConfig


@dataclass
class OutputConfig:
    output_dir: Path
    export_leaf_csv: bool
    export_summary_json: bool
    export_log_txt: bool


@dataclass
class VisualizationConfig:
    enable: bool
    formats: list[str]
    figure_size_inch: float
    dpi: int
    draw_obstacles: bool
    draw_quadtree_leaves: bool
    show_axes: bool
    occupied_color: str
    free_color: str
    boundary_color: str
    wall_color: str
    line_width: float


@dataclass
class Config:
    world_file: Path
    wall_prefixes: list
    obstacle_prefixes: list
    ignore_models: list
    quadtree: QuadtreeConfig
    uniform_grid: UniformGridConfig
    analysis: AnalysisConfig
    output: OutputConfig
    visualization: VisualizationConfig


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    base = Path(path).parent
    uniform_grid_data = data["uniform_grid"]
    nav2_data = uniform_grid_data.get("nav2", {})
    nav2_output_dir = nav2_data.get("output_dir")

    return Config(
        world_file=(base / data["input"]["world_file"]).resolve(),
        wall_prefixes=data["model_filter"]["wall_prefixes"],
        obstacle_prefixes=data["model_filter"]["obstacle_prefixes"],
        ignore_models=data["model_filter"]["ignore_models"],
        quadtree=QuadtreeConfig(
            min_resolution=float(data["quadtree"]["min_resolution"]),
            max_depth=int(data["quadtree"]["max_depth"]),
            treat_mixed_as_occupied_at_min_resolution=bool(
                data["quadtree"]["treat_mixed_as_occupied_at_min_resolution"]
            ),
        ),
        uniform_grid=UniformGridConfig(
            enable=bool(uniform_grid_data["enable"]),
            resolution=float(uniform_grid_data["resolution"]),
            export_csv=bool(uniform_grid_data.get("export_csv", False)),
            export_visualization=bool(uniform_grid_data.get("export_visualization", True)),
            coverage_targets=[
                float(value) for value in uniform_grid_data.get("coverage_targets", [])
            ],
            treat_mixed_as_occupied=bool(
                uniform_grid_data.get("treat_mixed_as_occupied", True)
            ),
            nav2=Nav2MapConfig(
                output_dir=(base / nav2_output_dir).resolve() if nav2_output_dir else None,
                mode=str(nav2_data.get("mode", "trinary")),
                negate=int(nav2_data.get("negate", 0)),
                occupied_thresh=float(nav2_data.get("occupied_thresh", 0.65)),
                free_thresh=float(nav2_data.get("free_thresh", 0.196)),
            ),
        ),
        analysis=AnalysisConfig(
            resolution_sweep=ResolutionSweepConfig(
                enable=bool(data["analysis"]["resolution_sweep"]["enable"]),
                min_resolutions=[
                    float(v) for v in data["analysis"]["resolution_sweep"]["min_resolutions"]
                ],
                export_csv=bool(data["analysis"]["resolution_sweep"]["export_csv"]),
                export_plot=bool(data["analysis"]["resolution_sweep"]["export_plot"]),
            )
        ),
        output=OutputConfig(
            output_dir=(base / data["output"]["output_dir"]).resolve(),
            export_leaf_csv=bool(data["output"]["export_leaf_csv"]),
            export_summary_json=bool(data["output"]["export_summary_json"]),
            export_log_txt=bool(data["output"]["export_log_txt"]),
        ),
        visualization=VisualizationConfig(
            enable=bool(data["visualization"]["enable"]),
            formats=list(data["visualization"]["formats"]),
            figure_size_inch=float(data["visualization"]["figure_size_inch"]),
            dpi=int(data["visualization"]["dpi"]),
            draw_obstacles=bool(data["visualization"]["draw_obstacles"]),
            draw_quadtree_leaves=bool(data["visualization"]["draw_quadtree_leaves"]),
            show_axes=bool(data["visualization"]["show_axes"]),
            occupied_color=str(data["visualization"]["occupied_color"]),
            free_color=str(data["visualization"]["free_color"]),
            boundary_color=str(data["visualization"]["boundary_color"]),
            wall_color=str(data["visualization"]["wall_color"]),
            line_width=float(data["visualization"]["line_width"]),
        ),
    )
