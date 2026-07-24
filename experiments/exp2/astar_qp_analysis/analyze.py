from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[3]
PPM_PLANNING_DIR = REPO_ROOT / "experiments" / "exp2" / "ppm_planning"
if str(PPM_PLANNING_DIR) not in sys.path:
    sys.path.insert(0, str(PPM_PLANNING_DIR))

from modules.config_loader import load_config
from modules.graph_manager import GraphManager
from modules.planner import AstarQPBase, AstarQPWideSpace


ENVIRONMENTS = ("01_sparse", "02_dense")
FOCUS_COVERAGES = (80, 90)
METHODS = (
    "astar_qp_base",
    "astar_qp_wide_space",
    "astar_qp_full",
)
ENVIRONMENT_LABELS = {
    "01_sparse": "Sparse",
    "02_dense": "Dense",
}
METHOD_LABELS = {
    "astar_qp_base": "Base",
    "astar_qp_wide_space": "Wide-space",
    "astar_qp_full": "Full",
}
METHOD_COLORS = {
    "astar_qp_base": "#4C78A8",
    "astar_qp_wide_space": "#F2A541",
    "astar_qp_full": "#59A14F",
}
METRIC_SPECS = (
    {
        "key": "effective_openness",
        "ylabel": r"Effective openness [m$^2$]",
        "output_name": "effective_openness_grouped_bars_coverage_80_90",
        "series": (
            ("astar_qp_base", "base_effective_openness_m2"),
            ("astar_qp_wide_space", "wide_effective_openness_m2"),
        ),
    },
    {
        "key": "second_difference_energy",
        "ylabel": r"Integrated second-difference energy [m$^{-1}$]",
        "output_name": "second_difference_energy_grouped_bars_coverage_80_90",
        "series": (
            ("astar_qp_base", "base_second_difference_energy_1pm"),
            ("astar_qp_wide_space", "wide_second_difference_energy_1pm"),
            ("astar_qp_full", "full_second_difference_energy_1pm"),
        ),
    },
)

COMMON_SPACING_M = 0.10
WIDE_SPACE_EPSILON = 1.0e-2
ZERO_ENERGY_EPSILON = 1.0e-12
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "exp2_astar_qp_mechanism_effects"
DEFAULT_OUTPUT_FORMATS = ("png", "eps")
SUPPORTED_OUTPUT_FORMATS = frozenset(("png", "eps", "pdf", "svg"))
PLOT_FONT_SIZE = 14.0
RASTER_DPI = 220
EPS_FONT_TYPE = 3


@dataclass(frozen=True)
class Condition:
    environment: str
    coverage: int
    config_path: Path
    result_dir: Path


def parse_args() -> argparse.Namespace:
    """Parse calculation and plot settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate absolute A*QP mechanism metrics and plot their medians "
            "from the standard Experiment 2 planner results."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives calculated CSVs and figures.",
    )
    parser.add_argument(
        "--spacing-m",
        type=float,
        default=COMMON_SPACING_M,
        help="Arc-length interval for second-difference energy.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_OUTPUT_FORMATS),
        help="One or more output formats: png, eps, pdf, or svg.",
    )
    return parser.parse_args()


def normalized_formats(raw_formats: list[str]) -> tuple[str, ...]:
    """Validate and normalize requested figure formats."""
    formats = tuple(value.lower().lstrip(".") for value in raw_formats)
    if not formats:
        raise ValueError("At least one output format is required.")
    if len(formats) != len(set(formats)):
        raise ValueError("Output formats must not contain duplicates.")
    unsupported = sorted(set(formats) - SUPPORTED_OUTPUT_FORMATS)
    if unsupported:
        raise ValueError(
            f"Unsupported output formats: {', '.join(unsupported)}. "
            f"Choose from: {', '.join(sorted(SUPPORTED_OUTPUT_FORMATS))}."
        )
    return formats


def build_conditions() -> list[Condition]:
    """Resolve the four environment and coverage inputs."""
    conditions = []
    for environment in ENVIRONMENTS:
        for coverage in FOCUS_COVERAGES:
            conditions.append(
                Condition(
                    environment=environment,
                    coverage=coverage,
                    config_path=(
                        REPO_ROOT
                        / "configs"
                        / "planners"
                        / "ppm"
                        / "exp2_planner_sim"
                        / f"{environment}_coverage_{coverage}.yml"
                    ),
                    result_dir=(
                        REPO_ROOT
                        / "results"
                        / "exp2_planner_sim"
                        / environment
                        / f"coverage_{coverage}"
                    ),
                )
            )
    return conditions


def read_existing_results(condition: Condition) -> dict[str, pd.DataFrame]:
    """Load the original success records for all three methods."""
    return {
        method: pd.read_csv(condition.result_dir / method / "results.csv").set_index(
            "count"
        )
        for method in METHODS
    }


def common_success_counts(existing: dict[str, pd.DataFrame]) -> list[int]:
    """Return query IDs successfully solved by every method."""
    common: set[int] | None = None
    for method in METHODS:
        successes = set(
            existing[method].index[existing[method]["success"] == 1].astype(int)
        )
        common = successes if common is None else common & successes
    return sorted(common or [])


def load_result_path(
    condition: Condition,
    method: str,
    count: int,
) -> np.ndarray:
    """Load and validate a successful path from the planner result directory."""
    path = condition.result_dir / method / "paths" / f"{count}_path.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Path CSV is missing: {path}")

    frame = pd.read_csv(path)
    required_columns = {"count", "x", "y"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "Path CSV is missing required columns "
            f"{sorted(missing_columns)}: {path}"
        )
    if frame.empty:
        raise ValueError(f"Path CSV is empty: {path}")

    path_counts = set(frame["count"].astype(int))
    if path_counts != {count}:
        raise ValueError(f"Path CSV count does not match count={count}: {path}")

    path_xy = frame[["x", "y"]].to_numpy(dtype=float)
    if len(path_xy) < 2:
        raise ValueError(f"Path CSV must contain at least two points: {path}")
    if not np.isfinite(path_xy).all():
        raise ValueError(f"Path CSV contains non-finite coordinates: {path}")
    return path_xy


def remove_duplicate_points(path_xy: np.ndarray) -> np.ndarray:
    """Remove consecutive duplicate points before arc-length sampling."""
    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) == 0:
        return path_xy
    keep = np.ones(len(path_xy), dtype=bool)
    keep[1:] = (
        np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1) > 1.0e-12
    )
    return path_xy[keep]


def resample_without_short_endpoint(
    path_xy: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """Sample exact arc-length multiples and omit a shorter residual interval."""
    if spacing <= 0.0:
        raise ValueError("spacing must be positive.")
    path_xy = remove_duplicate_points(path_xy)
    if len(path_xy) < 2:
        return path_xy
    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    sample_s = np.arange(0.0, total_length + 1.0e-12, spacing, dtype=float)
    sample_s = sample_s[sample_s <= total_length + 1.0e-10]
    sample_x = np.interp(sample_s, cumulative, path_xy[:, 0])
    sample_y = np.interp(sample_s, cumulative, path_xy[:, 1])
    return np.column_stack((sample_x, sample_y))


def integrated_second_difference_energy(
    path_xy: np.ndarray,
    spacing: float,
) -> float:
    """Calculate the original common-scale second-difference energy."""
    sampled = resample_without_short_endpoint(path_xy, spacing)
    if len(sampled) < 3:
        return 0.0
    second_difference = np.diff(sampled, n=2, axis=0) / (spacing**2)
    energy = float(
        np.sum(np.linalg.norm(second_difference, axis=1) ** 2) * spacing
    )
    return 0.0 if abs(energy) <= ZERO_ENERGY_EPSILON else energy


def make_route_planners(
    condition: Condition,
) -> tuple[GraphManager, AstarQPBase, AstarQPWideSpace]:
    """Construct the planners needed to reproduce both graph routes."""
    config = load_config(condition.config_path)
    graph_manager = GraphManager()
    graph_manager.load_omnia_files(
        str(config.ppm_sensor_csv),
        str(config.ppm_prox_csv),
    )
    base = AstarQPBase(
        graph_manager=graph_manager,
        qp_solver=config.qp_solver,
    )
    wide = AstarQPWideSpace(
        graph_manager=graph_manager,
        qp_solver=config.qp_solver,
    )
    base.make_astar_graph()
    wide.make_astar_graph()
    wide._precompute_wide_space_normalizers()
    return graph_manager, base, wide


def graph_routes_for_query(
    graph_manager: GraphManager,
    base: AstarQPBase,
    wide: AstarQPWideSpace,
    start: tuple[float, float],
    goal: tuple[float, float],
) -> tuple[list[int], list[int]]:
    """Regenerate the base and wide-space graph routes."""
    start_node = graph_manager.get_nearest_initial_node(start)
    goal_node = graph_manager.get_nearest_initial_node(goal)
    if start_node is None or goal_node is None:
        raise RuntimeError("Nearest graph node was not found.")
    start_id = int(start_node[0])
    goal_id = int(goal_node[0])
    base._validate_query_points(start, goal, start_id, goal_id)
    if start_id == goal_id:
        return [start_id], [start_id]
    base_route = base.astar_search(start_id, goal_id)
    wide_route = wide.wide_space_astar_search(start_id, goal_id)
    if base_route is None or wide_route is None:
        raise RuntimeError("Graph route regeneration failed.")
    return (
        [int(value) for value in base_route],
        [int(value) for value in wide_route],
    )


def effective_openness(
    planner: AstarQPWideSpace,
    route: list[int],
) -> float:
    """Calculate the original distance-weighted harmonic openness."""
    if len(route) < 2:
        return float("nan")
    distances = []
    regularized_openness = []
    clearance_scale = float(planner._node_clearance_median)
    portal_scale = float(planner._shared_boundary_length_median)
    for source, target in zip(route[:-1], route[1:]):
        distance = float(planner.edge_cost_via_nearest(source, target))
        clearance = float(planner._node_clearance_mean(target))
        portal_width = float(planner._shared_boundary_length(source, target))
        distances.append(distance)
        regularized_openness.append(
            clearance * portal_width
            + WIDE_SPACE_EPSILON * clearance_scale * portal_scale
        )
    distance_values = np.asarray(distances, dtype=float)
    openness_values = np.maximum(
        np.asarray(regularized_openness, dtype=float),
        1.0e-12,
    )
    return float(
        np.sum(distance_values)
        / np.sum(distance_values / openness_values)
    )


def collect_absolute_metrics(
    conditions: list[Condition],
    spacing: float,
) -> pd.DataFrame:
    """Calculate openness and three-method energy for every common query."""
    rows = []
    for condition in conditions:
        print(
            f"Calculating {condition.environment} coverage_{condition.coverage}...",
            flush=True,
        )
        existing = read_existing_results(condition)
        counts = common_success_counts(existing)
        pairs = pd.read_csv(
            condition.result_dir / "start_goal_pairs.csv"
        ).set_index("count")
        graph_manager, base_planner, wide_planner = make_route_planners(condition)

        for count in counts:
            pair = pairs.loc[count]
            start = (float(pair["start_x"]), float(pair["start_y"]))
            goal = (float(pair["goal_x"]), float(pair["goal_y"]))
            base_route, wide_route = graph_routes_for_query(
                graph_manager,
                base_planner,
                wide_planner,
                start,
                goal,
            )

            energies = {}
            for method in METHODS:
                path_xy = load_result_path(
                    condition,
                    method,
                    count,
                )
                energies[method] = integrated_second_difference_energy(
                    path_xy,
                    spacing,
                )

            rows.append(
                {
                    "environment": condition.environment,
                    "coverage": condition.coverage,
                    "count": count,
                    "start_x_m": start[0],
                    "start_y_m": start[1],
                    "goal_x_m": goal[0],
                    "goal_y_m": goal[1],
                    "is_straight_query": (
                        len(base_route) == 1 and len(wide_route) == 1
                    ),
                    "base_effective_openness_m2": effective_openness(
                        wide_planner,
                        base_route,
                    ),
                    "wide_effective_openness_m2": effective_openness(
                        wide_planner,
                        wide_route,
                    ),
                    "base_second_difference_energy_1pm": energies[
                        "astar_qp_base"
                    ],
                    "wide_second_difference_energy_1pm": energies[
                        "astar_qp_wide_space"
                    ],
                    "full_second_difference_energy_1pm": energies[
                        "astar_qp_full"
                    ],
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["environment", "coverage", "count"]
    )


def validate_calculated_metrics(
    details: pd.DataFrame,
    conditions: list[Condition],
) -> None:
    """Validate the common-query population and calculated values."""
    expected_counts = {
        (condition.environment, condition.coverage): len(
            common_success_counts(read_existing_results(condition))
        )
        for condition in conditions
    }
    actual_counts = details.groupby(["environment", "coverage"]).size().to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError("Calculated rows do not preserve the common-success set.")

    active = details.loc[~details["is_straight_query"]]
    required_columns = [
        "base_effective_openness_m2",
        "wide_effective_openness_m2",
        "base_second_difference_energy_1pm",
        "wide_second_difference_energy_1pm",
        "full_second_difference_energy_1pm",
    ]
    values = active[required_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("A plotted metric contains a non-finite value.")
    energy_columns = [
        column for column in required_columns if column.endswith("_energy_1pm")
    ]
    if (active[energy_columns].to_numpy(dtype=float) < 0.0).any():
        raise RuntimeError("Second-difference energy must be non-negative.")


def build_summary(details: pd.DataFrame) -> pd.DataFrame:
    """Build plotted medians in condition and method order."""
    active = details.loc[~details["is_straight_query"]]
    rows: list[dict[str, object]] = []
    for metric_index, spec in enumerate(METRIC_SPECS):
        for environment_index, environment in enumerate(ENVIRONMENTS):
            for coverage_index, coverage in enumerate(FOCUS_COVERAGES):
                condition = active.loc[
                    (active["environment"] == environment)
                    & (active["coverage"] == coverage)
                ]
                expected_n = len(condition)
                group_index = (
                    environment_index * len(FOCUS_COVERAGES) + coverage_index
                )
                for method, value_column in spec["series"]:
                    values = condition[value_column].to_numpy(dtype=float)
                    if values.size != expected_n or not np.isfinite(values).all():
                        raise RuntimeError(
                            f"Invalid values for {environment}, coverage "
                            f"{coverage}, {value_column}."
                        )
                    rows.append(
                        {
                            "metric_index": metric_index,
                            "metric": spec["key"],
                            "group_index": group_index,
                            "environment": environment,
                            "environment_label": ENVIRONMENT_LABELS[environment],
                            "coverage": coverage,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "value_column": value_column,
                            "n": int(values.size),
                            "median": float(np.median(values)),
                            "q1": float(np.quantile(values, 0.25)),
                            "q3": float(np.quantile(values, 0.75)),
                        }
                    )
    summary = pd.DataFrame(rows)
    if len(summary) != 20:
        raise RuntimeError("Expected eight openness and twelve energy bars.")
    if not (
        (summary["q1"] <= summary["median"])
        & (summary["median"] <= summary["q3"])
    ).all():
        raise RuntimeError("Summary quartiles are not ordered.")
    return summary


def configure_plot_style() -> None:
    """Apply the repository publication style."""
    if PLOT_FONT_SIZE <= 0.0:
        raise ValueError("Plot font size must be positive.")
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": PLOT_FONT_SIZE,
            "axes.labelsize": PLOT_FONT_SIZE,
            "xtick.labelsize": 0.90 * PLOT_FONT_SIZE,
            "ytick.labelsize": 0.90 * PLOT_FONT_SIZE,
            "legend.fontsize": 0.90 * PLOT_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": EPS_FONT_TYPE,
        }
    )


def validate_eps(output_path: Path) -> None:
    """Require embedded Times New Roman Type 3 glyphs in EPS output."""
    content = output_path.read_text(encoding="latin-1")
    if "/FontType 3 def" not in content:
        raise RuntimeError(f"EPS does not contain Type 3 glyphs: {output_path}")
    if "/FontName /TimesNewRoman" not in content:
        raise RuntimeError(f"EPS does not use Times New Roman: {output_path}")
    external_markers = (
        "%%DocumentNeededResources: font",
        "%%IncludeResource: font",
    )
    if any(marker in content for marker in external_markers):
        raise RuntimeError(f"EPS depends on an external font: {output_path}")


def save_figure(
    figure: plt.Figure,
    output_stem: Path,
    output_formats: tuple[str, ...],
) -> list[Path]:
    """Save and validate every requested figure format."""
    output_paths = []
    for output_format in output_formats:
        output_path = output_stem.with_suffix(f".{output_format}")
        save_options: dict[str, object] = {
            "format": output_format,
            "bbox_inches": "tight",
        }
        if output_format == "png":
            save_options["dpi"] = RASTER_DPI
        figure.savefig(output_path, **save_options)
        if output_format == "eps":
            validate_eps(output_path)
        output_paths.append(output_path)
    return output_paths


def plot_metric(
    summary: pd.DataFrame,
    spec: dict[str, object],
    figure_dir: Path,
    output_formats: tuple[str, ...],
) -> list[Path]:
    """Plot one calculated mechanism metric."""
    metric_summary = summary.loc[summary["metric"] == spec["key"]]
    method_count = len(spec["series"])
    group_centers = np.arange(4, dtype=float)
    if method_count == 2:
        bar_width = 0.30
        offsets = np.asarray((-0.56, 0.56)) * bar_width
    elif method_count == 3:
        bar_width = 0.22
        offsets = np.asarray((-1.05, 0.0, 1.05)) * bar_width
    else:
        raise ValueError("Only two- or three-method figures are supported.")

    max_q3 = float(metric_summary["q3"].max())
    if max_q3 <= 0.0:
        raise ValueError(f"{spec['key']} quartiles must contain a positive value.")

    figure, axis = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
    for center, group_index in zip(group_centers, range(4)):
        group = metric_summary.loc[
            metric_summary["group_index"] == group_index
        ]
        if len(group) != method_count:
            raise ValueError(
                f"{spec['key']} must contain {method_count} bars per condition."
            )
        methods = group["method"].tolist()
        medians = group["median"].to_numpy(dtype=float)
        q1 = group["q1"].to_numpy(dtype=float)
        q3 = group["q3"].to_numpy(dtype=float)
        positions = center + offsets
        axis.bar(
            positions,
            medians,
            width=bar_width,
            color=[METHOD_COLORS[method] for method in methods],
            edgecolor="#333333",
            linewidth=0.55,
            yerr=np.vstack((medians - q1, q3 - medians)),
            error_kw={
                "ecolor": "#333333",
                "elinewidth": 1.2,
                "capsize": 4.0,
                "capthick": 1.2,
            },
            zorder=3,
        )
        # for position, median in zip(positions, medians):
        #     axis.annotate(
        #         f"{median:.2f}",
        #         (position, median),
        #         xytext=(0, 5),
        #         textcoords="offset points",
        #         ha="center",
        #         va="bottom",
        #         fontsize=0.72 * PLOT_FONT_SIZE,
        #         fontweight="bold",
        #         color="black",
        #         bbox={
        #             "facecolor": "white",
        #             "edgecolor": "none",
        #             "pad": 0.25,
        #         },
        #         zorder=4,
        #     )

    labels = []
    for group_index in range(4):
        first = metric_summary.loc[
            metric_summary["group_index"] == group_index
        ].iloc[0]
        labels.append(
            f"{first['environment_label']} {int(first['coverage'])}%\n"
            f"(n={int(first['n'])})"
        )
    axis.set_xticks(group_centers, labels)
    axis.set_xlim(group_centers[0] - 0.55, group_centers[-1] + 0.55)
    axis.set_ylim(0.0, 1.08 * max_q3)
    axis.set_ylabel(spec["ylabel"])
    axis.grid(True, axis="y", linestyle=":", color="#C9CDD2", linewidth=0.8)
    axis.set_axisbelow(True)

    legend_methods = [method for method, _ in spec["series"]]
    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="#333333",
            linewidth=0.55,
            label=METHOD_LABELS[method],
        )
        for method in legend_methods
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=method_count,
        frameon=False,
    )

    output_paths = save_figure(
        figure,
        figure_dir / spec["output_name"],
        output_formats,
    )
    plt.close(figure)
    return output_paths


def main() -> None:
    """Calculate metrics, write summaries, and generate both figures."""
    args = parse_args()
    if args.spacing_m <= 0.0:
        raise ValueError("spacing-m must be positive.")
    output_formats = normalized_formats(args.formats)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_conditions()
    details = collect_absolute_metrics(
        conditions,
        args.spacing_m,
    )
    validate_calculated_metrics(details, conditions)
    summary = build_summary(details)

    details_path = output_dir / "absolute_mechanism_pair_metrics.csv"
    summary_path = output_dir / "absolute_mechanism_metrics_summary.csv"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    configure_plot_style()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = []
    for spec in METRIC_SPECS:
        figure_paths.extend(
            plot_metric(summary, spec, figure_dir, output_formats)
        )

    print(f"Pair metrics: {details_path}")
    print(f"Summary: {summary_path}")
    for figure_path in figure_paths:
        print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
