from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PPM_PLANNING_DIR = REPO_ROOT / "experiments" / "exp2" / "ppm_planning"
if str(PPM_PLANNING_DIR) not in sys.path:
    sys.path.insert(0, str(PPM_PLANNING_DIR))

from modules.config_loader import load_config
from modules.graph_manager import GraphManager
from modules.planner import AstarQPBase, AstarQPWideSpace
from plot_results import (
    EPS_FONT_TYPE,
    PLOT_FONT_SIZE,
    PLOT_OUTPUT_FORMATS,
    generate_mechanism_plots,
)


ENVIRONMENTS = ("01_sparse", "02_dense")
COVERAGES = (80, 90)
METHODS = ("astar_qp_base", "astar_qp_wide_space", "astar_qp_full")
COMMON_SPACING_M = 0.10
WIDE_SPACE_EPSILON = 1.0e-2
RATIO_DENOMINATOR_EPS = 1.0e-12
NUMERICAL_EQUAL_ATOL = 1.0e-12
NUMERICAL_EQUAL_RTOL = 1.0e-9
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "exp2_astar_qp_mechanism_effects"


@dataclass(frozen=True)
class Condition:
    environment: str
    coverage: int
    config_path: Path
    result_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate only the B/C A*QP mechanism comparisons."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Independent output directory for the B/C analysis.",
    )
    parser.add_argument(
        "--spacing-m",
        type=float,
        default=COMMON_SPACING_M,
        help="Common arc-length interval used for second-difference energy.",
    )
    return parser.parse_args()


def build_conditions() -> list[Condition]:
    conditions = []
    for environment in ENVIRONMENTS:
        for coverage in COVERAGES:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_existing_results(condition: Condition) -> dict[str, pd.DataFrame]:
    return {
        method: pd.read_csv(condition.result_dir / method / "results.csv").set_index(
            "count"
        )
        for method in METHODS
    }


def common_success_counts(existing: dict[str, pd.DataFrame]) -> list[int]:
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
    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) == 0:
        return path_xy
    keep = np.ones(len(path_xy), dtype=bool)
    keep[1:] = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1) > 1.0e-12
    return path_xy[keep]


def resample_without_short_endpoint(path_xy: np.ndarray, spacing: float) -> np.ndarray:
    """Sample exact arc-length multiples and omit a shorter residual interval."""
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
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
    sampled = resample_without_short_endpoint(path_xy, spacing)
    if len(sampled) < 3:
        return 0.0
    second_difference = np.diff(sampled, n=2, axis=0) / (spacing**2)
    energy = float(np.sum(np.linalg.norm(second_difference, axis=1) ** 2) * spacing)
    return 0.0 if abs(energy) <= RATIO_DENOMINATOR_EPS else energy


def make_route_planners(
    condition: Condition,
) -> tuple[GraphManager, AstarQPBase, AstarQPWideSpace]:
    config = load_config(condition.config_path)
    graph_manager = GraphManager()
    graph_manager.load_omnia_files(
        str(config.ppm_sensor_csv),
        str(config.ppm_prox_csv),
    )
    base = AstarQPBase(graph_manager=graph_manager, qp_solver=config.qp_solver)
    wide = AstarQPWideSpace(graph_manager=graph_manager, qp_solver=config.qp_solver)
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
        raise RuntimeError("Graph route regeneration failed for a common-success query.")
    return [int(value) for value in base_route], [
        int(value) for value in wide_route
    ]


def effective_openness(planner: AstarQPWideSpace, route: list[int]) -> float:
    """Compute the distance-weighted harmonic openness in square metres."""
    if len(route) < 2:
        return float("nan")
    distances = []
    regularized_openness = []
    c_scale = float(planner._node_clearance_median)
    l_scale = float(planner._shared_boundary_length_median)
    for source, target in zip(route[:-1], route[1:]):
        distance = float(planner.edge_cost_via_nearest(source, target))
        clearance = float(planner._node_clearance_mean(target))
        portal_width = float(planner._shared_boundary_length(source, target))
        distances.append(distance)
        regularized_openness.append(
            clearance * portal_width + WIDE_SPACE_EPSILON * c_scale * l_scale
        )
    distance_values = np.asarray(distances, dtype=float)
    openness_values = np.maximum(np.asarray(regularized_openness, dtype=float), 1.0e-12)
    return float(np.sum(distance_values) / np.sum(distance_values / openness_values))


def stable_difference(candidate: float, baseline: float) -> float:
    """Return zero when two values differ only by floating-point noise."""
    if not np.isfinite(baseline) or not np.isfinite(candidate):
        return float("nan")
    difference = candidate - baseline
    tolerance = NUMERICAL_EQUAL_ATOL + NUMERICAL_EQUAL_RTOL * max(
        abs(candidate), abs(baseline)
    )
    return 0.0 if abs(difference) <= tolerance else difference


def relative_change(candidate: float, baseline: float) -> float:
    if not np.isfinite(baseline) or not np.isfinite(candidate):
        return float("nan")
    if abs(baseline) <= RATIO_DENOMINATOR_EPS:
        return float("nan")
    return stable_difference(candidate, baseline) / baseline


def collect_pairwise_metrics(
    conditions: Iterable[Condition],
    spacing: float,
) -> pd.DataFrame:
    rows = []
    for condition in conditions:
        print(
            f"Processing {condition.environment} coverage_{condition.coverage}...",
            flush=True,
        )
        existing = read_existing_results(condition)
        counts = common_success_counts(existing)
        pairs = pd.read_csv(condition.result_dir / "start_goal_pairs.csv").set_index(
            "count"
        )
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

            smoothing_paths = {
                method: load_result_path(
                    condition,
                    method,
                    count,
                )
                for method in ("astar_qp_wide_space", "astar_qp_full")
            }
            wide_energy = integrated_second_difference_energy(
                smoothing_paths["astar_qp_wide_space"],
                spacing,
            )
            full_energy = integrated_second_difference_energy(
                smoothing_paths["astar_qp_full"],
                spacing,
            )
            base_openness = effective_openness(wide_planner, base_route)
            wide_openness = effective_openness(wide_planner, wide_route)
            is_straight_query = len(base_route) == 1 and len(wide_route) == 1

            rows.append(
                {
                    "environment": condition.environment,
                    "coverage": condition.coverage,
                    "count": count,
                    "start_x_m": start[0],
                    "start_y_m": start[1],
                    "goal_x_m": goal[0],
                    "goal_y_m": goal[1],
                    "is_straight_query": is_straight_query,
                    "base_effective_openness_m2": base_openness,
                    "wide_effective_openness_m2": wide_openness,
                    "relative_effective_openness_change": relative_change(
                        wide_openness,
                        base_openness,
                    ),
                    "wide_second_difference_energy_1pm": wide_energy,
                    "full_second_difference_energy_1pm": full_energy,
                    "relative_second_difference_energy_change": relative_change(
                        full_energy,
                        wide_energy,
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["environment", "coverage", "count"])


def distribution_summary(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
        }
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
    }


def build_base_wide_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (environment, coverage), group in details.groupby(
        ["environment", "coverage"],
        sort=True,
    ):
        active = group.loc[~group["is_straight_query"]]
        relative = distribution_summary(active["relative_effective_openness_change"])
        rows.append(
            {
                "environment": environment,
                "coverage": int(coverage),
                "n_common_success": len(group),
                "n_mechanism_active": len(active),
                "n_straight_excluded": len(group) - len(active),
                "n_openness_ratio_valid": relative["n"],
                "relative_openness_change_median": relative["median"],
                "relative_openness_change_q1": relative["q1"],
                "relative_openness_change_q3": relative["q3"],
                "relative_openness_change_median_pct": 100.0 * relative["median"],
                "relative_openness_change_q1_pct": 100.0 * relative["q1"],
                "relative_openness_change_q3_pct": 100.0 * relative["q3"],
                "n_openness_increase": int(
                    (active["relative_effective_openness_change"] > 0.0).sum()
                ),
                "n_openness_tie": int(
                    (active["relative_effective_openness_change"] == 0.0).sum()
                ),
                "n_openness_decrease": int(
                    (active["relative_effective_openness_change"] < 0.0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_wide_full_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (environment, coverage), group in details.groupby(
        ["environment", "coverage"],
        sort=True,
    ):
        active = group.loc[~group["is_straight_query"]]
        relative = distribution_summary(
            active["relative_second_difference_energy_change"]
        )
        rows.append(
            {
                "environment": environment,
                "coverage": int(coverage),
                "n_common_success": len(group),
                "n_mechanism_active": len(active),
                "n_straight_excluded": len(group) - len(active),
                "n_energy_ratio_valid": relative["n"],
                "relative_energy_change_median": relative["median"],
                "relative_energy_change_q1": relative["q1"],
                "relative_energy_change_q3": relative["q3"],
                "relative_energy_change_median_pct": 100.0 * relative["median"],
                "relative_energy_change_q1_pct": 100.0 * relative["q1"],
                "relative_energy_change_q3_pct": 100.0 * relative["q3"],
                "n_energy_decrease": int(
                    (active["relative_second_difference_energy_change"] < 0.0).sum()
                ),
                "n_energy_tie": int(
                    (active["relative_second_difference_energy_change"] == 0.0).sum()
                ),
                "n_energy_increase": int(
                    (active["relative_second_difference_energy_change"] > 0.0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(
    details: pd.DataFrame,
    base_wide: pd.DataFrame,
    wide_full: pd.DataFrame,
    conditions: Iterable[Condition],
) -> None:
    expected_counts = {}
    for condition in conditions:
        expected_counts[(condition.environment, condition.coverage)] = len(
            common_success_counts(read_existing_results(condition))
        )
    actual_counts = details.groupby(["environment", "coverage"]).size().to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError("The B/C analysis did not preserve the common-success set.")
    if len(base_wide) != len(expected_counts) or len(wide_full) != len(
        expected_counts
    ):
        raise RuntimeError("One or more B/C summary rows are missing.")
    if not base_wide[["environment", "coverage"]].equals(
        wide_full[["environment", "coverage"]]
    ):
        raise RuntimeError("The two B/C summaries are not aligned by condition.")

    openness_direction_total = base_wide[
        ["n_openness_increase", "n_openness_tie", "n_openness_decrease"]
    ].sum(axis=1)
    if not openness_direction_total.equals(base_wide["n_openness_ratio_valid"]):
        raise RuntimeError("Effective-openness direction counts are incomplete.")
    energy_direction_total = wide_full[
        ["n_energy_decrease", "n_energy_tie", "n_energy_increase"]
    ].sum(axis=1)
    if not energy_direction_total.equals(wide_full["n_energy_ratio_valid"]):
        raise RuntimeError("Energy direction counts are incomplete.")

    straight = details["is_straight_query"].astype(bool)
    openness_valid = details["relative_effective_openness_change"].notna()
    if not openness_valid.equals(~straight):
        raise RuntimeError("Effective openness must be defined for non-straight pairs.")
    expected_openness = np.asarray(
        [
            relative_change(candidate, baseline)
            for candidate, baseline in zip(
                details.loc[openness_valid, "wide_effective_openness_m2"],
                details.loc[openness_valid, "base_effective_openness_m2"],
            )
        ]
    )
    if not np.allclose(
        details.loc[openness_valid, "relative_effective_openness_change"],
        expected_openness,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("Effective-openness ratios failed formula validation.")

    energy_valid = details["relative_second_difference_energy_change"].notna()
    if not energy_valid.equals(~straight):
        raise RuntimeError("Energy ratios must use the same non-straight pair set.")
    expected_energy = np.asarray(
        [
            relative_change(candidate, baseline)
            for candidate, baseline in zip(
                details.loc[energy_valid, "full_second_difference_energy_1pm"],
                details.loc[energy_valid, "wide_second_difference_energy_1pm"],
            )
        ]
    )
    if not np.allclose(
        details.loc[energy_valid, "relative_second_difference_energy_change"],
        expected_energy,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("Second-difference-energy ratios failed formula validation.")


def write_manifest(
    output_dir: Path,
    conditions: Iterable[Condition],
    spacing: float,
) -> None:
    input_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("plot_results.py"),
        Path(__file__).resolve().with_name("requirements.txt"),
        PPM_PLANNING_DIR / "modules" / "config_loader.py",
        PPM_PLANNING_DIR / "modules" / "data_converter.py",
        PPM_PLANNING_DIR / "modules" / "planner.py",
        PPM_PLANNING_DIR / "modules" / "graph_manager.py",
    ]
    for condition in conditions:
        config = load_config(condition.config_path)
        input_paths.extend(
            [
                condition.config_path,
                condition.result_dir / "start_goal_pairs.csv",
                config.ppm_sensor_csv,
                config.ppm_prox_csv,
                *[
                    condition.result_dir / method / "results.csv"
                    for method in METHODS
                ],
            ]
        )
        existing = read_existing_results(condition)
        for count in common_success_counts(existing):
            input_paths.extend(
                condition.result_dir / method / "paths" / f"{count}_path.csv"
                for method in ("astar_qp_wide_space", "astar_qp_full")
            )
    unique_inputs = sorted(set(path.resolve() for path in input_paths))
    inputs = [
        {
            "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in unique_inputs
    ]

    output_files = []
    for path in sorted(
        candidate for candidate in output_dir.rglob("*") if candidate.is_file()
    ):
        if path.name == "run_manifest.json":
            continue
        output_files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    package_versions = {}
    for package in (
        "cvxpy",
        "matplotlib",
        "networkx",
        "numpy",
        "osqp",
        "pandas",
        "PyYAML",
        "scipy",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": package_versions,
        "coverages": list(COVERAGES),
        "spacing_m": spacing,
        "plot_style": {
            "font_family": "Times New Roman",
            "base_font_size_pt": PLOT_FONT_SIZE,
            "output_formats": list(PLOT_OUTPUT_FORMATS),
            "eps_font_type": EPS_FONT_TYPE,
            "eps_text_representation": "embedded_type3_glyphs",
        },
        "ratio_denominator_epsilon": RATIO_DENOMINATOR_EPS,
        "numerical_equality": {
            "absolute_tolerance": NUMERICAL_EQUAL_ATOL,
            "relative_tolerance": NUMERICAL_EQUAL_RTOL,
        },
        "relative_change_definitions": {
            "effective_openness": "(wide - base) / base",
            "second_difference_energy": "(full - wide) / wide",
        },
        "input_files": inputs,
        "output_files": output_files,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.spacing_m <= 0.0:
        raise ValueError("spacing-m must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_conditions()
    details = collect_pairwise_metrics(
        conditions,
        args.spacing_m,
    )
    base_wide = build_base_wide_summary(details)
    wide_full = build_wide_full_summary(details)
    validate_outputs(details, base_wide, wide_full, conditions)

    details.to_csv(output_dir / "pairwise_mechanism_metrics.csv", index=False)
    base_wide.to_csv(output_dir / "base_vs_wide_summary.csv", index=False)
    wide_full.to_csv(output_dir / "wide_vs_full_summary.csv", index=False)
    generate_mechanism_plots(base_wide, wide_full, output_dir)
    write_manifest(
        output_dir,
        conditions,
        args.spacing_m,
    )
    print(f"B/C mechanism analysis completed: {output_dir}")


if __name__ == "__main__":
    main()
