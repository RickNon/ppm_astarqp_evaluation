from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "results" / "exp2_planner_sim"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "figures"
OUTPUT_FORMATS = ("png", "eps")
TARGET_ENVIRONMENT_FOLDERS = ("01_sparse", "02_dense", "03_maze_wide", "04_maze_narrow")
RESULTS_FILENAME = "results.csv"

SUCCESS_COLUMN = "success"
REQUIRED_COLUMNS = {
    SUCCESS_COLUMN,
    "time",
    "mean_abs_curvature",
    "mean_clearance",
}
FIG_DPI = 260
FIG_FACE_COLOR = "#f8f8f8"
PREFERRED_FONT_FAMILIES = ("Times New Roman", "Times", "DejaVu Serif")
BASE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 15
LEGEND_TITLE_FONT_SIZE = 15

METHOD_LABELS = {
    "astar_qp_full": "A*QP (full)",
    "astar_qp_wide_space": "A*QP (wide-space)",
    "astar_qp_base": "A*QP (base)",
    "bitstar_ppm": "BIT*",
    "hybrid_astar": "Hybrid-A*",
    "theta_star": "Theta*",
}
METHOD_COLORS = {
    "astar_qp_full": "#2ca02c",
    "astar_qp_wide_space": "#9467bd",
    "astar_qp_base": "#1f77b4",
    "bitstar_ppm": "#d62728",
    "hybrid_astar": "#ff7f0e",
    "theta_star": "#8c564b",
}
METHODS = [
    "astar_qp_full",
    "astar_qp_wide_space",
    "astar_qp_base",
    "bitstar_ppm",
    "hybrid_astar",
    "theta_star",
]

FAMILY_ORDER = {
    "01_sparse": 0,
    "02_dense": 1,
    "03_maze_wide": 2,
    "04_maze_narrow": 3,
}
FAMILY_LABELS = {
    "01_sparse": "Sparse",
    "02_dense": "Dense",
    "03_maze_wide": "Wide\nMaze",
    "04_maze_narrow": "Narrow\nMaze",
}
BOXPLOT_SPECS = {
    "time": {
        "label": "Planning Time [ms] (log scale)",
        "filename": "boxplot_planning_time_by_environment.png",
        "yscale": "log",
    },
    "mean_abs_curvature": {
        "label": "Mean Absolute Curvature [1/m]",
        "filename": "boxplot_mean_abs_curvature_by_environment.png",
    },
    "mean_clearance": {
        "label": "Mean Clearance [m]",
        "filename": "boxplot_mean_clearance_by_environment.png",
    },
}


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    parsed = float(stripped)
    if math.isnan(parsed):
        return None
    return parsed


def resolve_font_family(requested_font: str) -> str:
    if requested_font != "auto":
        return requested_font
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    return next(
        font for font in PREFERRED_FONT_FAMILIES if font in available_fonts
    )


def apply_plot_style(font_family: str) -> None:
    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = BASE_FONT_SIZE
    plt.rcParams["axes.labelsize"] = AXIS_LABEL_FONT_SIZE
    plt.rcParams["xtick.labelsize"] = TICK_LABEL_FONT_SIZE
    plt.rcParams["ytick.labelsize"] = TICK_LABEL_FONT_SIZE
    plt.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE
    plt.rcParams["legend.title_fontsize"] = LEGEND_TITLE_FONT_SIZE


def normalize_method_name(method_name: str) -> str:
    alias_map = {
        "nav2_hybrid_astar": "hybrid_astar",
        "nav2_theta_star": "theta_star",
    }
    normalized = method_name.replace("-", "_")
    return alias_map.get(normalized, normalized)


def infer_environment_and_method(csv_path: Path, input_dir: Path) -> tuple[str, str]:
    relative_parts = csv_path.relative_to(input_dir).parts
    if len(relative_parts) != 4 or relative_parts[-1] != RESULTS_FILENAME:
        raise ValueError(f"Unexpected result path: {csv_path}")

    family, coverage_folder, method_name, _ = relative_parts
    coverage = coverage_folder.removeprefix("coverage_")
    if family not in TARGET_ENVIRONMENT_FOLDERS or not coverage.isdigit():
        raise ValueError(f"Unexpected environment path: {csv_path}")
    return f"{family}/{int(coverage)}", normalize_method_name(method_name)


def find_result_csv_paths(input_dir: Path) -> list[Path]:
    # Only planner summary files are valid input; path trace CSVs use a different schema.
    return sorted(input_dir.glob(f"*/*/*/{RESULTS_FILENAME}"))


def environment_sort_key(environment: str) -> tuple[int, int, str]:
    family, difficulty = environment.split("/")
    return FAMILY_ORDER.get(family, 999), int(difficulty), environment


def format_environment_label(environment: str) -> str:
    family, difficulty = environment.split("/")
    return f"{FAMILY_LABELS.get(family, family)}\n{difficulty}%"


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(FIG_FACE_COLOR)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, color="#bbbbbb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    output_formats: tuple[str, ...],
    output_stem: str,
) -> None:
    for extension in output_formats:
        fig.savefig(
            output_dir / f"{output_stem}.{extension}",
            dpi=FIG_DPI,
            bbox_inches="tight",
        )


def load_environment_rows(
    input_dir: Path,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    environment_rows: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(dict)
    csv_paths = find_result_csv_paths(input_dir)
    if not csv_paths:
        target_text = ", ".join(TARGET_ENVIRONMENT_FOLDERS)
        raise FileNotFoundError(
            f"No {RESULTS_FILENAME} files found under {input_dir} for: {target_text}"
        )

    for csv_path in csv_paths:
        environment, method_name = infer_environment_and_method(csv_path, input_dir)
        if method_name not in METHODS:
            raise ValueError(f"Unsupported planner method in {csv_path}: {method_name}")
        if method_name in environment_rows[environment]:
            raise ValueError(f"Duplicate planner results for {environment}/{method_name}")
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or ())
            missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
            if missing_columns:
                raise ValueError(f"Missing columns in {csv_path}: {missing_columns}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"No result rows in {csv_path}")
        environment_rows[environment][method_name] = rows

    expected_methods = set(METHODS)
    for environment, method_rows in environment_rows.items():
        missing_methods = sorted(expected_methods - set(method_rows))
        if missing_methods:
            raise ValueError(f"Missing planner results for {environment}: {missing_methods}")
        trial_counts = {len(rows) for rows in method_rows.values()}
        if len(trial_counts) != 1:
            raise ValueError(f"Planner trial counts differ for {environment}: {trial_counts}")
    return environment_rows


def build_success_rate_map(
    environment_rows: dict[str, dict[str, list[dict[str, str]]]]
) -> dict[str, dict[str, float]]:
    success_rates: dict[str, dict[str, float]] = {}
    for environment, method_rows in environment_rows.items():
        success_rates[environment] = {}
        for method, rows in method_rows.items():
            trials = len(rows)
            successes = sum(int(row[SUCCESS_COLUMN]) for row in rows)
            success_rates[environment][method] = (successes / trials) if trials else 0.0
    return success_rates


def build_metric_values(
    environment_rows: dict[str, dict[str, list[dict[str, str]]]]
) -> dict[str, dict[str, dict[str, list[float]]]]:
    metric_values: dict[str, dict[str, dict[str, list[float]]]] = {
        metric_name: defaultdict(lambda: defaultdict(list))
        for metric_name in BOXPLOT_SPECS
    }

    for environment, method_rows in environment_rows.items():
        for method, rows in method_rows.items():
            for row in rows:
                if int(row[SUCCESS_COLUMN]) != 1:
                    continue

                for metric_name in ("time", "mean_abs_curvature", "mean_clearance"):
                    value = parse_float(row.get(metric_name))
                    if value is not None:
                        metric_values[metric_name][environment][method].append(value)

    return metric_values

def plot_success_rate(
    success_rates: dict[str, dict[str, float]],
    output_dir: Path,
    output_formats: tuple[str, ...],
) -> None:
    environments = sorted(success_rates, key=environment_sort_key)
    x_positions = list(range(len(environments)))
    bar_width = min(0.20, 0.82 / len(METHODS))
    center_offset = (len(METHODS) - 1) / 2.0

    fig_width = max(11.0, 0.85 * len(environments) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8), constrained_layout=True)
    style_axes(ax)

    for method_index, method in enumerate(METHODS):
        x_offsets = [
            x_value + (method_index - center_offset) * bar_width
            for x_value in x_positions
        ]
        y_values = [100.0 * success_rates[environment].get(method, 0.0) for environment in environments]
        if not any(y_values):
            continue

        bars = ax.bar(
            x_offsets,
            y_values,
            width=bar_width,
            color=METHOD_COLORS[method],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
            label=METHOD_LABELS[method],
        )
        # Match label colors to the bars and stagger adjacent labels to prevent overlap.
        ax.bar_label(
            bars,
            labels=[f"{value:.0f}" for value in y_values],
            padding=2 + 8 * (method_index % 2),
            color=METHOD_COLORS[method],
            fontsize=8,
        )
    ax.set_xticks(x_positions, [format_environment_label(environment) for environment in environments])
    ax.set_ylim(0.0, 108.0)
    ax.set_ylabel("Success Rate [%]")
    ax.legend(
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
    )
    save_figure(fig, output_dir, output_formats, "success_rate_overview")
    plt.close(fig)


def plot_grouped_boxplot(
    metric_name: str,
    values_by_environment: dict[str, dict[str, list[float]]],
    output_dir: Path,
    output_formats: tuple[str, ...],
) -> None:
    spec = BOXPLOT_SPECS[metric_name]
    environments = [
        environment
        for environment in sorted(values_by_environment, key=environment_sort_key)
        if any(values_by_environment[environment].get(method) for method in METHODS)
    ]
    if not environments:
        return

    fig_width = max(11.0, 1.05 * len(environments) + 3.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8), constrained_layout=True)
    style_axes(ax)

    group_gap = 1.25
    bar_width = min(0.22, 0.98 / len(METHODS))
    center_offset = (len(METHODS) - 1) / 2.0
    group_centers = [index * group_gap for index in range(len(environments))]
    plotted_any = False

    for method_index, method in enumerate(METHODS):
        positions: list[float] = []
        plot_values: list[list[float]] = []
        for center, environment in zip(group_centers, environments):
            values = values_by_environment[environment].get(method, [])
            if not values:
                continue
            positions.append(center + (method_index - center_offset) * bar_width)
            plot_values.append(values)

        if not plot_values:
            continue

        plotted_any = True
        boxplot = ax.boxplot(
            plot_values,
            positions=positions,
            widths=bar_width * 0.85,
            patch_artist=True,
            whis=1.5,
            medianprops={"color": "black", "linewidth": 1.4},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markersize": 3.8,
                "markerfacecolor": METHOD_COLORS[method],
                "markeredgecolor": "none",
                "alpha": 0.35,
            },
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.58)

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xticks(group_centers, [format_environment_label(environment) for environment in environments])
    ax.set_ylabel(spec["label"])
    if "yscale" in spec:
        ax.set_yscale(str(spec["yscale"]))
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS[method], alpha=0.58, edgecolor="black")
            for method in METHODS
        ],
        labels=[METHOD_LABELS[method] for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        frameon=True,
    )
    output_stem = Path(str(spec["filename"])).stem
    save_figure(fig, output_dir, output_formats, output_stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=OUTPUT_FORMATS,
        default=list(OUTPUT_FORMATS),
    )
    parser.add_argument("--font-family", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_formats = tuple(args.formats)
    font_family = resolve_font_family(args.font_family)
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style(font_family)

    environment_rows = load_environment_rows(input_dir)
    success_rates = build_success_rate_map(environment_rows)
    metric_values = build_metric_values(environment_rows)

    plot_success_rate(success_rates, output_dir, output_formats)
    for metric_name, values_by_environment in metric_values.items():
        plot_grouped_boxplot(metric_name, values_by_environment, output_dir, output_formats)

    print(f"Saved 4 overview plots to: {output_dir} (font: {font_family})")


if __name__ == "__main__":
    main()
