from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 18


@dataclass(frozen=True)
class SeriesData:
    label: str
    source_path: Path
    num_sensors: list[float]
    coverage_ratio: list[float]


@dataclass(frozen=True)
class SeriesGroup:
    raw_series_list: list[SeriesData]
    mean_series: SeriesData | None = None

PPM_SAMPLING_METHODS = (
    ("independent_random", "PPM (Independent)"),
    ("halton_random", "PPM (Halton)"),
)


def get_main_x_limit(folder_name: str) -> int:
    # Use a wider main-axis view only for the narrow maze scenario.
    if folder_name == "04_maze_narrow":
        return 300
    return 200


def get_main_x_tick_step(main_x_limit: int) -> int:
    # Keep the main-axis ticks readable instead of crowding the zoomed view.
    if main_x_limit <= 200:
        return 20
    return 25


def find_min_positive_sensor_count(series_groups: Iterable[SeriesGroup]) -> float | None:
    # Log-scale axes cannot include zero, so find the smallest positive x value.
    min_positive_value: float | None = None
    for series_group in series_groups:
        candidate_series = list(series_group.raw_series_list)
        if series_group.mean_series is not None:
            candidate_series.append(series_group.mean_series)
        for series in candidate_series:
            for num_sensors in series.num_sensors:
                if num_sensors > 0 and (
                    min_positive_value is None or num_sensors < min_positive_value
                ):
                    min_positive_value = num_sensors
    return min_positive_value


def read_series(csv_path: Path, label: str) -> SeriesData:
    num_sensors: list[float] = []
    coverage_ratio: list[float] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"num_sensors", "coverage_ratio"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns in {csv_path}: {missing_str}")
        for row in reader:
            num_sensors.append(float(row["num_sensors"]))
            coverage_ratio.append(float(row["coverage_ratio"]))
    if not num_sensors:
        raise ValueError(f"No data rows found in {csv_path}")
    return SeriesData(label, csv_path, num_sensors, coverage_ratio)


def build_mean_series(
    series_list: Iterable[SeriesData], label: str, source_path: Path
) -> SeriesData:
    series_items = list(series_list)
    if not series_items:
        raise ValueError("At least one series is required to build a mean series.")
    coverage_by_sensor_count: dict[float, list[float]] = {}
    for series in series_items:
        if len(series.num_sensors) != len(series.coverage_ratio):
            raise ValueError(f"Invalid series length mismatch in {series.source_path}")
        # Average by sensor count because runs can terminate at different lengths.
        for num_sensors, coverage_ratio in zip(series.num_sensors, series.coverage_ratio):
            coverage_by_sensor_count.setdefault(num_sensors, []).append(coverage_ratio)
    mean_num_sensors = sorted(coverage_by_sensor_count)
    mean_coverage_ratio = [
        sum(coverage_by_sensor_count[count]) / len(coverage_by_sensor_count[count])
        for count in mean_num_sensors
    ]
    return SeriesData(label, source_path, mean_num_sensors, mean_coverage_ratio)


def save_plot(folder_name: str, series_groups: list[SeriesGroup], output_path: Path) -> None:
    fig, (ax, log_ax) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    main_x_limit = get_main_x_limit(folder_name)
    styles = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd", "#ff7f0e"]

    for index, series_group in enumerate(series_groups):
        color = styles[index % len(styles)]
        for series in series_group.raw_series_list:
            raw_label = series.label if series_group.mean_series is None else "_nolegend_"
            for target_ax, linewidth in ((ax, 0.9), (log_ax, 0.8)):
                target_ax.plot(
                    series.num_sensors,
                    series.coverage_ratio,
                    label=raw_label if target_ax is ax else "_nolegend_",
                    color=color,
                    linewidth=linewidth,
                    linestyle="--" if series_group.mean_series is not None else "-",
                )
        if series_group.mean_series is not None:
            mean = series_group.mean_series
            ax.plot(mean.num_sensors, mean.coverage_ratio, label=mean.label, color=color, linewidth=3.0)
            log_ax.plot(mean.num_sensors, mean.coverage_ratio, color=color, linewidth=2.4)

    ax.set_xlabel("Number of Cells")
    ax.set_ylabel("Free Space Coverage Ratio")
    ax.set_xlim(0, main_x_limit)
    ax.set_ylim(0.0, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(get_main_x_tick_step(main_x_limit)))
    ax.grid(True, linestyle="--", color="lightgrey", linewidth=0.5)
    ax.legend(loc="center right", frameon=True, framealpha=0.9, fontsize=15)

    log_ax.set_xlabel("Number of Cells (Log Scale)")
    min_positive_x = find_min_positive_sensor_count(series_groups)
    if min_positive_x is not None:
        log_ax.set_xscale("log")
        log_ax.set_xlim(left=min_positive_x)
    log_ax.set_ylim(0.0, 1.02)
    log_ax.grid(True, linestyle="--", color="lightgrey", linewidth=0.4)
    log_ax.tick_params(labelsize=16)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.92, wspace=0.22)
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".eps"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[3]
    ppm_output_dir = repository_root / "results" / "exp1_map"
    multi_output_dir = repository_root / "results" / "exp1_map"
    output_dir = repository_root / "results" / "exp1_map" / "figures"

    parser = argparse.ArgumentParser(
        description=(
            "Create coverage-ratio comparison plots for PPM random, PPM halton, "
            "and multi-resolution grid outputs."
        )
    )
    parser.add_argument(
        "--ppm-output-dir",
        type=Path,
        default=ppm_output_dir,
        help="Directory that contains PPM output folders.",
    )
    parser.add_argument(
        "--multi-output-dir",
        type=Path,
        default=multi_output_dir,
        help="Directory that contains multi-resolution grid output folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_dir,
        help="Directory where comparison plots will be saved.",
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="Specific folder names to plot. When omitted, all common folders are checked.",
    )
    return parser.parse_args()


def find_ppm_csv_paths(method_dir: Path) -> list[Path]:
    # Read repeated runs below the sampling-method directory.
    csv_paths = sorted(method_dir.glob("*/ppm_area.csv"))
    if csv_paths:
        return csv_paths

    # Accept a direct CSV for single-run output layouts.
    direct_csv = method_dir / "ppm_area.csv"
    if direct_csv.exists():
        return [direct_csv]

    return []


def collect_ppm_method_series(folder_dir: Path, method_name: str, label: str) -> SeriesGroup | None:
    method_dir = folder_dir / method_name
    if not method_dir.is_dir():
        return None

    csv_paths = find_ppm_csv_paths(method_dir)
    if not csv_paths:
        return None

    raw_series_list = []
    for csv_path in csv_paths:
        run_label = csv_path.parent.name if csv_path.parent != method_dir else csv_path.stem
        raw_series_list.append(read_series(csv_path, f"{label} ({run_label})"))

    mean_series = build_mean_series(
        raw_series_list,
        label=label,
        source_path=method_dir,
    )
    return SeriesGroup(raw_series_list=raw_series_list, mean_series=mean_series)


def collect_ppm_sampling_series(folder_dir: Path) -> list[SeriesGroup]:
    series_groups: list[SeriesGroup] = []

    for method_name, label in PPM_SAMPLING_METHODS:
        series_group = collect_ppm_method_series(folder_dir, method_name, label)
        if series_group is not None:
            series_groups.append(series_group)

    return series_groups


def collect_multi_resolution_grid_series(folder_dir: Path) -> list[SeriesGroup]:
    direct_csv = folder_dir / "ppm_area.csv"
    if direct_csv.exists():
        return [
            SeriesGroup(
                raw_series_list=[read_series(direct_csv, "Multi-resolution Grid")],
            )
        ]

    csv_paths = sorted(folder_dir.glob("free_area_vs_resolution_*.csv"))
    series_list: list[SeriesGroup] = []

    for csv_path in csv_paths:
        suffix = csv_path.stem.removeprefix("free_area_vs_resolution_")

        if suffix in {"1p0", "0p5", "0p1"}:
            continue
        if suffix == "0p05":
            label = "Multi-resolution Grid"
        else:
            label = f"Multi-resolution Grid ({suffix})"

        series_list.append(
            SeriesGroup(
                raw_series_list=[read_series(csv_path, label)],
            )
        )

    return series_list


def find_target_folders(
    ppm_output_dir: Path,
    multi_output_dir: Path,
    requested_folders: list[str] | None,
) -> list[str]:
    ppm_folders = {path.name for path in ppm_output_dir.iterdir() if path.is_dir()}
    multi_folders = {
        path.parent.parent.name
        for path in multi_output_dir.glob(
            "*/multi_resolution_map/free_area_vs_resolution_0p05.csv"
        )
    }
    common_folders = sorted(ppm_folders & multi_folders)

    if requested_folders is None:
        return common_folders

    requested_set = set(requested_folders)
    missing_folders = sorted(requested_set.difference(common_folders))
    if missing_folders:
        missing_str = ", ".join(missing_folders)
        raise ValueError(f"Folders not found in both output trees: {missing_str}")

    return requested_folders


def group_label(series_group: SeriesGroup) -> str:
    if series_group.mean_series is not None:
        return series_group.mean_series.label
    return series_group.raw_series_list[0].label


def missing_required_labels(series_groups: list[SeriesGroup]) -> list[str]:
    existing_labels = {group_label(series_group) for series_group in series_groups}
    required_labels = [label for _, label in PPM_SAMPLING_METHODS]
    required_labels.append("Multi-resolution Grid")

    return [label for label in required_labels if label not in existing_labels]


def main() -> None:
    args = parse_args()
    ppm_output_dir = args.ppm_output_dir.resolve()
    multi_output_dir = args.multi_output_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not ppm_output_dir.is_dir():
        raise NotADirectoryError(f"PPM output directory not found: {ppm_output_dir}")
    if not multi_output_dir.is_dir():
        raise NotADirectoryError(f"multi-res output directory not found: {multi_output_dir}")

    if args.folders is None:
        ppm_folder_names = {path.name for path in ppm_output_dir.iterdir() if path.is_dir()}
        multi_folder_names = {path.name for path in multi_output_dir.iterdir() if path.is_dir()}
        ppm_only_folders = sorted(ppm_folder_names.difference(multi_folder_names))
        if ppm_only_folders:
            ppm_only_str = ", ".join(ppm_only_folders)
            print(f"[WARN] Ignored PPM folders without matching multi-res folder: {ppm_only_str}")

    folder_names = find_target_folders(ppm_output_dir, multi_output_dir, args.folders)
    if not folder_names:
        raise ValueError("No matching folders were found between the two output directories.")

    output_dir.mkdir(parents=True, exist_ok=True)

    created_count = 0
    for folder_name in folder_names:
        ppm_series = collect_ppm_sampling_series(ppm_output_dir / folder_name)
        multi_series = collect_multi_resolution_grid_series(multi_output_dir / folder_name / "multi_resolution_map")
        series_groups = ppm_series + multi_series

        missing_labels = missing_required_labels(series_groups)
        if missing_labels:
            missing_str = ", ".join(missing_labels)
            print(f"[WARN] Skipped {folder_name}: missing required series: {missing_str}")
            continue

        output_path = output_dir / f"{folder_name}.png"
        save_plot(folder_name, series_groups, output_path)
        created_count += 1
        print(f"[INFO] Saved plot: {output_path}")
        for series_group in series_groups:
            for series in series_group.raw_series_list:
                print(f"       - {series.label}: {series.source_path}")
            if series_group.mean_series is not None:
                print(f"       - {series_group.mean_series.label}: {series_group.mean_series.source_path}")

    if created_count == 0:
        raise RuntimeError("No plots were created. Check the input directories and CSV layout.")


if __name__ == "__main__":
    main()
