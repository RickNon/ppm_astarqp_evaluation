from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


EPS_FONT_TYPE = 3
font_manager.findfont(
    "Times New Roman",
    fallback_to_default=False,
)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": EPS_FONT_TYPE,
    }
)


FOCUS_COVERAGES = (80, 90)
PLOT_FONT_SIZE = 14.0

PLOT_OUTPUT_FORMATS = ("png", "eps")
SUPPORTED_OUTPUT_FORMATS = frozenset(("png", "eps", "pdf", "svg"))
RASTER_DPI = 220
ENVIRONMENT_ORDER = ("01_sparse", "02_dense")
ENVIRONMENT_LABELS = {
    "01_sparse": "Sparse",
    "02_dense": "Dense",
}


def _focus_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and order the four conditions used in the main figures."""
    focused = frame.loc[frame["coverage"].isin(FOCUS_COVERAGES)].copy()
    focused["_environment_order"] = pd.Categorical(
        focused["environment"],
        categories=ENVIRONMENT_ORDER,
        ordered=True,
    )
    focused = focused.sort_values(
        ["_environment_order", "coverage"],
        kind="stable",
    ).drop(columns="_environment_order")
    focused = focused.reset_index(drop=True)

    expected = {
        (environment, coverage)
        for environment in ENVIRONMENT_ORDER
        for coverage in FOCUS_COVERAGES
    }
    actual = set(
        zip(
            focused["environment"].astype(str),
            focused["coverage"].astype(int),
        )
    )
    if actual != expected or len(focused) != len(expected):
        raise ValueError(
            "Plot input must contain exactly one row for each sparse/dense "
            "coverage-80/90 condition."
        )
    return focused


def _condition_labels(frame: pd.DataFrame, n_col: str) -> list[str]:
    """Format environment, coverage, and valid-pair count on one axis label."""
    return [
        (
            f"{ENVIRONMENT_LABELS.get(row.environment, row.environment)} "
            f"{int(row.coverage)}% (n={int(getattr(row, n_col))})"
        )
        for row in frame.itertuples(index=False)
    ]


def _scaled_font_size(scale: float) -> float:
    """Scale secondary text from the configured base font size."""
    return float(plt.rcParams["font.size"]) * scale


def _normalized_output_formats() -> tuple[str, ...]:
    """Validate and normalize the formats configured at module level."""
    formats = tuple(str(value).lower().lstrip(".") for value in PLOT_OUTPUT_FORMATS)
    if not formats:
        raise ValueError("At least one plot output format must be configured.")
    if len(formats) != len(set(formats)):
        raise ValueError("Plot output formats must not contain duplicates.")
    unsupported = sorted(set(formats) - SUPPORTED_OUTPUT_FORMATS)
    if unsupported:
        raise ValueError(
            f"Unsupported plot output formats: {', '.join(unsupported)}. "
            f"Choose from: {', '.join(sorted(SUPPORTED_OUTPUT_FORMATS))}."
        )
    return formats


def _validate_eps_embedded_glyphs(output_path: Path) -> None:
    """Require self-contained Times New Roman Type 3 glyph definitions."""
    content = output_path.read_text(encoding="latin-1")
    if "/FontType 3 def" not in content:
        raise RuntimeError(f"EPS does not contain embedded Type 3 glyphs: {output_path}.")
    if "/FontName /TimesNewRoman" not in content:
        raise RuntimeError(f"EPS does not use Times New Roman glyphs: {output_path}.")
    external_font_markers = (
        "%%DocumentNeededResources: font",
        "%%IncludeResource: font",
    )
    if any(marker in content for marker in external_font_markers):
        raise RuntimeError(f"EPS still depends on an external font: {output_path}.")


def _save_figure(
    figure: plt.Figure,
    output_stem: Path,
    output_formats: tuple[str, ...],
) -> None:
    """Save one figure in every configured format."""
    for output_format in output_formats:
        output_path = output_stem.with_suffix(f".{output_format}")
        save_options = {
            "format": output_format,
            "bbox_inches": "tight",
        }
        if output_format == "png":
            save_options["dpi"] = RASTER_DPI
        figure.savefig(output_path, **save_options)
        if output_format == "eps":
            _validate_eps_embedded_glyphs(output_path)


def _plot_relative_change_iqr(
    frame: pd.DataFrame,
    output_stem: Path,
    *,
    output_formats: tuple[str, ...],
    n_col: str,
    median_col: str,
    q1_col: str,
    q3_col: str,
    color: str,
    ylabel: str,
    extra_y_tick: float | None = None,
) -> None:
    """Plot vertical pairwise medians and interquartile ranges."""
    labels = _condition_labels(frame, n_col)
    x = np.arange(len(frame), dtype=float)
    medians = frame[median_col].to_numpy(dtype=float)
    q1 = frame[q1_col].to_numpy(dtype=float)
    q3 = frame[q3_col].to_numpy(dtype=float)
    y_min = min(0.0, float(np.min(q1)))
    y_max = max(0.0, float(np.max(q3)))
    span = max(y_max - y_min, 1.0)

    fig, axis = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    axis.errorbar(
        x,
        medians,
        yerr=np.vstack((medians - q1, q3 - medians)),
        fmt="o",
        color=color,
        ecolor=color,
        markersize=7,
        capsize=5,
        linewidth=2.0,
    )
    for position, median in zip(x, medians):
        axis.annotate(
            f"{median:+.1f}%" if median != 0.0 else "0.0%",
            (position, median),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=_scaled_font_size(0.90),
        )

    axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    axis.set_xlim(-0.70, len(frame) - 0.30)
    axis.set_ylim(y_min - 0.08 * span, y_max + 0.08 * span)
    if extra_y_tick is not None:
        lower_limit, upper_limit = axis.get_ylim()
        ticks = np.append(axis.get_yticks(), extra_y_tick)
        ticks = np.unique(ticks[(ticks >= lower_limit) & (ticks <= upper_limit)])
        axis.set_yticks(ticks)
    axis.set_xticks(x, labels)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", linestyle=":", color="#C9CDD2", linewidth=0.8)
    _save_figure(fig, output_stem, output_formats)
    plt.close(fig)


def _plot_direction_composition(
    frame: pd.DataFrame,
    output_stem: Path,
    *,
    output_formats: tuple[str, ...],
    n_col: str,
    count_cols: tuple[str, str, str],
    direction_labels: tuple[str, str, str],
    direction_colors: tuple[str, str, str],
) -> None:
    """Plot vertical direction shares and exact counts for every valid pair."""
    labels = _condition_labels(frame, n_col)
    x = np.arange(len(frame), dtype=float)
    counts = frame.loc[:, list(count_cols)].to_numpy(dtype=float)
    totals = counts.sum(axis=1)
    expected_totals = frame[n_col].to_numpy(dtype=float)
    if not np.array_equal(totals, expected_totals):
        raise ValueError("Direction counts must sum to the valid-pair count.")
    shares = 100.0 * counts / totals[:, None]

    fig, axis = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    bottom = np.zeros(len(frame), dtype=float)
    for direction_index in range(3):
        axis.bar(
            x,
            shares[:, direction_index],
            bottom=bottom,
            width=0.62,
            color=direction_colors[direction_index],
            label=direction_labels[direction_index],
        )
        for row_index, x_value in enumerate(x):
            height = shares[row_index, direction_index]
            if height >= 8.0:
                axis.text(
                    x_value,
                    bottom[row_index] + 0.5 * height,
                    str(int(counts[row_index, direction_index])),
                    ha="center",
                    va="center",
                    fontsize=_scaled_font_size(0.85),
                    color="white" if direction_index != 1 else "#263238",
                    fontweight="bold",
                )
        bottom += shares[:, direction_index]

    for row_index, x_value in enumerate(x):
        axis.text(
            x_value,
            103.0,
            "/".join(str(int(value)) for value in counts[row_index]),
            ha="center",
            va="center",
            fontsize=_scaled_font_size(0.80),
        )

    axis.set_xlim(-0.70, len(frame) - 0.30)
    axis.set_ylim(0.0, 110.0)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_xticks(x, labels)
    axis.set_ylabel("Share of valid pairs [%]")
    axis.grid(True, axis="y", linestyle=":", color="#C9CDD2", linewidth=0.8)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=_scaled_font_size(0.90),
    )
    _save_figure(fig, output_stem, output_formats)
    plt.close(fig)


def generate_mechanism_plots(
    base_wide: pd.DataFrame,
    wide_full: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate only the separate B and C figures for both comparisons."""
    if PLOT_FONT_SIZE <= 0.0:
        raise ValueError("Plot font size must be positive.")
    output_formats = _normalized_output_formats()
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE,
            "axes.labelsize": PLOT_FONT_SIZE,
            "xtick.labelsize": 0.90 * PLOT_FONT_SIZE,
            "ytick.labelsize": 0.90 * PLOT_FONT_SIZE,
            "legend.fontsize": 0.90 * PLOT_FONT_SIZE,
        }
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    wide_frame = _focus_frame(base_wide)
    _plot_relative_change_iqr(
        wide_frame,
        figure_dir / "wide_space_relative_change_median_iqr_coverage_80_90",
        output_formats=output_formats,
        n_col="n_openness_ratio_valid",
        median_col="relative_openness_change_median_pct",
        q1_col="relative_openness_change_q1_pct",
        q3_col="relative_openness_change_q3_pct",
        color="#2F80ED",
        ylabel="Relative effective-openness change [%]",
        extra_y_tick=-20.0,
    )
    _plot_direction_composition(
        wide_frame,
        figure_dir / "wide_space_direction_composition_coverage_80_90",
        output_formats=output_formats,
        n_col="n_openness_ratio_valid",
        count_cols=(
            "n_openness_increase",
            "n_openness_tie",
            "n_openness_decrease",
        ),
        direction_labels=("Higher", "Unchanged", "Lower"),
        direction_colors=("#0072B2", "#B8BDC5", "#D55E00"),
    )

    smoothing_frame = _focus_frame(wide_full)
    _plot_relative_change_iqr(
        smoothing_frame,
        figure_dir / "smoothing_relative_change_median_iqr_coverage_80_90",
        output_formats=output_formats,
        n_col="n_energy_ratio_valid",
        median_col="relative_energy_change_median_pct",
        q1_col="relative_energy_change_q1_pct",
        q3_col="relative_energy_change_q3_pct",
        color="#2CA25F",
        ylabel="Relative second-difference-energy change [%]",
    )
    _plot_direction_composition(
        smoothing_frame,
        figure_dir / "smoothing_direction_composition_coverage_80_90",
        output_formats=output_formats,
        n_col="n_energy_ratio_valid",
        count_cols=("n_energy_decrease", "n_energy_tie", "n_energy_increase"),
        direction_labels=("Lower", "Unchanged", "Higher"),
        direction_colors=("#0072B2", "#B8BDC5", "#D55E00"),
    )
