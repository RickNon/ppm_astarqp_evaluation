from __future__ import annotations

import math
from pathlib import Path

import matplotlib
from matplotlib.patches import Circle, Polygon, Rectangle

from modules.config_loader import PlacementConfig, VisualizationConfig, WorldConfig
from modules.noise_field import DensityFieldResult
from modules.obstacle_generator import ObstacleInstance
from modules.sampler import SamplingResult
from modules.selector import SelectionResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class DensityFieldVisualizer:
    """Save a single density-field image for the earliest pipeline stage."""

    def save_density_map(
        self,
        density: object,
        width: float,
        height: float,
        output_path: str | Path,
        show_axes: bool = True,
        dpi: int = 180,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 7))
        image = ax.imshow(density, extent=[0.0, width, 0.0, height], origin="lower", aspect="equal")

        if show_axes:
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
        else:
            ax.axis("off")

        ax.set_title("Density Field")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="density")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


class Visualizer:
    """Save diagnostic images across density, sampling, and center-selection stages."""

    def __init__(
        self,
        world_cfg: WorldConfig,
        vis_cfg: VisualizationConfig,
        placement_cfg: PlacementConfig | None = None,
    ) -> None:
        self.world_cfg = world_cfg
        self.vis_cfg = vis_cfg
        self.placement_cfg = placement_cfg

    def save_density_only(self, density_result: DensityFieldResult, output_path: str | Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 7))
        self._draw_density(ax, density_result, title="Density Field")
        fig.savefig(output_path, dpi=self.vis_cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    def save_candidates_overlay(
        self,
        density_result: DensityFieldResult,
        sampling_result: SamplingResult,
        output_path: str | Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 7))
        self._draw_density(ax, density_result, title="Sobol Candidates")
        ax.scatter(
            sampling_result.candidates_xy[:, 0],
            sampling_result.candidates_xy[:, 1],
            s=self.vis_cfg.point_size,
            alpha=self.vis_cfg.candidate_alpha,
        )
        fig.savefig(output_path, dpi=self.vis_cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    def save_accepted_overlay(
        self,
        density_result: DensityFieldResult,
        sampling_result: SamplingResult,
        output_path: str | Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 7))
        self._draw_density(ax, density_result, title="Accepted Candidates")
        ax.scatter(
            sampling_result.accepted_xy[:, 0],
            sampling_result.accepted_xy[:, 1],
            s=self.vis_cfg.point_size,
            alpha=self.vis_cfg.accepted_alpha,
        )
        fig.savefig(output_path, dpi=self.vis_cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    def save_selected_overlay(
        self,
        density_result: DensityFieldResult,
        selection_result: SelectionResult,
        output_path: str | Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 7))
        self._draw_density(ax, density_result, title="Selected Obstacle Centers")
        ax.scatter(
            selection_result.selected_xy[:, 0],
            selection_result.selected_xy[:, 1],
            s=self.vis_cfg.point_size * 2.2,
            alpha=self.vis_cfg.selected_alpha,
        )

        if self.vis_cfg.draw_spacing_circle:
            if self.placement_cfg is None:
                raise ValueError("placement_cfg is required when draw_spacing_circle is enabled.")
            radius = self.placement_cfg.min_spacing * 0.5
            for point in selection_result.selected_xy:
                # English comment: This circle is only a spacing aid and not the final obstacle footprint.
                ax.add_patch(Circle((point[0], point[1]), radius=radius, fill=False, alpha=0.18))

        fig.savefig(output_path, dpi=self.vis_cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    def _draw_density(self, ax: plt.Axes, density_result: DensityFieldResult, title: str) -> None:
        image = ax.imshow(
            density_result.density,
            extent=[0.0, self.world_cfg.width, 0.0, self.world_cfg.height],
            origin="lower",
            aspect="equal",
        )
        if self.vis_cfg.show_axes:
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
        else:
            ax.axis("off")
        ax.set_title(title)
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="density")


class LayoutVisualizer:
    """Create a 2D preview of the generated world layout."""

    def __init__(self, world_cfg: WorldConfig, vis_cfg: VisualizationConfig) -> None:
        self.world_cfg = world_cfg
        self.vis_cfg = vis_cfg

    def save(self, obstacles: list[ObstacleInstance], output_path: str | Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")

        ax.add_patch(Rectangle((0.0, 0.0), self.world_cfg.width, self.world_cfg.height, fill=False, linewidth=1.2))
        self._draw_outer_walls(ax)

        for obstacle in obstacles:
            if obstacle.kind == "box":
                self._draw_box(ax, obstacle)
            elif obstacle.kind == "cylinder":
                self._draw_cylinder(ax, obstacle)

        ax.set_xlim(-self.world_cfg.wall_thickness * 1.5, self.world_cfg.width + self.world_cfg.wall_thickness * 1.5)
        ax.set_ylim(-self.world_cfg.wall_thickness * 1.5, self.world_cfg.height + self.world_cfg.wall_thickness * 1.5)

        if self.vis_cfg.show_axes:
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
        else:
            ax.axis("off")

        ax.set_title("Random Obstacle Layout Preview")
        fig.savefig(output_path, dpi=self.vis_cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    def _draw_outer_walls(self, ax: plt.Axes) -> None:
        t = self.world_cfg.wall_thickness
        w = self.world_cfg.width
        h = self.world_cfg.height
        wall_rects = [
            Rectangle((-t, -t), w + 2.0 * t, t, alpha=self.vis_cfg.wall_alpha),
            Rectangle((-t, h), w + 2.0 * t, t, alpha=self.vis_cfg.wall_alpha),
            Rectangle((-t, 0.0), t, h, alpha=self.vis_cfg.wall_alpha),
            Rectangle((w, 0.0), t, h, alpha=self.vis_cfg.wall_alpha),
        ]
        for rect in wall_rects:
            ax.add_patch(rect)

    def _draw_box(self, ax: plt.Axes, obstacle: ObstacleInstance) -> None:
        half_x = 0.5 * float(obstacle.size_x)
        half_y = 0.5 * float(obstacle.size_y)
        corners = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]

        c = math.cos(obstacle.yaw)
        s = math.sin(obstacle.yaw)
        rotated = []
        for x, y in corners:
            rx = c * x - s * y + obstacle.center_x
            ry = s * x + c * y + obstacle.center_y
            rotated.append((rx, ry))

        ax.add_patch(Polygon(rotated, closed=True, alpha=self.vis_cfg.obstacle_alpha))

    def _draw_cylinder(self, ax: plt.Axes, obstacle: ObstacleInstance) -> None:
        ax.add_patch(
            Circle((obstacle.center_x, obstacle.center_y), float(obstacle.radius), alpha=self.vis_cfg.obstacle_alpha)
        )
