from __future__ import annotations

from pathlib import Path
from collections import Counter

from modules.area_analyzer import AreaAnalysisResult
from modules.obstacle_generator import ObstacleInstance


class LogWriter:
    """Write a text log that summarizes the generated world."""

    def write(
        self,
        output_path: str | Path,
        world_name: str,
        obstacles: list[ObstacleInstance],
        area_result: AreaAnalysisResult,
    ) -> None:
        counter = Counter(obstacle.template_name for obstacle in obstacles)
        lines: list[str] = []
        lines.append(f"world_name: {world_name}")
        lines.append(f"obstacle_count: {len(obstacles)}")
        lines.append(f"area_resolution: {area_result.area_resolution:.6f}")
        lines.append(f"analysis_grid_shape: {area_result.grid_shape[0]} x {area_result.grid_shape[1]}")
        lines.append(f"inner_area_m2: {area_result.inner_area:.6f}")
        lines.append(f"occupied_area_m2: {area_result.occupied_area:.6f}")
        lines.append(f"free_area_m2: {area_result.free_area:.6f}")
        lines.append(f"occupied_ratio: {area_result.occupied_ratio:.6f}")
        lines.append(f"free_ratio: {area_result.free_ratio:.6f}")
        lines.append("obstacle_counts_by_template:")
        for name, count in sorted(counter.items()):
            lines.append(f"  {name}: {count}")

        Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
