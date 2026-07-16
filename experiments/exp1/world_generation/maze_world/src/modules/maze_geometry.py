from __future__ import annotations

from dataclasses import dataclass

from .config_loader import MazeWorldConfig
from .maze_topology import MazeTopology


@dataclass(frozen=True)
class WallBox:
    name: str
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    height: float


class MazeGeometryBuilder:
    """Convert the occupancy raster into wall-aligned box primitives."""

    def __init__(self, config: MazeWorldConfig) -> None:
        self.config = config

    def build_walls(self, topology: MazeTopology) -> list[WallBox]:
        tile_widths = self._build_axis_widths(
            count=self.config.maze.cols,
            corridor_width=self.config.layout.corridor_width,
            wall_thickness=self.config.world.wall_thickness,
        )
        tile_heights = self._build_axis_widths(
            count=self.config.maze.rows,
            corridor_width=self.config.layout.corridor_width,
            wall_thickness=self.config.world.wall_thickness,
        )
        x_edges = self._build_edges(tile_widths)
        y_edges = self._build_edges(tile_heights)
        occupancy = topology.occupancy
        last_row = len(occupancy) - 1
        last_col = len(occupancy[0]) - 1

        walls: list[WallBox] = self._build_outer_walls()
        internal_boxes = self._merge_internal_walls(
            occupancy=occupancy,
            x_edges=x_edges,
            y_edges=y_edges,
            max_internal_row=last_row - 1,
            max_internal_col=last_col - 1,
        )
        for wall_index, box in enumerate(internal_boxes):
            walls.append(
                WallBox(
                    name=f"maze_wall_{wall_index}",
                    center_x=box.center_x,
                    center_y=box.center_y,
                    size_x=box.size_x,
                    size_y=box.size_y,
                    height=self.config.world.wall_height,
                )
            )
        return walls

    def _build_outer_walls(self) -> list[WallBox]:
        world = self.config.world
        half_height = 0.5 * world.wall_height
        _ = half_height  # Keep the value explicit for symmetry with SDF serialization.
        wall_specs = [
            ("outer_wall_bottom", 0.5 * world.width, -0.5 * world.wall_thickness, world.width + 2.0 * world.wall_thickness, world.wall_thickness),
            ("outer_wall_top", 0.5 * world.width, world.height + 0.5 * world.wall_thickness, world.width + 2.0 * world.wall_thickness, world.wall_thickness),
            ("outer_wall_left", -0.5 * world.wall_thickness, 0.5 * world.height, world.wall_thickness, world.height),
            ("outer_wall_right", world.width + 0.5 * world.wall_thickness, 0.5 * world.height, world.wall_thickness, world.height),
        ]
        return [
            WallBox(
                name=name,
                center_x=center_x,
                center_y=center_y,
                size_x=size_x,
                size_y=size_y,
                height=world.wall_height,
            )
            for name, center_x, center_y, size_x, size_y in wall_specs
        ]

    def _merge_internal_walls(
        self,
        occupancy: list[list[int]],
        x_edges: list[float],
        y_edges: list[float],
        max_internal_row: int,
        max_internal_col: int,
    ) -> list[WallBox]:
        visited = [[False for _ in row] for row in occupancy]
        merged_boxes: list[WallBox] = []

        for row_index in range(1, max_internal_row + 1):
            for col_index in range(1, max_internal_col + 1):
                if occupancy[row_index][col_index] != 1 or visited[row_index][col_index]:
                    continue

                col_end = col_index
                while (
                    col_end + 1 <= max_internal_col
                    and occupancy[row_index][col_end + 1] == 1
                    and not visited[row_index][col_end + 1]
                ):
                    col_end += 1

                row_end = row_index
                while row_end + 1 <= max_internal_row:
                    if not self._is_full_unvisited_run(
                        occupancy=occupancy,
                        visited=visited,
                        row_index=row_end + 1,
                        col_start=col_index,
                        col_end=col_end,
                    ):
                        break
                    row_end += 1

                for fill_row in range(row_index, row_end + 1):
                    for fill_col in range(col_index, col_end + 1):
                        visited[fill_row][fill_col] = True

                merged_boxes.append(
                    WallBox(
                        name="internal_wall",
                        center_x=0.5 * (x_edges[col_index] + x_edges[col_end + 1]),
                        center_y=0.5 * (y_edges[row_index] + y_edges[row_end + 1]),
                        size_x=x_edges[col_end + 1] - x_edges[col_index],
                        size_y=y_edges[row_end + 1] - y_edges[row_index],
                        height=self.config.world.wall_height,
                    )
                )

        return merged_boxes

    @staticmethod
    def _is_full_unvisited_run(
        occupancy: list[list[int]],
        visited: list[list[bool]],
        row_index: int,
        col_start: int,
        col_end: int,
    ) -> bool:
        for col_index in range(col_start, col_end + 1):
            if occupancy[row_index][col_index] != 1 or visited[row_index][col_index]:
                return False
        return True

    @staticmethod
    def _build_axis_widths(count: int, corridor_width: float, wall_thickness: float) -> list[float]:
        widths: list[float] = []
        for index in range(2 * count + 1):
            if index == 0 or index == 2 * count:
                widths.append(0.0)
            elif index % 2 == 0:
                widths.append(wall_thickness)
            else:
                widths.append(corridor_width)
        return widths

    @staticmethod
    def _build_edges(widths: list[float]) -> list[float]:
        edges = [0.0]
        for width in widths:
            edges.append(edges[-1] + width)
        return edges

