from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MazeTopology:
    cols: int
    rows: int
    occupancy: list[list[int]]

    def count_open_passages(self) -> int:
        """Count corridor connectors between logical maze cells."""
        passages = 0
        for row_index in range(self.rows):
            for col_index in range(self.cols):
                grid_y = 2 * row_index + 1
                grid_x = 2 * col_index + 1
                if col_index + 1 < self.cols and self.occupancy[grid_y][grid_x + 1] == 0:
                    passages += 1
                if row_index + 1 < self.rows and self.occupancy[grid_y + 1][grid_x] == 0:
                    passages += 1
        return passages


class MazeTopologyGenerator:
    """Generate a perfect maze as an occupancy raster with explicit wall slots."""

    def __init__(self, cols: int, rows: int, seed: int) -> None:
        self.cols = cols
        self.rows = rows
        self._random = random.Random(seed)

    def generate(self) -> MazeTopology:
        occupancy = [[1 for _ in range(2 * self.cols + 1)] for _ in range(2 * self.rows + 1)]
        visited = [[False for _ in range(self.cols)] for _ in range(self.rows)]

        def carve(cell_x: int, cell_y: int) -> None:
            visited[cell_y][cell_x] = True
            occupancy[2 * cell_y + 1][2 * cell_x + 1] = 0

            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            self._random.shuffle(directions)
            for dx, dy in directions:
                next_x = cell_x + dx
                next_y = cell_y + dy
                if not (0 <= next_x < self.cols and 0 <= next_y < self.rows):
                    continue
                if visited[next_y][next_x]:
                    continue

                # Open the wall slot between the current cell and the next cell.
                occupancy[2 * cell_y + 1 + dy][2 * cell_x + 1 + dx] = 0
                carve(next_x, next_y)

        carve(0, 0)
        return MazeTopology(cols=self.cols, rows=self.rows, occupancy=occupancy)

