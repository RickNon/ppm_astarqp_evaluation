from enum import Enum, auto
from .geometry2d import AARect
from .obstacles import Obstacle2D


class CellState(Enum):
    FREE = auto()
    OCCUPIED = auto()
    MIXED = auto()


class OccupancyClassifier:
    def __init__(self, obstacles: list[Obstacle2D]) -> None:
        self._obstacles = obstacles

    def classify(self, rect: AARect) -> CellState:
        # First, check whether the cell is fully inside any obstacle.
        # If so, it is occupied and subdivision can stop.
        for obs in self._obstacles:
            if obs.contains_rect(rect):
                return CellState.OCCUPIED

        # Then, check whether the cell intersects any obstacle boundary or body.
        # If there is any intersection but no full containment, it is mixed.
        for obs in self._obstacles:
            if obs.intersects_rect(rect):
                return CellState.MIXED

        # Otherwise, the cell is completely free.
        return CellState.FREE
