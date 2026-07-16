from dataclasses import dataclass, field
from .geometry2d import AARect
from .occupancy_classifier import CellState


@dataclass
class QuadtreeNode:
    rect: AARect
    depth: int
    state: CellState | None = None
    children: list["QuadtreeNode"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def size(self) -> float:
        # Cells are expected to remain square.
        return self.rect.width

    def split(self) -> None:
        # Split the current square cell into 4 equal child cells.
        xm = 0.5 * (self.rect.xmin + self.rect.xmax)
        ym = 0.5 * (self.rect.ymin + self.rect.ymax)

        self.children = [
            QuadtreeNode(
                rect=AARect(self.rect.xmin, self.rect.ymin, xm, ym),
                depth=self.depth + 1,
            ),
            QuadtreeNode(
                rect=AARect(xm, self.rect.ymin, self.rect.xmax, ym),
                depth=self.depth + 1,
            ),
            QuadtreeNode(
                rect=AARect(self.rect.xmin, ym, xm, self.rect.ymax),
                depth=self.depth + 1,
            ),
            QuadtreeNode(
                rect=AARect(xm, ym, self.rect.xmax, self.rect.ymax),
                depth=self.depth + 1,
            ),
        ]
