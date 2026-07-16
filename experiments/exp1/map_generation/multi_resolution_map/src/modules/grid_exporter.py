import csv
from pathlib import Path
from .quadtree_node import QuadtreeNode


def export_leaf_cells_csv(leaves: list[QuadtreeNode], csv_path: Path) -> None:
    # Export all final quadtree leaves for inspection and downstream processing.
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "width",
                "height",
                "center_x",
                "center_y",
                "depth",
                "state",
            ]
        )

        for node in leaves:
            rect = node.rect
            center = rect.center
            writer.writerow(
                [
                    rect.xmin,
                    rect.ymin,
                    rect.xmax,
                    rect.ymax,
                    rect.width,
                    rect.height,
                    center.x,
                    center.y,
                    node.depth,
                    node.state.name if node.state is not None else "UNKNOWN",
                ]
            )
