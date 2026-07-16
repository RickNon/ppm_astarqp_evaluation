from .models import Model, Box
from .geometry2d import AARect


def compute_inner_domain_from_walls(models: list[Model]) -> AARect:
    walls = [m for m in models if m.type == "wall"]
    if len(walls) != 4:
        raise ValueError(f"Expected exactly 4 walls, but got {len(walls)}")

    wall_map = {w.name: w for w in walls}

    required = [
        "outer_wall_bottom",
        "outer_wall_top",
        "outer_wall_left",
        "outer_wall_right",
    ]
    for name in required:
        if name not in wall_map:
            raise ValueError(f"Missing required wall model: {name}")

    bottom = wall_map["outer_wall_bottom"]
    top = wall_map["outer_wall_top"]
    left = wall_map["outer_wall_left"]
    right = wall_map["outer_wall_right"]

    xmin = left.pose.x + left.geometry.size_x * 0.5
    xmax = right.pose.x - right.geometry.size_x * 0.5
    ymin = bottom.pose.y + bottom.geometry.size_y * 0.5
    ymax = top.pose.y - top.geometry.size_y * 0.5

    return AARect(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def compute_outer_domain_from_walls(models: list[Model]) -> AARect:
    walls = [m for m in models if m.type == "wall"]
    if len(walls) != 4:
        raise ValueError(f"Expected exactly 4 walls, but got {len(walls)}")

    xmin = min(w.pose.x - w.geometry.size_x * 0.5 for w in walls if isinstance(w.geometry, Box))
    xmax = max(w.pose.x + w.geometry.size_x * 0.5 for w in walls if isinstance(w.geometry, Box))
    ymin = min(w.pose.y - w.geometry.size_y * 0.5 for w in walls if isinstance(w.geometry, Box))
    ymax = max(w.pose.y + w.geometry.size_y * 0.5 for w in walls if isinstance(w.geometry, Box))

    return AARect(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
