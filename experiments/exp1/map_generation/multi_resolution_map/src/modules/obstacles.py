from dataclasses import dataclass
from .models import Model, Box, Cylinder
from .geometry2d import (
    Vec2,
    AARect,
    OrientedRect,
    Circle,
    oriented_rect_intersects_aarect,
    circle_intersects_aarect,
    aarect_fully_inside_oriented_rect,
    aarect_fully_inside_circle,
)


class Obstacle2D:
    def intersects_rect(self, rect: AARect) -> bool:
        raise NotImplementedError

    def contains_rect(self, rect: AARect) -> bool:
        raise NotImplementedError

    def bounding_box(self) -> AARect:
        raise NotImplementedError


@dataclass
class BoxObstacle2D(Obstacle2D):
    shape: OrientedRect
    name: str

    def intersects_rect(self, rect: AARect) -> bool:
        return oriented_rect_intersects_aarect(self.shape, rect)

    def contains_rect(self, rect: AARect) -> bool:
        return aarect_fully_inside_oriented_rect(rect, self.shape)

    def bounding_box(self) -> AARect:
        return self.shape.aabb()


@dataclass
class CylinderObstacle2D(Obstacle2D):
    shape: Circle
    name: str

    def intersects_rect(self, rect: AARect) -> bool:
        return circle_intersects_aarect(self.shape, rect)

    def contains_rect(self, rect: AARect) -> bool:
        return aarect_fully_inside_circle(rect, self.shape)

    def bounding_box(self) -> AARect:
        return self.shape.aabb()


def model_to_obstacle(model: Model) -> Obstacle2D:
    center = Vec2(model.pose.x, model.pose.y)

    if isinstance(model.geometry, Box):
        shape = OrientedRect(
            center=center,
            width=model.geometry.size_x,
            height=model.geometry.size_y,
            yaw=model.pose.yaw,
        )
        return BoxObstacle2D(shape=shape, name=model.name)

    if isinstance(model.geometry, Cylinder):
        shape = Circle(
            center=center,
            radius=model.geometry.radius,
        )
        return CylinderObstacle2D(shape=shape, name=model.name)

    raise TypeError(f"Unsupported geometry type: {type(model.geometry)}")


def models_to_obstacles(models: list[Model]) -> list[Obstacle2D]:
    # Convert only actual obstacles. Walls are handled separately by map_domain.
    return [model_to_obstacle(m) for m in models if m.type == "obstacle"]
