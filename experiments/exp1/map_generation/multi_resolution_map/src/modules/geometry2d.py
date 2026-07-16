import math
from dataclasses import dataclass


@dataclass
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y


@dataclass
class AARect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center(self) -> Vec2:
        return Vec2((self.xmin + self.xmax) * 0.5, (self.ymin + self.ymax) * 0.5)

    def corners(self) -> list[Vec2]:
        return [
            Vec2(self.xmin, self.ymin),
            Vec2(self.xmax, self.ymin),
            Vec2(self.xmax, self.ymax),
            Vec2(self.xmin, self.ymax),
        ]

    def contains_point(self, p: Vec2) -> bool:
        return self.xmin <= p.x <= self.xmax and self.ymin <= p.y <= self.ymax

    def intersects(self, other: "AARect") -> bool:
        return not (
            self.xmax < other.xmin or
            self.xmin > other.xmax or
            self.ymax < other.ymin or
            self.ymin > other.ymax
        )


@dataclass
class OrientedRect:
    center: Vec2
    width: float
    height: float
    yaw: float

    def half_extents(self) -> tuple[float, float]:
        return self.width * 0.5, self.height * 0.5

    def local_corners(self) -> list[Vec2]:
        hx, hy = self.half_extents()
        return [
            Vec2(-hx, -hy),
            Vec2(hx, -hy),
            Vec2(hx, hy),
            Vec2(-hx, hy),
        ]

    def world_corners(self) -> list[Vec2]:
        return [transform_point(c, self.center, self.yaw) for c in self.local_corners()]

    def aabb(self) -> AARect:
        pts = self.world_corners()
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        return AARect(min(xs), min(ys), max(xs), max(ys))

    def contains_point(self, p: Vec2) -> bool:
        # Transform the point into the rectangle local frame.
        pl = inverse_transform_point(p, self.center, self.yaw)
        hx, hy = self.half_extents()
        return (-hx <= pl.x <= hx) and (-hy <= pl.y <= hy)


@dataclass
class Circle:
    center: Vec2
    radius: float

    def aabb(self) -> AARect:
        return AARect(
            self.center.x - self.radius,
            self.center.y - self.radius,
            self.center.x + self.radius,
            self.center.y + self.radius,
        )

    def contains_point(self, p: Vec2) -> bool:
        dx = p.x - self.center.x
        dy = p.y - self.center.y
        return dx * dx + dy * dy <= self.radius * self.radius


def rotate(v: Vec2, yaw: float) -> Vec2:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return Vec2(c * v.x - s * v.y, s * v.x + c * v.y)


def transform_point(local_point: Vec2, translation: Vec2, yaw: float) -> Vec2:
    # Apply 2D rigid transform.
    return rotate(local_point, yaw) + translation


def inverse_transform_point(world_point: Vec2, translation: Vec2, yaw: float) -> Vec2:
    # Apply inverse 2D rigid transform.
    shifted = world_point - translation
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    return Vec2(c * shifted.x - s * shifted.y, s * shifted.x + c * shifted.y)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def point_in_aarect(p: Vec2, rect: AARect) -> bool:
    return rect.contains_point(p)


def segments_intersect(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2) -> bool:
    # Robust enough for current geometric use.
    def orient(a: Vec2, b: Vec2, c: Vec2) -> float:
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    def on_segment(a: Vec2, b: Vec2, c: Vec2) -> bool:
        return (
            min(a.x, b.x) <= c.x <= max(a.x, b.x) and
            min(a.y, b.y) <= c.y <= max(a.y, b.y)
        )

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
        return True

    eps = 1e-12
    if abs(o1) < eps and on_segment(p1, p2, q1):
        return True
    if abs(o2) < eps and on_segment(p1, p2, q2):
        return True
    if abs(o3) < eps and on_segment(q1, q2, p1):
        return True
    if abs(o4) < eps and on_segment(q1, q2, p2):
        return True

    return False


def oriented_rect_intersects_aarect(orect: OrientedRect, rect: AARect) -> bool:
    # First reject with AABB.
    if not orect.aabb().intersects(rect):
        return False

    rect_corners = rect.corners()
    orect_corners = orect.world_corners()

    # If any axis-aligned rect corner is inside the oriented rect, they intersect.
    for p in rect_corners:
        if orect.contains_point(p):
            return True

    # If any oriented rect corner is inside the axis-aligned rect, they intersect.
    for p in orect_corners:
        if rect.contains_point(p):
            return True

    # Check edge-edge intersections.
    rect_edges = [
        (rect_corners[0], rect_corners[1]),
        (rect_corners[1], rect_corners[2]),
        (rect_corners[2], rect_corners[3]),
        (rect_corners[3], rect_corners[0]),
    ]
    orect_edges = [
        (orect_corners[0], orect_corners[1]),
        (orect_corners[1], orect_corners[2]),
        (orect_corners[2], orect_corners[3]),
        (orect_corners[3], orect_corners[0]),
    ]

    for e1 in rect_edges:
        for e2 in orect_edges:
            if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                return True

    return False


def circle_intersects_aarect(circle: Circle, rect: AARect) -> bool:
    nearest_x = clamp(circle.center.x, rect.xmin, rect.xmax)
    nearest_y = clamp(circle.center.y, rect.ymin, rect.ymax)
    dx = circle.center.x - nearest_x
    dy = circle.center.y - nearest_y
    return dx * dx + dy * dy <= circle.radius * circle.radius


def aarect_fully_inside_oriented_rect(rect: AARect, orect: OrientedRect) -> bool:
    # All corners must be inside.
    return all(orect.contains_point(p) for p in rect.corners())


def aarect_fully_inside_circle(rect: AARect, circle: Circle) -> bool:
    # All corners must be inside.
    return all(circle.contains_point(p) for p in rect.corners())
