# ppm_builder

## Purpose
This repository provides a minimal, reusable "submodule-like" Python package that builds Polygonal Proximity Map (PPM) cells from:
- sensor poses (nodes of a proximity-point pose graph), and
- proximity points observed at each node.

It supports:
- 2D and 3D PPM construction in *halfspace form* (A x <= b)
- optional extraction of *exact* vertices (polygon in 2D, polyhedron vertices in 3D) as a separate export layer

PPM definition implemented here matches the provided specification:
- per-node free region = Voronoi halfspaces ∩ proximity halfspaces (optionally ∩ AABB)
- global free space = union of non-overlapping convex regions

## Quick start
Example test code:  
-----------
```python
import numpy as np
from ppm_builder import PPMBuilder2D, PPMBuilder3D, SensorPose, ProximityPointSet

# Example: 3D
sensors = [
    SensorPose(id=0, position=np.array([0.0, 0.0, 0.0])),
    SensorPose(id=1, position=np.array([1.0, 0.0, 0.0])),
    SensorPose(id=2, position=np.array([0.0, 1.0, 0.0])),
    SensorPose(id=3, position=np.array([0.0, 0.0, 1.0])),
]

prox = ProximityPointSet.from_dict({
    0: [(2.0, 0.0, 0.0)],
    1: [(2.0, 1.0, 0.0)],
    2: [(0.0, 2.0, 0.0)],
    3: [(0.0, 0.0, 2.0)],
})

builder = PPMBuilder3D()
cells = builder.build_halfspaces(
    sensors=sensors,
    prox_points=prox,
    bounds_min=(-5, -5, -5),
    bounds_max=( 5,  5,  5),
    eps=1e-3,
    include_aabb=True,
)
print(cells[0].halfspaces.A.shape, cells[0].halfspaces.b.shape)
```

## Notes
------------
- Core geometry output is always halfspaces (A, b). This is the stable interface across 2D/3D.
- Exact vertex extraction is intentionally separated into `ppm_builder.export` to keep the builder light.
- Voronoi constraints are computed from Delaunay neighbors (2D triangles / 3D tetrahedra) to avoid full Voronoi construction.
