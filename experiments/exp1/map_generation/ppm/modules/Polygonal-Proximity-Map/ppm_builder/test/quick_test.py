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
