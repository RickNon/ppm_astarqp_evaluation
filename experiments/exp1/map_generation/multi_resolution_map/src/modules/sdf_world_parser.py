import xml.etree.ElementTree as ET
from .models import Model, Pose, Box, Cylinder


def parse_world(path, config):
    tree = ET.parse(path)
    root = tree.getroot()

    models = []

    for m in root.findall(".//model"):
        name = m.get("name")

        # ignore
        if name in config.ignore_models:
            continue

        # pose
        pose_text = m.findtext("pose", default="0 0 0 0 0 0").split()
        x = float(pose_text[0])
        y = float(pose_text[1])
        yaw = float(pose_text[5])

        # collision
        collision = m.find(".//collision")
        if collision is None:
            continue

        geom = None

        box = collision.find(".//box")
        if box is not None:
            size = list(map(float, box.findtext("size").split()))
            geom = Box(size_x=size[0], size_y=size[1])

        cylinder = collision.find(".//cylinder")
        if cylinder is not None:
            radius = float(cylinder.findtext("radius"))
            geom = Cylinder(radius=radius)

        if geom is None:
            continue

        if any(name.startswith(p) for p in config.wall_prefixes):
            model_type = "wall"
        elif any(name.startswith(p) for p in config.obstacle_prefixes):
            model_type = "obstacle"
        else:
            continue

        models.append(
            Model(
                name=name,
                pose=Pose(x=x, y=y, yaw=yaw),
                geometry=geom,
                type=model_type,
            )
        )

    return models
