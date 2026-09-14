"""Original, unbranded servo visuals with independently pinned inertial proxies.

The mesh generator uses only elementary solids and mating dimensions (mm);
it never reads manufacturer CAD. The physics constants are the pinned
effective mass properties in SI units, not hardware measurements.
"""
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


PHYSICS = {
    "servo": dict(
        pos=(-5.5724486651656165e-05, -0.00649093067373353, -0.008025253645528377),
        quat=(0.7191351688127579, 0.6948656374882143, -0.0018128561965194314, -0.0017516755281883894),
        inertia=(2.2970844932020605e-06, 1.934384838820217e-06, 1.3011166323711306e-06),
        rbound=0.025842547504309088),
    "servo_single": dict(
        pos=(-5.825471686744567e-05, -0.0069680954184237805, -0.007253844813837991),
        quat=(0.6768233950023156, 0.7361415688199183, -0.00161343471233371, -0.0017548393999025217),
        inertia=(2.266708561112053e-06, 2.0314298814489927e-06, 1.2302561378350526e-06),
        rbound=0.025212611802907736),
}
MASS = 0.018


def _prism(polygon, low, high):
    """Closed CCW polygon extrusion, with outward triangle winding."""
    bottom = np.column_stack([polygon, np.full(len(polygon), low)])
    top = np.column_stack([polygon, np.full(len(polygon), high)])
    triangles = []
    for i in range(1, len(polygon)-1):
        triangles.extend(((bottom[0], bottom[i+1], bottom[i]),
                          (top[0], top[i], top[i+1])))
    for i in range(len(polygon)):
        j = (i+1) % len(polygon)
        triangles.extend(((bottom[i], bottom[j], top[j]),
                          (bottom[i], top[j], top[i])))
    return triangles


def triangles(dual_horn):
    # Chamfered 20 x 34 mm case; small seams separate the end covers.
    outline = np.array([[-8.5, -24.5], [8.5, -24.5], [10, -23],
                        [10, 8], [8.5, 9.5], [-8.5, 9.5],
                        [-10, 8], [-10, -23]])
    faces = []
    for low, high in ((-19.5, -18.5), (-18.35, 2.35), (2.5, 3.5)):
        faces.extend(_prism(outline, low, high))
    angle = np.arange(32) * (2*np.pi/32)
    circle = np.column_stack([np.cos(angle), np.sin(angle)])
    faces.extend(_prism(circle*7, 3.5, 6.5))
    if dual_horn:
        faces.extend(_prism(circle*7, -22.5, -19.5))
    for x in (-8, 8):
        for y in (-22.5, 7.5):
            faces.extend(_prism(circle*1.2 + [x, y], 3.5, 3.9))
    return np.asarray(faces, dtype=np.float32)


def write_meshes(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in PHYSICS:
        faces = triangles(name == "servo")
        records = np.zeros(len(faces), dtype=[("normal", "<f4", 3),
                           ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        records["vertices"] = faces
        normals = np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])
        records["normal"] = normals / np.linalg.norm(normals, axis=1)[:, None]
        header = b"RLE-Bench original generic micro-servo; Apache-2.0".ljust(80, b"\0")
        (directory/f"{name}.stl").write_bytes(
            header + struct.pack("<I", len(faces)) + records.tobytes())


def _vec(values):
    return " ".join(format(float(v), ".17g") for v in values)


def separate_visuals(root):
    """Keep body topology and equivalent box inertia; visuals have zero mass."""
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            if not geom.get("name", "").startswith("servo"):
                continue
            visual = ET.fromstring(ET.tostring(geom))
            visual.set("name", "visual_" + geom.get("name"))
            visual.set("mass", "0")
            body.append(visual)
            physics = PHYSICS[geom.attrib.pop("mesh")]
            rotation = np.zeros(9)
            quat = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
            mujoco.mju_quat2Mat(rotation, quat)
            pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            geom.set("pos", _vec(pos + rotation.reshape(3, 3) @ physics["pos"]))
            result = np.zeros(4)
            mujoco.mju_mulQuat(result, quat, np.asarray(physics["quat"]))
            geom.set("quat", _vec(result))
            inertia = np.asarray(physics["inertia"])
            # Ixx = m*(hy^2+hz^2)/3 for a uniform box with half-sizes h.
            halfsize = np.sqrt(3/(2*MASS) * (inertia.sum()-2*inertia))
            geom.set("type", "box")
            geom.set("size", _vec(halfsize))
            geom.set("rgba", "0 0 0 0")
            geom.set("group", "3")
