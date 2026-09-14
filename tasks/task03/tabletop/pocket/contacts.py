"""Close-fitting collision shells; visual meshes and passive joints are retained."""

import numpy as np
import xml.etree.ElementTree as ET

HALF_SIZE = 0.00945
SOLREF = "0.004 1"
SOLIMP = "0.999 0.9999 0.001 0.5 2"
SOLMIX = "1000"


def reinforce(root):
    for joint in root.iter("joint"):
        if joint.get("name", "").startswith("cube_") and joint.get("type") != "free":
            joint.set("frictionloss", "0.00005")
            joint.set("armature", "0.000005")
    for geom in root.iter("geom"):
        if geom.get("name", "").startswith("cube_geom") and geom.get("group") == "0" and geom.get("type") == "mesh":
            geom.set("type", "box")
            geom.attrib.pop("mesh")
            geom.set("size", " ".join([str(HALF_SIZE)] * 3))
            geom.set("solref", SOLREF)
            geom.set("solimp", SOLIMP)
            # Dominate softness mixing without overriding the finger's friction dimensions.
            geom.set("solmix", SOLMIX)


def add_handles(root):
    for name, axis, sign, rgba in (
        ("pX", 0, 1, ".8 .05 .03 1"), ("nX", 0, -1, "1 .35 .02 1"),
        ("pY", 1, 1, ".02 .15 .8 1"), ("nY", 1, -1, ".02 .55 .12 1"),
        ("pZ", 2, 1, ".95 .95 .95 1"), ("nZ", 2, -1, ".95 .8 .02 1"),
    ):
        body = root.find(f".//body[@name='cube_{name}']")
        position = np.zeros(3)
        position[axis] = sign * .0555
        size = np.full(3, .007)
        size[axis] = .027
        for group in (0, 1):
            ET.SubElement(body, "geom", name=f"cube_handle_{name}_{group}", type="box",
                          pos=" ".join(map(str, position)), size=" ".join(map(str, size)),
                          group=str(group), mass=".0015" if group == 0 else "0", rgba=rgba,
                          contype="1" if group == 0 else "0", conaffinity="1" if group == 0 else "0",
                          condim="4", friction="4 .02 .0001", solref=SOLREF, solimp=SOLIMP,
                          solmix=SOLMIX)


def fork_support(world):
    """Leave the center shaft and approaching fingertips clear of the support."""
    world.find("geom[@name='pedestal_pad']").attrib.update(pos="0 0 .933", size=".026 .026 .002")
    world.find("geom[@name='pedestal']").attrib.update(pos="0 0 .8435", size=".006 .0875")
    for x in (-.019, .019):
        for y in (-.019, .019):
            ET.SubElement(world, "geom", name=f"support_{x}_{y}", type="cylinder", size=".004 .03825",
                          pos=f"{x} {y} .97325", rgba=".1 .1 .1 1", group="1")
