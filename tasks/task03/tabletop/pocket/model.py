"""Eight rigid corner cubies; no edges, centers, shafts or cube actuators."""

import xml.etree.ElementTree as ET
import numpy as np
from . import contacts
from .metrics import POSITIONS

GAP = .00005
OUTER = .032
STICKER_MARGIN = .001
CORNER_MASS = .008
PAD_FRICTION = "2 0.006 0.0001"
COLORS = {(0, 1): ".8 .05 .03 1", (0, -1): "1 .35 .02 1",
          (1, 1): ".02 .15 .8 1", (1, -1): ".02 .55 .12 1",
          (2, 1): ".95 .95 .95 1", (2, -1): ".95 .8 .02 1"}


def configure_grippers(root, profile):
    if profile not in ("baseline", "torsion", "elliptic", "wide"):
        raise ValueError(f"Unknown gripper profile: {profile}")
    if profile == "baseline":
        return
    for geom in root.iter("geom"):
        if "pad_collision" in geom.get("name", ""):
            geom.set("condim", "4")
            geom.set("friction", PAD_FRICTION)
            if profile == "wide":
                geom.set("size", "0.012 0.004 0.012")
    if profile == "elliptic":
        root.find("option").attrib.update(cone="elliptic", impratio="5")


def box(body, name, lo, hi, rgba, mass=0):
    lo, hi = np.asarray(lo), np.asarray(hi)
    attrs = dict(type="box", pos=" ".join(map(str, (lo+hi)/2)),
                 size=" ".join(map(str, (hi-lo)/2)), rgba=rgba)
    if mass:
        ET.SubElement(body, "geom", name=name, group="0", mass=str(mass),
                      condim="1", friction="0 0 0", solref=contacts.SOLREF,
                      solimp=contacts.SOLIMP, solmix=contacts.SOLMIX, **attrs)
    ET.SubElement(body, "geom", name=name+"_visual", group="1", mass="0",
                  contype="0", conaffinity="0", **attrs)


def convert(root, obj):
    core = root.find(".//body[@name='cube_core']")
    for body in list(core.findall("body")):
        position = np.fromstring(body.find("geom").get("pos"), sep=" ")
        if np.count_nonzero(position) != 3:
            core.remove(body)
            continue
        sign = np.sign(position).astype(int)
        index = next(i for i, p in enumerate(POSITIONS) if np.array_equal(p, sign))
        body.set("name", f"cube_pocket_corner_{index}")
        for geom in list(body.findall("geom")):
            body.remove(geom)
        ends = np.array([np.full(3, GAP)*sign, np.full(3, OUTER)*sign])
        box(body, f"cube_pocket_{index}", ends.min(axis=0), ends.max(axis=0),
            ".015 .015 .015 1", CORNER_MASS)
        for axis in range(3):
            lo, hi = np.full(3, STICKER_MARGIN), np.full(3, OUTER-STICKER_MARGIN)
            lo[axis], hi[axis] = OUTER, OUTER+.0001
            ends = np.array([lo*sign, hi*sign])
            box(body, f"cube_sticker_{index}_{axis}", ends.min(axis=0), ends.max(axis=0),
                COLORS[axis, int(sign[axis])])
    for geom in root.find("worldbody").findall("geom"):
        if geom.get("name", "").startswith("support_"):
            p = np.fromstring(geom.get("pos"), sep=" ")
            top, bottom = 1.04-OUTER, .935
            p[2] = (top+bottom)/2
            geom.attrib.update(pos=" ".join(map(str, p)), size=f".004 {(top-bottom)/2}")
    # The enclosing robosuite task registers these names after model compilation.
    for attr, group in (("_contact_geoms", "0"), ("_visual_geoms", "1")):
        setattr(obj, attr, [g.get("name").removeprefix("cube_")
                           for g in core.iter("geom") if g.get("group") == group])
    obj._bodies = [b.get("name").removeprefix("cube_") for b in core.iter("body")]
    obj._joints = [j.get("name").removeprefix("cube_") for j in core.iter("joint")]
