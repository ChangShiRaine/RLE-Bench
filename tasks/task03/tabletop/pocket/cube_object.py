"""Adapt the vendored cube asset for the pocket-cube model builder."""
from pathlib import Path
from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.models.objects import MujocoObject
from robosuite.utils.mjcf_utils import add_prefix

CUBE_POSITION = np.array([0.0, 0.0, 1.04])


class PocketCubeObject(MujocoObject):
    """Adapt the upstream articulated cube to robosuite's object conventions."""

    def __init__(self, xml_path: Path):
        super().__init__(duplicate_collision_geoms=False)
        self._name = "cube"
        source = ET.parse(xml_path).getroot()
        defaults = source.find("default")
        base = {node.tag: dict(node.attrib) for node in defaults if node.tag != "default"}
        classes = {}
        for cls in defaults.findall("default"):
            classes[cls.get("class")] = {key: value.copy() for key, value in base.items()}
            for node in cls:
                classes[cls.get("class")].setdefault(node.tag, {}).update(node.attrib)
        self._obj = source.find("worldbody/body")

        def inline(node, inherited=None):
            childclass = node.attrib.pop("childclass", inherited)
            cls = node.attrib.pop("class", childclass)
            attributes = classes.get(cls, base).get(node.tag, {}).copy()
            attributes.update(node.attrib)
            node.attrib = attributes
            if "euler" in attributes:
                angles = np.deg2rad(np.fromstring(attributes["euler"], sep=" "))
                node.set("euler", " ".join(map(str, angles)))
            for child in node:
                inline(child, childclass)

        inline(self._obj)
        for i, geom in enumerate(self._obj.iter("geom")):
            geom.set("name", f"geom{i}")
            geom.set("group", "0")
        for body in self._obj.iter("body"):
            for geom in list(body.findall("geom")):
                if geom.get("type") != "sphere":
                    visual = deepcopy(geom)
                    visual.attrib.update(name=geom.get("name") + "_visual", group="1",
                                         contype="0", conaffinity="0", mass="0")
                    body.append(visual)
        ET.SubElement(self._obj, "joint", name="free", type="free")
        self._obj.set("pos", " ".join(map(str, CUBE_POSITION)))
        for node in source.find("asset"):
            if node.get("type") == "skybox":
                continue
            if "file" in node.attrib:
                filename = node.get("file")
                node.set("name", node.get("name", Path(filename).stem))
                node.set("file", str(xml_path.parent / "assets" / filename))
            self.asset.append(node)
        self._get_object_properties()
        add_prefix(self.asset, self.naming_prefix)

    def exclude_from_prefixing(self, inp):
        return False

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.0285])

    @property
    def top_offset(self):
        return -self.bottom_offset

    @property
    def horizontal_radius(self):
        return np.sqrt(2) * 0.0285

