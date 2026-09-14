"""Let RoboCasa run against a READ-ONLY asset tree.

WHY THIS EXISTS
---------------
RLE-Bench ships the RoboCasa asset tree as a versioned read-only dataset (a zstd
SquashFS mounted over models/assets) rather than as image content. The premise
"RoboCasa only reads these assets" turns out to be FALSE:

    robocasa/models/objects/objects.py:245-265  (MJCFObject.__init__)
        xml_str = self.postprocess_model_xml(xml_str)
        time_str = str(time.time()).replace(".", "_")
        new_xml_path = os.path.join(folder, "{}_{}.xml".format(time_str, os.getpid()))
        f = open(new_xml_path, "w")      # <-- writes INTO the asset tree
        ...
        os.remove(new_xml_path)          # <-- transient; deleted immediately

Every object instantiation writes a transient MJCF next to the object's source
XML. Against a read-only mount that raises:

    OSError: [Errno 30] Read-only file system:
        '.../objects/lightwheel/tongs/Tongs001/1785369596_2329464_916.xml'

It writes in-place because `postprocess_model_xml` only absolutizes paths
containing "robosuite"; robocasa's own mesh/texture `file` attributes stay
RELATIVE and so resolve against the XML's own directory. The scratch file
therefore has to sit beside its assets -- unless the paths are absolutized first,
which is what this patch does.

The filename embeds `time.time()` and the pid, so the file is NOT cacheable and
cannot be pre-generated into the dataset.

WHAT THIS CHANGES
-----------------
Exactly two things, both about *where a scratch file lives*, never about
simulation semantics:
  1. relative mesh/texture paths are rewritten to absolute paths pointing back
     into the read-only tree;
  2. the scratch MJCF is written to a writable temp dir.
The resulting MJCF is semantically identical, so the RoboCasa test protocol is
unaffected.

Alternatives considered and rejected:
  - overlayfs / fuse-overlayfs union over the mount: no code change, but needs
    CAP_SYS_ADMIN or /dev/fuse in every trial container, which Harbor's task.toml
    cannot express;
  - a writable shared bind mount: abandons immutability, lets one trial mutate
    the dataset other trials read, and races on the time+pid filename.

Call install() once before constructing any RoboCasa env. Forgetting it fails
loudly with the OSError above, never silently.
"""

from __future__ import annotations

import os
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

from robocasa.models.objects import objects as rc_objects

_installed = False
_scratch_dir: str | None = None


def _absolutize(xml_str: str, folder: str) -> str:
    """Rewrite relative mesh/texture file paths to absolute paths under `folder`.

    Needed because the scratch MJCF is written away from its assets.
    """
    root = ET.fromstring(xml_str)
    asset = root.find("asset")
    if asset is None:
        return xml_str
    for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
        path = elem.get("file")
        if path is None or os.path.isabs(path):
            continue
        elem.set("file", os.path.normpath(os.path.join(folder, path)))
    return ET.tostring(root, encoding="utf8").decode("utf8")


def _patched_init(
    self,
    name,
    mjcf_path,
    scale=1.0,
    solimp=(0.998, 0.998, 0.001),
    solref=(0.001, 1),
    density=100,
    friction=(0.95, 0.3, 0.1),
    margin=None,
    rgba=None,
    priority=None,
):
    # Faithful mirror of robocasa MJCFObject.__init__ (objects.py:204-268). Keep in
    # sync when bumping ROBOCASA_SHA -- the two intentional deviations are marked.
    if isinstance(scale, float):
        scale = [scale, scale, scale]
    elif isinstance(scale, (tuple, list)):
        assert len(scale) == 3
        scale = tuple(scale)
    else:
        raise Exception("got invalid scale: {}".format(scale))
    scale = np.array(scale)

    self.solimp = solimp
    self.solref = solref
    self.density = density
    self.friction = friction
    self.margin = margin
    self.priority = priority
    self.rgba = rgba
    self.mjcf_path = mjcf_path

    folder = os.path.dirname(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
    xml_str = self.postprocess_model_xml(xml_str)
    xml_str = _absolutize(xml_str, folder)  # DEVIATION 1

    time_str = str(time.time()).replace(".", "_")
    # DEVIATION 2: scratch dir instead of `folder` (which is read-only).
    new_xml_path = os.path.join(
        _scratch_dir, "{}_{}.xml".format(time_str, os.getpid())
    )
    with open(new_xml_path, "w") as f:
        f.write(xml_str)

    # Base class is MujocoXMLObjectRobocasa, not robosuite's MujocoXMLObject.
    rc_objects.MujocoXMLObjectRobocasa.__init__(
        self,
        fname=new_xml_path,
        name=name,
        joints=[dict(type="free", damping="0.0005")],
        obj_type="all",
        duplicate_collision_geoms=False,
        scale=scale,
    )

    if os.path.exists(new_xml_path):
        os.remove(new_xml_path)

    # Without these the object has no _regions and placement sampling dies on
    # mj_obj.size (objects.py:438).
    self._regions = dict()
    self._setup_region_dict()


def install(scratch_dir: str | None = None) -> str:
    """Patch MJCFObject so its scratch MJCF lands in a writable directory.

    Idempotent. Returns the scratch directory in use.
    """
    global _installed, _scratch_dir
    if _installed:
        return _scratch_dir  # type: ignore[return-value]

    _scratch_dir = scratch_dir or tempfile.mkdtemp(prefix="robocasa-mjcf-")
    os.makedirs(_scratch_dir, exist_ok=True)
    rc_objects.MJCFObject.__init__ = _patched_init
    _installed = True
    return _scratch_dir
