"""Render the development-only hidden center-of-mass scene with robosuite."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import robosuite

ROOT = Path(__file__).resolve().parents[4]
import sys
sys.path.insert(0, str(ROOT / "tasks/task03"))
from tabletop.hidden_com.scene import CAMERAS, HALF, HiddenCOM, mass_properties


def export(env, output):
    root = ET.fromstring(env.model.get_xml())
    root.find("compiler").set("meshdir", ".")
    assets = output / "assets"; assets.mkdir(exist_ok=True)
    for node in root.findall("asset/*[@file]"):
        source = Path(node.get("file"))
        target = assets / (hashlib.sha256(source.read_bytes()).hexdigest()[:16]+source.suffix)
        shutil.copy2(source, target)
        node.set("file", target.relative_to(output).as_posix())
    shutil.copy2(Path(robosuite.__file__).resolve().parents[1] / "LICENSE", assets / "ROBOSUITE_LICENSE")
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="preview",
        qpos=" ".join(map(str, env.sim.data.qpos)), ctrl=" ".join(map(str, env.sim.data.ctrl)))
    ET.indent(root)
    ET.ElementTree(root).write(output / "scene.xml", encoding="unicode")
    mujoco.MjModel.from_xml_path(str(output / "scene.xml"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/task03/hidden_com")
    parser.add_argument("--quadrant", choices="ABCD", default="B")
    args = parser.parse_args()
    pin = re.search(r'mujoco==([\d.]+)', (ROOT / "sim/robocasa/base/Dockerfile").read_text()).group(1)
    assert mujoco.__version__ == pin, (mujoco.__version__, pin)
    args.output.mkdir(parents=True, exist_ok=True)
    env = HiddenCOM(args.quadrant, resolution=768, camera_observations=True)
    try:
        obs = env.reset()
        sheet = Image.new("RGB", (2304, 824), "#edf0f4")
        draw = ImageDraw.Draw(sheet)
        for i, camera in enumerate(CAMERAS):
            pixels = obs[camera+"_image"]
            if robosuite.macros.IMAGE_CONVENTION == "opengl": pixels = pixels[::-1]
            view = Image.fromarray(pixels)
            view.save(args.output / f"{camera}.png")
            sheet.paste(view, (768*i, 56))
            draw.text((768*i+20, 18), camera.upper(), fill="#17212d")
        sheet.save(args.output / "preview.png")
        export(env, args.output)
        model = env.sim.model._model
        mass, center, _ = mass_properties(args.quadrant)
        np.testing.assert_allclose(model.body(env.box.root_body).mass, [mass])
        np.testing.assert_allclose(model.body(env.box.root_body).ipos, center, atol=1e-12)
        states = []
        for _ in range(2):
            env.reset()
            for _ in range(20): env.step(np.zeros(env.action_dim))
            states.append(env.sim.data.qpos.copy())
        np.testing.assert_array_equal(*states)
        assert np.isfinite(states[0]).all()
        assert not any(w.number for w in env.sim.data._data.warning)
        report = {"mujoco": mujoco.__version__, "action_dim": env.action_dim,
            "box_size_m": list(2*HALF), "mass_kg": mass,
            "repeat_rollout_identical": True, "smoke_seconds": 1,
            "status": "Deterministic scene preview with rigid grasp handle."}
        (args.output / "checks.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
