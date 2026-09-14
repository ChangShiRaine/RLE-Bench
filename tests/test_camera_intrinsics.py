"""The published lens numbers must be the simulator's own.

task01 and task02 hand the agent `fovy` per camera and the pinhole formula, because a
depth map without a focal length is not a metric measurement. That makes the numbers part
of the observation contract: if RoboCasa's camera config moves and the instructions do
not, every agent deprojects onto a lie and nothing says so.

Read from the vendored sources rather than by importing robocasa -- it pins mujoco 3.3.1,
which conflicts with the repo-wide requirements, so it lives in .venv-robocasa and is not
importable here. Skips when third_party/ has not been populated.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CAMERA_UTILS = REPO / "third_party/robocasa/robocasa/utils/camera_utils.py"
KITCHEN = REPO / "third_party/robocasa/robocasa/environments/kitchen/kitchen.py"
PANDA_XML = REPO / "third_party/robosuite/robosuite/models/assets/robots/panda/robot.xml"

needs_vendor = pytest.mark.skipif(
    not CAMERA_UTILS.is_file(), reason="third_party/ not populated (make sim-robocasa)"
)


def _as_dict(node) -> dict:
    """A `dict(fovy="60")` call or a `{...}` literal, whichever upstream wrote."""
    if node is None:
        return {}
    if isinstance(node, ast.Call):
        return {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    return ast.literal_eval(node)


def _cam_configs_default() -> dict:
    """CAM_CONFIGS["DEFAULT"], evaluated as a literal without importing robocasa.

    The quaternions are computed by a helper call, which ast.literal_eval cannot take, so
    only the `camera_attribs` sub-dict of each camera is read -- which is where fovy is.
    """
    tree = ast.parse(CAMERA_UTILS.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == "CAM_CONFIGS" for t in node.targets):
            continue
        default = node.value.keywords[0]
        assert default.arg == "DEFAULT", default.arg
        out = {}
        for cam in default.value.keywords:
            attribs = next((k.value for k in cam.value.keywords
                            if k.arg == "camera_attribs"), None)
            out[cam.arg] = _as_dict(attribs)
        return out
    raise AssertionError("CAM_CONFIGS not found")


def _kitchen_default(name: str) -> bool:
    """One keyword default off Kitchen.__init__ -- again without importing."""
    tree = ast.parse(KITCHEN.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        args = node.args
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if arg.arg == name and default is not None:
                return ast.literal_eval(default)
        offset = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args[offset:]):
            if arg.arg == name:
                return ast.literal_eval(args.defaults[i])
    raise AssertionError(f"{name} default not found in a Kitchen __init__")


@needs_vendor
def test_agentview_cameras_are_60_degrees():
    cams = _cam_configs_default()
    for name in ("robot0_agentview_left", "robot0_agentview_right"):
        assert cams[name]["fovy"] == "60", f"{name} moved to {cams[name]}"


@needs_vendor
def test_the_wrist_camera_is_75_degrees():
    """robot0_eye_in_hand carries no camera_attribs, so it inherits the robot XML's."""
    cams = _cam_configs_default()
    assert "fovy" not in cams["robot0_eye_in_hand"], (
        "eye_in_hand now overrides fovy in CAM_CONFIGS; the instructions say 75")
    xml = PANDA_XML.read_text()
    match = re.search(r'name="eye_in_hand"[^>]*fovy="([0-9.]+)"', xml)
    assert match and match.group(1) == "75", match


@needs_vendor
def test_the_published_numbers_are_the_stable_ones():
    """Both randomisers must stay off: the instructions publish ONE number per camera.

    `randomize_cameras` perturbs pose and `use_cotraining_cameras` selects a different
    config entirely -- COTRAIN_CAM_CONFIGS puts the agentview cameras at fovy 65. Neither
    is overridden by tasks/{task01,task02}/harness/env.py, so both defaults are what the
    agent is actually rendered through.
    """
    assert _kitchen_default("randomize_cameras") is False
    assert _kitchen_default("use_cotraining_cameras") is False
