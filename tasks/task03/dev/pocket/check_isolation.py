"""Run in a fresh pocket container as agent while its daemon is ready."""
import importlib
import os
from pathlib import Path

import numpy as np
from harness.client import SpeedrunClient, RemoteError
from harness.obs import ObsSpec

assert os.geteuid() != 0
for path in ("/opt/private/harness/tabletop/pocket/recovery.py",
             "/opt/private/harness/tabletop/pocket/assets/cube_3x3x3.xml",
             "/var/lib/rlebench/cost.jsonl", "/var/lib/rlebench/recovery.json",
             "/proc/1/environ", "/proc/1/mem"):
    try:
        Path(path).read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError(f"private file readable: {path}")
for name in ("harness.skills", "harness.tabletop", "robosuite"):
    try:
        importlib.import_module(name)
    except (ImportError, PermissionError):
        pass
    else:
        raise AssertionError(f"private module importable: {name}")
assert not os.access("/var/lib/rlebench", os.W_OK)
robot_fields = {"joint_pos", "joint_pos_cos", "joint_pos_sin", "joint_vel", "joint_acc",
                "eef_pos", "eef_quat", "eef_quat_site", "gripper_qpos", "gripper_qvel",
                "proprio-state"}
allowed = {f"robot{i}_{field}" for i in range(2) for field in robot_fields}
allowed.update(("front_image", "front_depth"))
with SpeedrunClient() as sim:
    before = sim.observe(ObsSpec(cameras=("front",), depth=True))["obs"]
    steps = sim.status()["steps_used"]
for op, args in (("recover_drop", {"seed": 0}), ("get_state", {}),
                 ("open_evaluation", {}), ("seal", {})):
    with SpeedrunClient() as sim:
        try:
            sim._request(op, **args)
        except RemoteError:
            pass
        else:
            raise AssertionError(f"forbidden request accepted: {op}")
with SpeedrunClient() as sim:
    after = sim.observe(ObsSpec(cameras=("front",), depth=True))["obs"]
    assert steps == sim.status()["steps_used"]
    assert set(after) <= allowed
    assert after["front_image"].shape == (512, 512, 3)
    assert after["front_depth"].shape[:2] == (512, 512)
    assert np.array_equal(before["front_image"], after["front_image"])
    for i in range(2):
        assert {f"robot{i}_{field}" for field in ("joint_pos", "eef_pos", "eef_quat")} <= set(after)
print("pocket isolation ok: private files blocked, front RGB/depth and robot state only")
