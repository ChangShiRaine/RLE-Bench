#!/bin/sh
set -eu
python - <<'PYTHON'
import importlib
import os
from pathlib import Path
from harness.client import TabletopClient, RemoteError

assert os.geteuid() != 0
for name in ("/opt/private/harness/tabletop/scenes.py", "/opt/src/robosuite/robosuite/__init__.py",
             "/var/lib/rlebench/cost.jsonl", "/proc/1/environ", "/proc/1/mem"):
    try:
        Path(name).read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError(f"private file readable: {name}")
assert not os.access("/var/lib/rlebench", os.W_OK)
assert not os.access("/opt/rlebench", os.W_OK)
assert not Path("/opt/HARNESS_MANUAL.md").exists()
assert not Path("/run/rlebench/perception.sock").exists()
for name in ("harness.skills", "harness.privileged", "harness.tabletop", "robosuite"):
    try:
        importlib.import_module(name)
    except (ImportError, PermissionError):
        pass
    else:
        raise AssertionError(f"private or auxiliary module importable: {name}")
with TabletopClient() as sim:
    for op, args in (("observe", {"depth": True}), ("observe", {"level": "L3"}),
                     ("observe", {"cameras": ["private"]}), ("get_state", {}),
                     ("open_evaluation", {}), ("reset", {"seed": 7}), ("seal", {})):
        try:
            sim._request(op, **args)
        except RemoteError:
            pass
        else:
            raise AssertionError(f"forbidden request accepted: {op} {args}")
with TabletopClient() as sim:
    obs = sim.observe()
    extra = {"cube_positions", "pan_positions"} if obs["task"] == "BalanceCoins" else set()
    assert set(obs) - extra == {"images", "joint_pos", "joint_vel", "eef_pos", "eef_quat",
                        "gripper_pos", "base_pos", "base_quat", "eef_base_pos", "eef_base_quat",
                        "task", "steps_used", "steps_remaining", "interaction_budget", "done", "ended"}
    if extra:
        assert set(obs["cube_positions"]) == {f"coin_{i}" for i in range(9)}
        assert set(obs["pan_positions"]) == {"left", "right"}
        assert all(len(p) == 3 for key in extra for p in obs[key].values())
    assert set(obs["images"]) == {"left", "right", "wrist"}
    assert all(frame.shape == (512, 512, 3) for frame in obs["images"].values())
print("isolation ok: authorized observations only; private files remain inaccessible")
PYTHON
