"""Run inside the task container as agent; fails on any privileged read."""
import importlib
import json
from pathlib import Path

from harness.client import HiddenCOMClient

assert __import__("os").geteuid() != 0
for filename in ("/opt/private/harness/tabletop/hidden_com/config.py",
                 "/opt/private/harness/tabletop/hidden_com/scene.py",
                 "/var/lib/rlebench/hidden-com.json", "/var/lib/rlebench/daemon.log",
                 "/var/lib/rlebench/media/trial-01/interaction.mp4",
                 "/proc/1/environ", "/proc/1/mem"):
    try:
        Path(filename).read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError(f"privileged file readable: {filename}")
try:
    Path("/var/lib/rlebench/hidden-com.json").write_text('{"answer":"A"}')
except PermissionError:
    pass
else:
    raise AssertionError("submission record writable")
for module in ("robosuite", "harness.tabletop.hidden_com.scene", "harness.tabletop.hidden_com.config"):
    try:
        importlib.import_module(module)
    except (ImportError, PermissionError):
        pass
    else:
        raise AssertionError(f"private module importable: {module}")
sim = HiddenCOMClient()
for request in ({"op": "finalize"}, {"op": "reset"}, {"op": "submit", "quadrant": "AB"},
                {"op": "get_state"}):
    try:
        sim._call(request)
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"forbidden operation accepted: {request}")
obs = sim.observe()
assert set(obs) == {"images", "steps", "remaining_steps", "submitted", "joint_pos",
                    "joint_vel", "eef_pos", "eef_quat", "gripper_pos", "trial", "trial_count",
                    "total_steps"}
assert set(obs["images"]) == {"workspace", "closeup", "top"}
assert all(image.shape == (512, 512, 3) for image in obs["images"].values())
print(json.dumps({"private_files_blocked": True, "record_write_blocked": True,
                  "private_imports_blocked": True, "root_operations_blocked": True,
                  "observation_allowlist_ok": True}))
