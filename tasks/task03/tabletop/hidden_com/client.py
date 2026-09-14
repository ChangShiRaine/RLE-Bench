"""Public controls for the sealed-box experiment.

The box is 0.15 × 0.15 × 0.05 m, initially centered at (0, 0, 0.779) m on a
4 mm mat above the z=0.75 m table. Its rigid centered handle runs along x; the
bar is 80 × 16 × 12 mm, centered at (0, 0, 0.860) m, with 50 mm clearance below.
You may push, lift, or rotate the box on the clear tabletop.
The top camera looks straight down from (0.03, 0, 1.71) m
with a 43° vertical field of view.
"""
import base64
from io import BytesIO
import json
import socket

from PIL import Image
import numpy as np


class HiddenCOMClient:
    """Metered robot controls. Movement returns an observation.

    Actions use the last observed trial ID; stale actions are rejected.
    Movement batches contain 1–200 control steps at 20 Hz and cannot exceed
    the remaining budget. Observation and submission do not advance physics.
    """
    def __init__(self, socket_path="/run/rlebench/hidden-com.sock"):
        self.socket_path = socket_path
        self.trial = None

    def _call(self, request):
        if request.get("op") in ("step", "move", "submit") and "trial" not in request:
            if self.trial is None:
                self.observe()
            request = {**request, "trial": self.trial}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(300)
            connection.connect(self.socket_path)
            connection.sendall(json.dumps(request, allow_nan=False).encode()+b"\n")
            with connection.makefile("rb") as stream:
                response = json.loads(stream.readline(8*1024*1024))
        if not response["ok"]:
            raise RuntimeError(response["error"])
        result = response["result"]
        if "images" in result:
            self.trial = result["trial"]
            result["images"] = {name: np.asarray(Image.open(BytesIO(base64.b64decode(value))))
                                for name, value in result["images"].items()}
        return result

    def observe(self):
        """Return three 512 × 512 RGB arrays under images (workspace, closeup, top),
        joint_pos/joint_vel, eef_pos/eef_quat (xyzw), gripper_pos, trial,
        trial_count, steps, remaining_steps, and total_steps, without advancing
        physics. No force/torque, object-pose, mass, or contact readings are available.
        """
        return self._call({"op": "observe"})

    def step(self, action, steps=1):
        """Repeat a seven-component action in [-1, 1], returning an observation.

        Three world-aligned position deltas scale by 0.05 m, three axis-angle
        deltas by 0.5 rad, followed by gripper (+1 closes, -1 opens).
        """
        return self._call({"op": "step", "action": list(action), "steps": steps})

    def move(self, position, quaternion, gripper=1, steps=40):
        """Physically track a world-frame tool pose, returning an observation.

        Use a unit xyzw quaternion and intermediate waypoints as needed;
        collisions can obstruct movement. Gripper +1 closes, -1 opens.
        """
        return self._call(dict(op="move", position=list(position), quaternion=list(quaternion),
                               gripper=gripper, steps=steps))

    def submit(self, quadrant, trial=None):
        request = {"op": "submit", "quadrant": quadrant}
        if trial is not None:
            request["trial"] = trial
        return self._call(request)
