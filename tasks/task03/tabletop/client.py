"""RGB and robot controls for the tabletop experiment.

The PandaOmron has an arm, gripper, mobile base, and torso. Control runs at
20 Hz. The table top is z=0.9 m, spanning 0.7 × 1.6 m around world (0, 0).
Quaternions use unit xyzw. Images are upright 512 × 512 RGB arrays.

step() uses 12 values in [-1, 1]: base-frame arm translation (3, scaled by
0.05 m), axis-angle rotation (3, scaled by 0.5 rad), gripper (+1 closes,
-1 opens), base forward/lateral/yaw velocity (3), torso velocity (1), and
mode (-1 arm, +1 base). Neutral hold: [0]*6 + [-1] + [0]*4 + [-1].
move() instead tracks a world-frame tool pose through the same controller.
Use intermediate waypoints; collisions and reach limits can obstruct motion.
No automatic grasping or perception tools are provided. BalanceCoins also
provides cube_positions and pan_positions in world coordinates.
"""
import base64
from io import BytesIO
import json
import socket

import numpy as np
from PIL import Image


class RemoteError(RuntimeError):
    pass


class TabletopClient:
    def __init__(self, socket_path="/run/rlebench/speedrun.sock"):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(300)
        self._sock.connect(socket_path)
        self._reader = self._sock.makefile("rb")

    def _request(self, op, **fields):
        self._sock.sendall(json.dumps(dict(op=op, **fields), allow_nan=False).encode()+b"\n")
        result = json.loads(self._reader.readline(8*1024*1024))
        if not result.pop("ok"):
            raise RemoteError(result["error"])
        if "images" in result:
            result["images"] = {name: np.asarray(Image.open(BytesIO(base64.b64decode(data))))
                                for name, data in result["images"].items()}
        return result

    def observe(self):
        """Return images (left, right, wrist), joint_pos/joint_vel, eef_pos/eef_quat,
        gripper_pos, base_pos/base_quat, and eef_base_pos/eef_base_quat. Positions
        are in meters; eef_base_* is relative to the robot base, other poses are
        world-frame. Includes task, steps_used, steps_remaining, interaction_budget,
        done, and ended. Observation is free. BalanceCoins includes cube_positions (cube centers)
        and pan_positions (left/right upper-surface centers), keyed by name,
        in world meters. No depth, force/torque, contact, mass, or success
        readings are available.
        """
        return self._request("observe")

    def step(self, action, steps=1):
        """Repeat a 12-component action, or execute a batch of distinct actions.

        Batches contain 1–200 steps and may not exceed the remaining budget.
        Returns an observation. For a batch, leave steps=1.
        """
        actions = np.asarray(action).tolist()
        if actions and isinstance(actions[0], list):
            if steps != 1:
                raise ValueError("steps must be 1 for a batch")
        else:
            if type(steps) is not int or not 1 <= steps <= 200:
                raise ValueError("steps must be an integer in [1, 200]")
            actions = [actions] * steps
        return self._request("step", actions=actions)

    def move(self, position, quaternion, gripper=1, steps=40):
        """Track a world-frame tool pose for 1–200 metered steps, then observe.

        position is xyz in meters, quaternion is unit xyzw. Gripper +1 closes,
        -1 opens. The base and torso hold still during this movement.
        """
        return self._request("move", position=list(position), quaternion=list(quaternion),
                             gripper=gripper, steps=steps)

    def reset(self):
        """Restore the same initial scene and observe; costs one step."""
        return self._request("reset")

    def finish(self):
        """Submit the current simulator state once and return budget status."""
        return self._request("finish")

    def status(self):
        return self._request("status")

    def disconnect(self):
        self._reader.close()
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()
