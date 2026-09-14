"""Single-stage operations over the shared authenticated socket transport."""
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
from PIL import Image

from harness.obs import ObsSpec

from harness import protocol as P
from harness.service import Service as BaseService, _phase_seconds_from_env


class Service(BaseService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tabletop-sim")
        self._phase_seconds = _phase_seconds_from_env("RLEBENCH_SECONDS")

    def warm_up(self):
        return self._worker.submit(super().warm_up).result()

    def _dispatch(self, op, msg):
        return self._worker.submit(self._dispatch_on_worker, op, msg).result()

    def _shown(self, obs):
        # Explicit allowlist: raw observations also contain private object state.
        keys = {
            "joint_pos": "robot0_joint_pos", "joint_vel": "robot0_joint_vel",
            "eef_pos": "robot0_eef_pos", "eef_quat": "robot0_eef_quat",
            "gripper_pos": "robot0_gripper_qpos", "base_pos": "robot0_base_pos",
            "base_quat": "robot0_base_quat", "eef_base_pos": "robot0_base_to_eef_pos",
            "eef_base_quat": "robot0_base_to_eef_quat",
        }
        out = {name: np.asarray(obs[key]).tolist() for name, key in keys.items() if key in obs}
        out["images"] = {}
        for name, camera in (("left", "robot0_agentview_left"),
                             ("right", "robot0_agentview_right"), ("wrist", "robot0_eye_in_hand")):
            if camera+"_image" in obs:
                buffer = BytesIO()
                Image.fromarray(np.asarray(obs[camera+"_image"], dtype=np.uint8)).save(buffer, format="PNG")
                out["images"][name] = base64.b64encode(buffer.getvalue()).decode("ascii")
        return out

    def _dispatch_on_worker(self, op, msg):
        allowed = {
            "observe": {"op"}, "reset": {"op"}, "finish": {"op"}, "status": {"op"},
            "step": {"op", "actions"},
            "move": {"op", "position", "quaternion", "gripper", "steps"},
        }
        if op not in allowed or set(msg) - allowed[op]:
            raise P.ProtocolError("unknown operation or argument")
        spec = ObsSpec(width=512, height=512)
        if op == "finish":
            return {"ok": True, **self._session.finish()}
        if op == "status":
            return {"ok": True, **self._session.status()}
        if op == "reset":
            self._session.reset()
        if op in {"observe", "reset"}:
            obs = self._session.observe(spec)["obs"]
        elif op == "step":
            actions = P.validate_actions(msg.get("actions"), self._require_action_dim())
            if not 1 <= len(actions) <= 200:
                raise P.ProtocolError("batches must contain 1–200 steps")
            obs = self._session.step(actions, obs_spec=spec)["obs"]
        else:
            obs = self._session.move(msg, obs_spec=spec)["obs"]
        shown = self._shown(obs)
        if self._session.task == "BalanceCoins":
            shown.update(self._session.current_env.position_observation())
        return {"ok": True, **shown, **self._session.status()}

    def _dispatch_control(self, op, msg, sock):
        if op != "seal":
            raise P.ProtocolError(f"unknown op {op!r}")
        return self._worker.submit(super()._dispatch_control, op, msg, sock).result()

    def seal_if_open(self):
        # Stop a long batch at its next control-step boundary before sealing.
        self._session.request_stop()
        with self._lock:
            return self._worker.submit(super().seal_if_open).result()

    def shutdown(self):
        super().shutdown()
        self._worker.shutdown(wait=True)
