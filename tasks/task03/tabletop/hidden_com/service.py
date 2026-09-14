"""Three private experiments with bounded interaction and immutable submissions."""
import base64
import gc
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .config import CASES, MAX_BATCH, MAX_STEPS
from .scene import CAMERAS, HiddenCOM
from .render import render


class Session:
    def __init__(self, state_path, env_factory=HiddenCOM, video_root=None, cases=CASES):
        self.state_path = Path(state_path)
        self.env_factory = env_factory
        self.cases = tuple(cases)
        self.video_root = Path(video_root) if video_root is not None else None
        self.results = []
        self.closed = False
        self.in_trial = False
        self.env = None
        self.video = None
        self._start_trial()

    def _start_trial(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            gc.collect()  # Release the old GL context before constructing its replacement.
        self.trial_index = len(self.results)
        case = self.cases[self.trial_index]
        self.quadrant = str(np.random.default_rng(case.seed).choice(list("ABCD")))
        self.steps = 0
        self.answer = None
        self.env = self.env_factory(self.quadrant, case=case)
        self.env.reset()
        self.in_trial = True
        self.video = None
        self._save()
        if self.video_root is not None:
            try:
                from .video import InteractionVideo
                self.video = InteractionVideo(self.video_root / f"trial-{self.trial_index+1:02d}",
                                              trial=self.trial_index+1, total=len(self.cases))
                self.video.capture(self.env, self.steps)
            except Exception:
                self.video = None

    def _finish_recording(self):
        if self.video is not None:
            try:
                self.video.finish(self.env, self.steps, self.answer)
            except Exception:
                pass

    def _trial_record(self, closed):
        return dict(trial=self.trial_index+1, quadrant=self.quadrant, answer=self.answer,
                    steps=self.steps, closed=closed)

    def _save(self):
        record = dict(closed=self.closed, trials=self.results,
                      active=self._trial_record(False) if self.in_trial else None)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record)+"\n")
        temporary.chmod(0o600)
        temporary.replace(self.state_path)

    def finalize(self):
        if not self.closed:
            if self.in_trial:
                self.results.append(self._trial_record(True))
                self.in_trial = False
            self.closed = True
            self._save()
        self._finish_recording()

    def observe(self):
        robot = self.env.robots[0]
        sim = self.env.sim
        site = robot.eef_site_id["right"]
        cameras = {}
        for name in CAMERAS:
            buffer = BytesIO()
            render(self.env, name).save(buffer, format="PNG")
            cameras[name] = base64.b64encode(buffer.getvalue()).decode("ascii")
        return dict(images=cameras, trial=self.trial_index+1, trial_count=len(self.cases),
            steps=self.steps, remaining_steps=MAX_STEPS-self.steps,
            total_steps=sum(r["steps"] for r in self.results)+self.steps,
            submitted=False,
            joint_pos=sim.data.qpos[robot._ref_joint_pos_indexes].tolist(),
            joint_vel=sim.data.qvel[robot._ref_joint_vel_indexes].tolist(),
            eef_pos=sim.data.site_xpos[site].tolist(),
            eef_quat=Rotation.from_matrix(sim.data.site_xmat[site].reshape(3, 3)).as_quat().tolist(),
            gripper_pos=sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]].tolist())

    @staticmethod
    def vector(value, length):
        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"expected a list of {length} finite numbers")
        if any(type(x) not in (int, float) for x in value):
            raise ValueError("expected numeric values")
        array = np.array(value, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("expected finite values")
        return array

    def _count(self, request):
        count = request.get("steps", 1)
        if type(count) is not int or not 1 <= count <= MAX_BATCH:
            raise ValueError(f"steps must be an integer in [1, {MAX_BATCH}]")
        if self.steps + count > MAX_STEPS:
            raise ValueError("interaction budget exhausted; submit your answer")
        return count

    def _advance(self, action):
        self.steps += 1
        self.env.step(action)
        if not np.isfinite(self.env.sim.data.qpos).all():
            self.finalize()
            raise RuntimeError("simulation stopped")
        if self.video is not None:
            self.video.capture(self.env, self.steps)

    def handle(self, request, uid):
        op = request.get("op")
        if op == "finalize":
            if uid != 0:
                raise ValueError("operation unavailable")
            self.finalize()
            return {"finalized": True}
        if self.closed:
            raise ValueError("session closed")
        if op == "observe":
            return self.observe()
        if op not in ("submit", "step", "move"):
            raise ValueError("unknown operation; use observe, step, move, or submit")
        trial = request.get("trial")
        if type(trial) is not int or trial != self.trial_index+1:
            raise ValueError("stale or missing trial id; observe the current trial")
        if op == "submit":
            answer = request.get("quadrant")
            if type(answer) is not str or answer not in ("A", "B", "C", "D"):
                raise ValueError("quadrant must be A, B, C, or D")
            self.answer = answer
            self.results.append(self._trial_record(True))
            self.in_trial = False
            self.closed = len(self.results) == len(self.cases)
            response = dict(submitted=True, quadrant=answer, steps=self.steps, trial=trial,
                            remaining_trials=len(self.cases)-len(self.results), done=self.closed)
            self._save()
            self._finish_recording()
            if not self.closed:
                self._start_trial()
            return response
        count = self._count(request)
        if op == "step":
            action = self.vector(request.get("action"), 7)
            if np.max(np.abs(action)) > 1:
                raise ValueError("action values must be in [-1, 1]")
            for _ in range(count):
                self._advance(action)
        else:
            target = self.vector(request.get("position"), 3)
            quat = self.vector(request.get("quaternion"), 4)
            if abs(np.linalg.norm(quat)-1) > 1e-5:
                raise ValueError("quaternion must be unit length, xyzw")
            gripper = request.get("gripper", 1)
            if type(gripper) not in (int, float) or not -1 <= gripper <= 1:
                raise ValueError("gripper must be in [-1, 1]")
            orientation = Rotation.from_quat(quat).as_matrix()
            robot = self.env.robots[0]
            site = robot.eef_site_id["right"]
            for _ in range(count):
                data = self.env.sim.data
                # The fixed robot base has identity orientation, so base/world deltas coincide.
                dp = (target-data.site_xpos[site]) / 0.05
                dr = Rotation.from_matrix(orientation @ data.site_xmat[site].reshape(3, 3).T).as_rotvec() / 0.5
                self._advance(np.r_[np.clip(dp, -1, 1), np.clip(dr, -1, 1), gripper])
        self._save()
        return self.observe()
