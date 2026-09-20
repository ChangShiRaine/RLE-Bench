"""One metered attempt, scored from simulator state when it finishes."""
from math import isfinite
from threading import Event

import numpy as np
from scipy.spatial.transform import Rotation

from harness import config as C, env as ENV, ledger as L
from harness.session import (BudgetExhausted, EpisodeOver, SessionClosed,
                             SessionState, _NullRecorder, episode_instruction,
                             seed_episode)


class Session:
    evaluating = False

    def __init__(self, task, budgets, ledger, env_factory, trial_seeds=None,
                 max_episode_steps=None, eval_plan_fn=None, transcript=None,
                 recorder=None):
        if budgets.interaction_steps <= 0:
            raise ValueError("interaction budget must be positive")
        self.task, self.budgets, self.ledger = task, budgets, ledger
        self._factory = env_factory
        plan = eval_plan_fn(task) if eval_plan_fn else [(1, 1, 1, trial_seeds[0])]
        self._seed = plan[0][3]
        self.max_episode_steps = budgets.interaction_steps
        self.transcript = transcript
        self.recorder = recorder or _NullRecorder()
        self.state = SessionState()
        self._env = None
        self._obs = None
        self._closed = False
        self._episode_over = False
        self._failed = False
        self._end_reason = None
        self._stop = Event()
        ledger.append(L.KIND_START, task=task, interaction_budget=budgets.interaction_steps,
                      submission_budget=1, planned_trials=1,
                      image_convention=ENV.image_convention())

    @property
    def current_env(self):
        return self._env

    @property
    def steps_remaining(self):
        return self.budgets.interaction_steps - self.state.steps_used

    def _require_open(self):
        if self._closed:
            raise SessionClosed("attempt is finished")

    def _start_episode(self):
        if self._env is None:
            self._env = self._factory(task=self.task, split="target")
        self._obs = seed_episode(self._env, self._seed)
        self.state.episodes += 1
        self.state.episode_steps = 0
        self._episode_over = False
        # The shared recorder names scored trajectories "evaluation" internally.
        self.recorder.episode_started("evaluation", self.state.episodes, self._obs)

    def probe(self):
        if self._obs is None:
            self._start_episode()
        low, _ = self._env.action_spec
        return dict(action_dim=len(low), instruction=episode_instruction(self._env))

    def _charge(self, count):
        self.state.steps_used += count
        self.ledger.append(L.KIND_INTERACT, steps=count)

    def reset(self, seed=None):
        self._require_open()
        if seed is not None:
            raise ValueError("reset does not accept a seed")
        if self.steps_remaining <= 0:
            raise BudgetExhausted("interaction budget exhausted")
        self._charge(1)
        try:
            self._start_episode()
        except Exception as exc:
            self._failed = True
            self.close("simulation_error")
            raise RuntimeError("simulation reset failed") from exc
        obs = self._obs
        if self.steps_remaining == 0:
            self.close("budget_exhausted")
        return obs

    def observe(self, spec=None):
        if self._obs is None:
            return dict(obs={}, instruction=None, resolution=None, live=False)
        obs, resolution = ENV.apply_obs_spec(
            self._env, dict(self._obs), spec, default=C.OBS_RESOLUTION,
            rendered=C.RENDER_RESOLUTION, ceiling=C.OBS_MAX_RESOLUTION)
        return dict(obs=obs, instruction=episode_instruction(self._env),
                    resolution=resolution, live=not self._closed and not self._episode_over)

    def step(self, actions, obs_spec=None):
        self._require_open()
        if self._episode_over:
            raise EpisodeOver("episode ended; finish or reset the attempt")
        if len(actions) > self.steps_remaining:
            raise BudgetExhausted(
                f"requested {len(actions)} steps but only {self.steps_remaining} remain")
        applied = 0
        try:
            for action in actions:
                if self._stop.is_set():
                    break
                # Charge before stepping so a failed simulator call cannot be retried free.
                self._charge(1)
                self.state.episode_steps += 1
                applied += 1
                self._obs, _, done, _ = self._env.step(action)
                self.recorder.frame(self._obs)
                if done:
                    self._episode_over = True
                    break
        except Exception as exc:
            self._failed = True
            self.close("simulation_error")
            raise RuntimeError("simulation step failed") from exc
        # Render before scoring: some physical score hooks settle or shake the scene.
        obs = self.observe(obs_spec)["obs"]
        if self.steps_remaining == 0:
            self.close("budget_exhausted")
        elif self._stop.is_set():
            self.close("harness_seal")
        return dict(obs=obs, steps=applied, steps_remaining=self.steps_remaining,
                    episode_over=self._episode_over or self._closed,
                    done=self._closed,
                    ended=self._end_reason or ("env_done" if self._episode_over else None))

    def move(self, request, obs_spec=None):
        def vector(key, length):
            value = request.get(key)
            if not isinstance(value, list) or len(value) != length or any(
                    type(v) not in (int, float) or not isfinite(v) for v in value):
                raise ValueError(f"{key} requires {length} finite numbers")
            return np.asarray(value, dtype=float)
        target = vector("position", 3)
        quaternion = vector("quaternion", 4)
        if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-4):
            raise ValueError("quaternion must be unit xyzw")
        gripper = request.get("gripper", 1)
        if type(gripper) not in (int, float) or not isfinite(gripper) or not -1 <= gripper <= 1:
            raise ValueError("gripper must be in [-1, 1]")
        count = request.get("steps", 40)
        if type(count) is not int or not 1 <= count <= 200:
            raise ValueError("steps must be an integer in [1, 200]")
        orientation = Rotation.from_quat(quaternion).as_matrix()
        session = self
        class Actions:
            def __len__(self):
                return count
            def __iter__(self):
                for _ in range(count):
                    obs = session._obs
                    base = Rotation.from_quat(obs["robot0_base_quat"]).as_matrix()
                    target_base = base.T @ (target - obs["robot0_base_pos"])
                    current = Rotation.from_quat(obs["robot0_base_to_eef_quat"]).as_matrix()
                    dp = (target_base - obs["robot0_base_to_eef_pos"]) / .05
                    dr = Rotation.from_matrix(base.T @ orientation @ current.T).as_rotvec() / .5
                    yield np.r_[np.clip(dp, -1, 1), np.clip(dr, -1, 1), gripper, 0, 0, 0, 0, -1]
        return self.step(Actions(), obs_spec=obs_spec)

    def _score(self):
        if self._failed or self._env is None:
            return 0.0
        try:
            hook = getattr(self._env, "_trial_score", None)
            value = float(hook() if callable(hook) else self._env._check_success())
            return min(1.0, max(0.0, value)) if isfinite(value) else 0.0
        except Exception:
            return 0.0

    def close(self, end_reason="harness_seal"):
        if not self._closed:
            value = self._score()
            self.state.best_success_rate = value
            self.state.submissions_used = 1
            self.ledger.append(L.KIND_SUBMIT, success_rate=value, trials=1,
                               trials_attempted=int(self.state.steps_used > 0))
            count = sum(1 for _ in L.read_records(self.ledger.path)) + 1
            self.ledger.append(L.KIND_SEAL, record_count=count,
                               steps_used=self.state.steps_used, submissions_used=1,
                               best_success_rate=value, end_reason=end_reason)
            self._closed = True
            self._end_reason = end_reason
            self.recorder.close()
        return L.summarize(L.load_verified(self.ledger.path))

    def request_stop(self):
        self._stop.set()

    def finish(self):
        self.close("agent_finish")
        return self.status()

    def status(self):
        return dict(task=self.task, steps_used=self.state.steps_used,
                    steps_remaining=self.steps_remaining,
                    interaction_budget=self.budgets.interaction_steps,
                    done=self._closed, ended=self._end_reason)
