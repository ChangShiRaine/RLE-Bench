"""Task11 anti-gaming: the suction, the verifier gate and the output
hand-off can only be passed the intended way."""
import json
import os
import sys
import tempfile
import textwrap
import unittest
import unittest.mock
from pathlib import Path

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco  # noqa: E402

from harness import boxes, config, runtime, scene, scorer, spec  # noqa: E402
from harness.golden import GoldenPacker  # noqa: E402

SEED = spec.DESIGN_SEEDS[0]


def cell_with_box(pos, riding: bool, v: float = 0.1):
    """An episode cell with box 0 active at ``pos``; riding boxes move with
    the belt."""
    c = runtime._Cell(SEED)
    c.v = v
    scene.set_box_active(c.model, 0, True)
    scene.set_box_pose(c.data, 0, pos, [1, 0, 0, 0],
                       vel=(-v, 0, 0) if riding else (0, 0, 0))
    c.active.append(0)
    scene.set_arm(c.model, c.data, spec.ARM_HOME)
    scene.settle(c.model, c.data, 0.3, v)
    return c


def put_cup_on(c, xy_offset=(0.0, 0.0)):
    """Press the cup face 1.5 mm into box 0's top face (IK on a copy)."""
    top = c.pos(0) + [xy_offset[0], xy_offset[1],
                      c.stream.boxes[0].dims[2] / 2 - 0.0015]
    p = GoldenPacker()
    p.sim, p.dt = c, 0.05
    p.model, p.ik_data = c.model, mujoco.MjData(c.model)
    p.site = c.model.site("arm_tcp").id
    p.dofs = [c.model.joint(f"arm_joint{i}").dofadr[0] for i in range(1, 8)]
    p.qadr = [c.model.joint(f"arm_joint{i}").qposadr[0] for i in range(1, 8)]
    p.q_lo = c.model.jnt_range[[c.model.joint(f"arm_joint{i}").id
                                for i in range(1, 8)], 0]
    p.q_hi = c.model.jnt_range[[c.model.joint(f"arm_joint{i}").id
                                for i in range(1, 8)], 1]
    q, err = p.ik(top, 0.0)
    assert err < 2e-3
    scene.set_arm(c.model, c.data, q)
    mujoco.mj_forward(c.model, c.data)


class Suction(unittest.TestCase):
    def setUp(self):
        self.h = boxes.stream(SEED).boxes[0].dims[2]

    def engine(self, c):
        return runtime.SuctionEngine(c.model, c.n, c.v)

    def test_seals_a_riding_box_under_the_cup(self):
        c = cell_with_box([0.35, spec.BELT_Y, spec.BELT_TOP + self.h / 2],
                          True)
        put_cup_on(c)
        mujoco.mj_step(c.model, c.data)
        self.assertEqual(self.engine(c).update(c.data, True, set()), 0)

    def test_no_seal_at_a_distance_on_an_edge_or_when_committed(self):
        c = cell_with_box([0.35, spec.BELT_Y, spec.BELT_TOP + self.h / 2],
                          True)
        s = self.engine(c)
        self.assertIsNone(s.update(c.data, True, set()))   # arm at home
        put_cup_on(c, (c.stream.boxes[0].dims[0] / 2 - 0.005, 0))
        mujoco.mj_step(c.model, c.data)
        self.assertIsNone(s.update(c.data, True, set()))   # cup off the face
        put_cup_on(c)
        mujoco.mj_step(c.model, c.data)
        self.assertIsNone(s.update(c.data, True, {0}))     # committed

    def test_no_seal_on_a_stationary_box(self):
        """Picks happen on the moving belt: a box parked in the tote or held
        still off the belt cannot be re-picked."""
        for pos in ([spec.BIN_CENTER[0], spec.BIN_CENTER[1],
                     spec.BIN_FLOOR_Z + self.h / 2 + 0.001],
                    [0.35, spec.BELT_Y, spec.BELT_TOP + self.h / 2]):
            c = cell_with_box(pos, riding=False)
            c.data.joint("belt_slide").qvel[0] = 0.0
            scene.set_box_pose(c.data, 0, c.pos(0), [1, 0, 0, 0])
            put_cup_on(c)
            mujoco.mj_step(c.model, c.data)
            c.data.joint(scene.box_name(0)).qvel[:] = 0.0
            self.assertIsNone(self.engine(c).update(c.data, True, set()))


def package(code: str, extra: dict | None = None) -> str:
    d = Path(tempfile.mkdtemp()) / "policy"
    d.mkdir()
    (d / "policy.py").write_text(textwrap.dedent(code))
    for name, text in (extra or {}).items():
        (d / name).write_text(text)
    return str(d)


HOLD = """
import numpy as np
class P:
    def reset(self, cell, seed): self.home = cell["arm_home"]
    def act(self, obs): return np.array([*self.home, 0.0])
def make_policy(): return P()
"""


class Verifier(unittest.TestCase):
    """score_submission on short episodes (patched budgets, two seeds)."""

    def setUp(self):
        self._t, self._s = spec.EPISODE_T, config.SMOKE_T
        spec.EPISODE_T, config.SMOKE_T = 3.0, 1.0
        self.seeds = {"smoke": 1, "eval": [2, 3]}

    def tearDown(self):
        spec.EPISODE_T, config.SMOKE_T = self._t, self._s

    def score(self, path):
        return scorer.score_submission(path, seeds=self.seeds)

    def test_missing_or_unloadable_policy_scores_zero(self):
        self.assertEqual(self.score(tempfile.mkdtemp())["reward"], 0.0)
        r = self.score(package("import no_such_module\n"))
        self.assertEqual((r["reward"], r["gated"]), (0.0, True))

    def test_forged_reports_are_ignored(self):
        r = self.score(package(HOLD, {
            "reward.json": '{"reward": 1.0}',
            "report.json": '{"utilization": 1.0, "n_packed": 99}'}))
        self.assertEqual(r["reward"], 0.0)
        self.assertFalse(r["gated"])
        self.assertEqual(r["n_packed"], 0)

    def test_faulty_and_nondeterministic_policies_fail_the_gate(self):
        bad = self.score(package(HOLD.replace(
            "return np.array([*self.home, 0.0])", "return [float('nan')] * 8")))
        self.assertFalse(bad["gate"]["contract"])
        wobble = self.score(package(HOLD.replace(
            "return np.array([*self.home, 0.0])",
            "import os; return np.array([*self.home, 0.0]) + np.r_["
            "os.urandom(1)[0] / 2550, np.zeros(7)]")))
        self.assertTrue(wobble["gate"]["contract"])
        self.assertFalse(wobble["gate"]["determinism"])
        self.assertTrue(wobble["gated"])

    def test_symlinked_package_is_rejected(self):
        d = package(HOLD)
        os.symlink("/etc/hosts", os.path.join(d, "leak.txt"))
        self.assertEqual(self.score(d)["reward"], 0.0)


class Hardening(unittest.TestCase):
    def test_oversized_and_crowded_packages_are_rejected(self):
        from harness import sandbox
        d = package(HOLD, {"weights.bin": "x" * 4096})
        self.assertIsNone(sandbox.package_problem(d))
        with unittest.mock.patch.object(spec, "POLICY_MAX_BYTES", 1024):
            self.assertIn("exceeds", sandbox.package_problem(d))
            self.assertEqual(scorer.score_submission(
                d, seeds={"smoke": 1, "eval": [2]})["reward"], 0.0)
        with unittest.mock.patch.object(spec, "POLICY_MAX_FILES", 1):
            self.assertIn("files", sandbox.package_problem(d))

    def test_publish_replaces_planted_outputs(self):
        """A policy cannot pre-plant reward.json (as a link, a directory or a
        forged file); the scorer's private outputs replace them."""
        from harness import score_task
        out, private, target = (tempfile.mkdtemp() for _ in range(3))
        os.symlink(os.path.join(target, "x"), os.path.join(out, "reward.json"))
        os.mkdir(os.path.join(out, "report.json"))
        os.symlink(target, os.path.join(out, "media"))
        score_task._dump(os.path.join(private, "reward.json"), {"reward": 0.25})
        score_task._dump(os.path.join(private, "report.json"), {"ok": 1})
        with unittest.mock.patch.object(score_task, "VERIFIER_DIR", out):
            score_task._publish(private)
        reward = os.path.join(out, "reward.json")
        self.assertFalse(os.path.islink(reward))
        with open(reward) as f:
            self.assertEqual(json.load(f), {"reward": 0.25})
        self.assertTrue(os.path.isfile(os.path.join(out, "report.json")))
        self.assertFalse(os.path.lexists(os.path.join(out, "media")))
        self.assertEqual(os.listdir(target), [])
