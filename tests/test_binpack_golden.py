"""Task11 simulation regressions: the box stream, the belt, the sensing
contract, and the golden packer's clean, deterministic run."""
import os
import sys
import unittest
from collections import Counter

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")

import numpy as np  # noqa: E402

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


class Stream(unittest.TestCase):
    def test_amazon_types_copies_shuffle_and_shared_density(self):
        a, b = boxes.stream(SEED), boxes.stream(SEED)
        self.assertEqual(a, b)
        counts = Counter(x.kind for x in a.boxes)
        self.assertEqual(set(counts), {n for n, _ in spec.BOX_TYPES})
        lo, hi = spec.BOX_COPIES_RANGE
        self.assertTrue(all(lo <= c <= hi for c in counts.values()))
        self.assertNotEqual([x.kind for x in a.boxes],
                            [x.kind for x in boxes.stream(SEED + 1).boxes])
        for x in a.boxes:
            self.assertAlmostEqual(x.mass, x.volume * spec.BOX_DENSITY)
            self.assertLess(x.mass, spec.SUCTION_MAX_KG / 2)
        gaps = np.diff([x.arrival_t for x in a.boxes])
        self.assertTrue(((gaps >= 4.0 - 1e-6) & (gaps <= 7.0 + 1e-6)).all())
        # enough boxes that the stream never runs dry within an episode
        self.assertGreater(a.boxes[-1].arrival_t, spec.EPISODE_T + 60)


class Belt(unittest.TestCase):
    def test_null_policy_boxes_ride_the_belt_and_pass(self):
        log = runtime.run_episode(runtime.NullPolicy(), SEED, budget_t=60,
                                  render=False)
        mt = log.metrics()
        self.assertEqual((mt["n_packed"], mt["pick_attempts"]), (0, 0))
        self.assertGreater(mt["boxes_passed"], 5)
        # an empty tote fits every passing box: never declared full
        self.assertEqual(mt["missed_fitting"], mt["boxes_passed"])
        self.assertFalse(mt["bin_full"])
        self.assertEqual(scorer.episode_score(mt), 0.0)

    def test_box_moves_at_belt_speed(self):
        c = cell_with_box([0.8, spec.BELT_Y, spec.BELT_TOP + 0.04], True, 0.1)
        x0 = c.pos(0)[0]
        scene.settle(c.model, c.data, 2.0, 0.1)
        self.assertAlmostEqual(c.pos(0)[0], x0 - 0.2, delta=0.005)


class Observations(unittest.TestCase):
    def test_sensing_contract(self):
        """Proprioception, scanner rows and belt speed every tick; top-view
        and wrist RGB-D on frame ticks; nothing else."""
        seen = []

        class Spy(runtime.NullPolicy):
            def act(self, obs):
                seen.append(obs)
                return super().act(obs)

        runtime.run_episode(Spy(), SEED, budget_t=2.0, render=True)
        every = {"qpos", "qvel", "tau", "ft", "suction_on", "seal", "t",
                 "belt_v", "scans"}
        frame = {"top_rgb", "top_depth", "wrist_rgb", "wrist_depth",
                 "wrist_T_world_cam"}
        self.assertEqual(set(seen[1]), every)
        self.assertEqual(set(seen[0]), every | frame)
        self.assertEqual(sum("top_rgb" in o for o in seen),
                         len(seen) // spec.FRAME_EVERY_TICKS)
        self.assertEqual(seen[0]["top_depth"].shape, (spec.TOP_H, spec.TOP_W))
        self.assertEqual(seen[0]["wrist_rgb"].shape,
                         (spec.WRIST_H, spec.WRIST_W, 3))
        scans = seen[-1]["scans"]
        self.assertEqual(scans.shape, (1, 9))       # first box has entered
        truth = boxes.stream(SEED).boxes[0]
        np.testing.assert_allclose(scans[0, 5:8], truth.dims, atol=0.006)


class Golden(unittest.TestCase):
    def test_golden_packs_cleanly_and_deterministically(self):
        a = runtime.run_episode(GoldenPacker(), SEED, budget_t=70,
                                render=False)
        b = runtime.run_episode(GoldenPacker(), SEED, budget_t=70,
                                render=False)
        self.assertEqual(a.digest, b.digest)
        mt = a.metrics()
        self.assertGreaterEqual(mt["n_packed"], 5)
        self.assertEqual((mt["floor_drops"], mt["damage_events"],
                          mt["bin_hits"]), (0, 0, 0))
        self.assertEqual(mt["pick_place_success_rate"], 1.0)
        self.assertLess(a.peak_sustained, config.DAMAGE_FORCE_N / 1.3 + 1)
