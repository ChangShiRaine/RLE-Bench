"""Task11 analytic checks: fit rule, trackers and reward formula (no MuJoCo
stepping). Expected values are derived by hand from the published rules."""
import unittest

import numpy as np

from harness import config, metrics, packing, scorer, spec

H = spec.BIN_INNER[2]


def box_corners(x0, y0, x1, y1, top):
    """World corners of an axis-aligned box resting on the tote floor."""
    z0 = spec.BIN_FLOOR_Z
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1)
                     for z in (z0, spec.BIN_FLOOR_Z + top)])


class FitRule(unittest.TestCase):
    def test_empty_tote_fits_every_type(self):
        empty = np.zeros(packing.grid())
        for name, dims in spec.BOX_TYPES:
            self.assertTrue(packing.fits(empty, dims), name)

    def test_oversized_and_overtall_boxes_do_not_fit(self):
        empty = np.zeros(packing.grid())
        # 0.39 + 2 * 0.01 clearance > 0.40 m interior length, both yaws
        self.assertFalse(packing.fits(empty, (0.39, 0.29, 0.05)))
        self.assertFalse(packing.fits(empty, (0.10, 0.10, H + 0.001)))

    def test_full_height_load_blocks_everything(self):
        full = np.full(packing.grid(), H - 0.03)
        for _, dims in spec.BOX_TYPES:
            self.assertFalse(packing.fits(full, dims))

    def test_heightmap_rasterizes_box_top(self):
        lo = np.array(spec.BIN_LO[:2])
        hm = packing.heightmap([box_corners(*(lo + 0.05), *(lo + 0.15), 0.07)])
        self.assertAlmostEqual(hm[5:15, 5:15].min(), 0.07)
        self.assertEqual(hm[:5].max(), 0.0)
        self.assertEqual(hm[15:].max(), 0.0)
        self.assertAlmostEqual(hm.sum(), 100 * 0.07)

    def test_support_rule_rejects_overhang_and_unsupported_centre(self):
        dims = (0.10, 0.08, 0.05)          # grown footprint 12 x 10 cells
        hm = np.zeros(packing.grid())
        hm[:, :] = 0.20                     # tall load everywhere ...
        hm[0:12, 0:10] = 0.10               # ... except one exact pocket
        self.assertEqual(
            {(p[0], p[1], p[3]) for p in packing.placements(hm, dims)},
            {(0, 0, 0.10)})
        # half the pocket drops to 0.05: 50% support < 60% -> no fit
        hm[0:6, 0:10] = 0.05
        self.assertFalse(packing.fits(hm, dims))
        # 70% supported, but the centre cells sit 2 cm low -> no fit
        hm[0:12, 0:10] = 0.10
        hm[5:7, 4:6] = 0.08
        self.assertFalse(packing.fits(hm, dims))


class Trackers(unittest.TestCase):
    def test_full_needs_five_consecutive_non_fitting_passes(self):
        full = metrics.FullDetector()
        for t in range(4):
            full.on_pass(t, fitted=False)
        full.on_pass(4, fitted=True)        # would have fitted: reset
        for t in range(5, 9):
            full.on_pass(t, fitted=False)
        full.reset()                        # a grasp or commit: reset
        for t in range(10, 14):
            self.assertFalse(full.on_pass(t, fitted=False))
        self.assertTrue(full.on_pass(14, fitted=False))
        self.assertEqual((full.full_t, full.missed_fitting, full.passed),
                         (14.0, 1, 14))

    def test_damage_crush_and_impact(self):
        d = metrics.DamageTracker(250.0, 2, 1.2)
        # one-tick spike is not a crush; two consecutive ticks are
        d.update({0: 400.0, 1: 0.0}, {}, {}, set())
        d.update({0: 10.0, 1: 0.0}, {}, {}, set())
        self.assertEqual(d.damaged, {})
        d.update({0: 300.0}, {}, {}, set())
        self.assertEqual(d.update({0: 300.0}, {}, {}, set()), [0])
        # a fast free box that makes new contact is an impact
        d.update({1: 0.0}, {1: 1.5}, {1: False}, {1})
        self.assertEqual(d.update({1: 5.0}, {1: 0.0}, {1: True}, {1}), [1])
        self.assertEqual(d.damaged, {0: "crush", 1: "impact"})
        # the same motion while held (not free) is not an impact
        d.update({2: 0.0}, {2: 1.5}, {2: False}, set())
        self.assertEqual(d.update({2: 5.0}, {2: 0.0}, {2: True}, set()), [])

    def test_bin_hit_counts_strikes_not_grinding(self):
        b = metrics.BinHitTracker(150.0, 2, 5)
        for f in [200] * 10 + [0] * 5 + [200] * 2 + [0] * 3 + [200] * 2:
            b.update(f)
        self.assertEqual(b.hits, 2)         # the last burst came before rearm

    def test_throughput_paced_by_makespan_only_when_full(self):
        kw = dict(n_packed=12, packed_volume=0.0144, attempts=15, grasps=14,
                  commit_times=list(np.linspace(10, 120, 12)), floor_drops=1,
                  damage_events=0, bin_hits=0, passed=20, missed_fitting=3,
                  spawned=35)
        full = metrics.episode_metrics(full_t=150.0, **kw)
        open_ = metrics.episode_metrics(full_t=None, **kw)
        self.assertAlmostEqual(full["throughput_ppm"], 6.0)
        self.assertAlmostEqual(open_["throughput_ppm"], 3.0)
        self.assertAlmostEqual(full["utilization"], 0.5)
        self.assertAlmostEqual(full["pick_success_rate"], 14 / 15)
        self.assertAlmostEqual(full["pick_place_success_rate"], 12 / 15)
        self.assertAlmostEqual(full["floor_drop_rate"], 1 / 14)


def m(**kw):
    base = dict(utilization=0.0, throughput_ppm=0.0, n_packed=0,
                pick_place_success_rate=0.0, bin_full=False, floor_drops=0,
                damage_events=0, bin_hits=0)
    base.update(kw)
    return base


class Reward(unittest.TestCase):
    def test_nothing_packed_scores_zero(self):
        self.assertEqual(scorer.episode_score(m()), 0.0)

    def test_saturated_episode_scores_one(self):
        self.assertAlmostEqual(scorer.episode_score(m(
            utilization=0.75, throughput_ppm=9.0, n_packed=20,
            pick_place_success_rate=1.0, bin_full=True)), 1.0)

    def test_hand_computed_episode(self):
        # u = 0.35/0.70 = 0.5, t = 3.5/7 = 0.5, pp = 0.8 * 5/10 = 0.4, full
        # 0.4*0.5 + 0.25*0.5*0.5 + 0.2*0.4 + 0.15*0.5 - 0.04 = 0.3775
        got = scorer.episode_score(m(
            utilization=0.35, throughput_ppm=3.5, n_packed=5,
            pick_place_success_rate=0.8, bin_full=True, damage_events=1))
        self.assertAlmostEqual(got, 0.3775)

    def test_floor_drop_forfeits_full_bonus_and_penalties_clamp(self):
        a = m(utilization=0.70, throughput_ppm=7, n_packed=10,
              pick_place_success_rate=1.0, bin_full=True)
        self.assertAlmostEqual(scorer.episode_score(a), 1.0)
        self.assertAlmostEqual(scorer.episode_score({**a, "floor_drops": 1}),
                               1.0 - config.W_FULL - config.W_FLOOR)
        self.assertEqual(scorer.episode_score({**a, "bin_hits": 100}), 0.0)

    def test_fast_sloppy_fill_cannot_reach_a_dense_one(self):
        sloppy = m(utilization=0.20, throughput_ppm=10, n_packed=5,
                   pick_place_success_rate=1.0, bin_full=True)
        dense = m(utilization=0.55, throughput_ppm=5, n_packed=12,
                  pick_place_success_rate=1.0, bin_full=True)
        self.assertLess(scorer.episode_score(sloppy),
                        scorer.episode_score(dense) - 0.3)
