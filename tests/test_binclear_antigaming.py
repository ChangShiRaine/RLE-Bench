"""Task07 anti-gaming: the reward can only be earned the intended way.

These tests pin the simulation seams: the magnet and the latch cannot be
abused to mint clears, and the scorer only pays for the intended outcome.
"""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

from harness import runtime, scene, spec
from harness.runtime import MagnetEngine


def _one_part_cell(pos):
    return scene.build_scene(
        part_poses=[(np.asarray(pos), np.array([1.0, 0, 0, 0]))])


def test_magnet_only_welds_on_touch():
    """An energized head far from every part holds nothing — there is no
    action at a distance to exploit."""
    model, data = _one_part_cell(
        [spec.BIN_POS[0], spec.BIN_POS[1], spec.STAND_H + 0.03])
    scene.set_arm(model, data, spec.ARM_HOME)
    mag = MagnetEngine(model)
    for _ in range(20):
        mag.update(data, energized=True)
        scene.settle(model, data, 0.05)
    assert mag.held(data) == set()


def test_magnet_releases_everything_on_deenergize():
    model, data = _one_part_cell(
        [spec.BIN_POS[0], spec.BIN_POS[1], spec.STAND_H + 0.03])
    mag = MagnetEngine(model)
    mag._weld(data, 0)
    assert mag.held(data) == {0}
    held = mag.update(data, energized=False)
    assert held == set() and mag.held(data) == set()


def test_cleared_parts_can_never_be_rewelded():
    """Graveyarded parts are excluded: no double-clearing by re-grabbing."""
    model, data = _one_part_cell(
        [spec.BIN_POS[0], spec.BIN_POS[1], spec.STAND_H + 0.03])
    mag = MagnetEngine(model)
    mag._weld(data, 0)
    held = mag.update(data, energized=True, exclude={0})
    assert held == set() and mag.held(data) == set()


def test_flung_parts_do_not_latch():
    """A part crossing the zone at speed never counts as cleared — only
    resting in the zone latches (throwing must still stick the landing)."""
    zc = (np.asarray(spec.ZONE_LO) + np.asarray(spec.ZONE_HI)) / 2
    import harness.episodes as ep
    orig = ep.build_episode_scene

    def flying_part(seed, with_arm=True):
        model, data = scene.build_scene(
            part_poses=[(np.array([zc[0], zc[1] - 0.3, 0.35]),
                         np.array([1.0, 0, 0, 0]))], seed=seed)
        data.joint("part0").qvel[:3] = [0.0, 3.0, 0.5]   # flung across
        return model, data

    ep.build_episode_scene = flying_part
    try:
        log = runtime.run_episode(runtime.NullPolicy(), 1, budget_t=2.0,
                                  render=False)
    finally:
        ep.build_episode_scene = orig
    # it crossed the zone airborne and left: no clear latched
    assert 0 not in log.clear_times or log.clear_times[0] > 0.4


def test_dangled_parts_do_not_latch():
    """THE HOVER EXPLOIT (found by an agent): holding a part stationary
    over the drop zone must never count as clearing it — only a released,
    resting part can be carried away."""
    from harness.metrics import ClearanceTracker
    zc = (np.asarray(spec.ZONE_LO) + np.asarray(spec.ZONE_HI)) / 2
    tr = ClearanceTracker(1)
    pos, v = np.asarray([zc]), np.asarray([0.0])
    for k in range(4 * spec.CLEAR_REST_TICKS):     # long stationary hover
        assert tr.update(k * 0.05, pos, v, exclude={0}) == []
    assert tr.cleared == 0
    # release: the rest window starts fresh from zero
    for k in range(spec.CLEAR_REST_TICKS - 1):
        assert tr.update(10 + k * 0.05, pos, v) == []
    assert tr.update(11.0, pos, v) == [0]


def test_hover_exploit_end_to_end(monkeypatch):
    """A privileged policy that welds a part and hovers it in-zone earns
    nothing until it de-energizes and lets the part land."""
    from harness.runtime import MagnetEngine
    zc = (np.asarray(spec.ZONE_LO) + np.asarray(spec.ZONE_HI)) / 2
    import harness.episodes as ep

    def part_on_magnet(seed, with_arm=True):
        model, data = scene.build_scene(
            part_poses=[(np.array([zc[0], zc[1], spec.ZONE_HI[2] - 0.02]),
                         np.array([1.0, 0, 0, 0]))], seed=seed)
        return model, data

    monkeypatch.setattr(ep, "build_episode_scene", part_on_magnet)

    class Hoverer:
        """Welds a part to the tool (privileged) and keeps the magnet
        energized forever — the part dangles in-zone, stationary."""
        privileged = True

        def bind(self, model, data):
            self.mag = MagnetEngine(model)
            self.mag._weld(data, 0)          # part rigidly on the tool

        def reset(self, cell, seed):
            pass

        def act(self, obs):
            return np.array([*spec.ARM_HOME, 255.0])

    log = runtime.run_episode(Hoverer(), 7, budget_t=4.0, render=False)
    assert log.clear_times == {}      # dangled in-zone: never cleared


def test_exclusive_zone_blocks_doubles():
    """Singulated delivery: two unlatched parts resting in the zone block
    each other; picking one back up (held) lets the other latch."""
    from harness.metrics import ClearanceTracker
    zc = (np.asarray(spec.ZONE_LO) + np.asarray(spec.ZONE_HI)) / 2
    p2 = zc + np.array([0.05, 0.0, 0.0])
    tr = ClearanceTracker(2)
    pos, v = np.asarray([zc, p2]), np.zeros(2)
    for k in range(4 * spec.CLEAR_REST_TICKS):     # both resting, crowded
        assert tr.update(k * 0.05, pos, v) == []
    assert tr.cleared == 0
    # agent lifts part 1 (held): part 0 is now alone and latches fresh
    for k in range(spec.CLEAR_REST_TICKS - 1):
        assert tr.update(20 + k * 0.05, pos, v, exclude={1}) == []
    assert tr.update(21.0, pos, v, exclude={1}) == [0]
    # part 1 re-delivered alone: latches too
    for k in range(spec.CLEAR_REST_TICKS):
        out = tr.update(22 + k * 0.05, pos, v)
    assert out == [1] and tr.cleared == 2


def test_bin_is_welded_down():
    """No dump-the-bin: bin geoms are static world geometry."""
    model, _ = _one_part_cell(
        [spec.BIN_POS[0], spec.BIN_POS[1], spec.STAND_H + 0.03])
    import mujoco
    for g in ("bin_bottom", "bin_wall_ny", "bin_wall_py",
              "bin_wall_nx", "bin_wall_px"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        assert model.geom_bodyid[gid] == 0


# ---------------------------------------------------------------------------
# Scorer fast paths (the full battery runs in the Harbor verifier)
# ---------------------------------------------------------------------------
def _ep(n, cleared, makespan, floor=0, damage=0):
    return dict(n_parts=n, parts_cleared=cleared, clear_frac=cleared / n,
                makespan_s=makespan, floor_drops=floor, damage_events=damage)


def test_scorer_episode_score_formula():
    from harness import config
    from harness.scorer import (clear_curve, episode_score,
                                           episode_throughput)
    # perfect + at/above the bonus pace cap => exactly 1.0
    fast = _ep(20, 20, makespan=80.0)          # 15 ppm
    assert abs(episode_score(fast) - 1.0) < 1e-9
    # perfect but ordinary pace: full base terms + scaled perfection bonus
    ok = _ep(20, 20, makespan=100.0)           # 12 ppm
    want = (config.PERFECT_BONUS + config.W_CLEAR + config.W_SPEED
            + config.W_SPEED_PERFECT * (12.0 / config.TP_BONUS_CAP_PPM))
    assert abs(episode_score(ok) - want) < 1e-9
    # a single floor drop forfeits BOTH perfection terms
    fumbled = _ep(20, 20, makespan=100.0, floor=1)
    assert episode_score(fumbled) < episode_score(ok) - config.PERFECT_BONUS
    # progressive curve: the easy 60% earns almost nothing
    assert abs(clear_curve(0.6) - config.CLEAR_BASE) < 1e-9
    assert abs(clear_curve(0.3) - config.CLEAR_BASE / 2) < 1e-9
    assert abs(clear_curve(0.8) - (0.1 + 0.9 * 0.5 ** 3)) < 1e-9
    assert clear_curve(1.0) == 1.0
    # stalling early cannot inflate the pace: unfinished bins are paced
    # against the full budget, not the makespan
    stalled = _ep(20, 5, makespan=30.0)
    assert abs(episode_throughput(stalled) - 5 * 60 / spec.EPISODE_T) < 1e-9
    # nothing cleared => zero
    assert episode_score(_ep(20, 0, None)) == 0.0
    # penalties clamp at zero, never mint reward
    wrecked = _ep(20, 2, makespan=170.0, floor=20, damage=20)
    assert episode_score(wrecked) == 0.0


def test_scorer_budgets_smoke_and_aggregates_diagnostics(monkeypatch,
                                                           tmp_path):
    from harness import config, scorer

    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "policy.py").write_text("def make_policy(): pass\n")
    calls = []

    def fake_run(submission_dir, seed, budget_t, wall_budget_s=None,
                 video=None):
        calls.append((budget_t, wall_budget_s))
        log = runtime.EpisodeLog(seed=seed, n_parts=1)
        log.ticks = int(budget_t * spec.CTRL_HZ)
        log.digest = "deterministic"
        diagnostics = {
            "reset_wall_s": 0.1,
            "act_wall_s": 0.4,
            "policy_wall_s": 0.5,
            "n_acts": log.ticks,
            "sandbox_dead": None,
            "sandbox_stderr_tail": "",
            "episode_wall_s": 2.0,
            "harness_wall_s": 1.5,
        }
        return log, diagnostics

    monkeypatch.setattr(scorer, "_run_sandboxed", fake_run)
    report = scorer.score_submission(
        str(submission), seeds={"smoke": 1, "eval": [2]})

    assert calls == [
        (config.SMOKE_T, spec.EPISODE_WALL_BUDGET_S),
        (config.SMOKE_T, spec.EPISODE_WALL_BUDGET_S),
        (spec.EPISODE_T, spec.EPISODE_WALL_BUDGET_S),
    ]
    assert len(report["smoke"]) == 2
    assert report["episode_wall_s_total"] == 2.0
    assert report["policy_wall_s_total"] == 0.5
    assert report["harness_wall_s_total"] == 1.5


def test_scorer_missing_policy_is_gated(tmp_path):
    from harness.scorer import score_submission
    report = score_submission(str(tmp_path))
    assert report["reward"] == 0.0
    assert report["raw_total"] == 0.0     # verifier serializes this field
    assert report["gated"] and not any(report["gate"].values())
    assert "policy.py missing" in report["errors"][0]
