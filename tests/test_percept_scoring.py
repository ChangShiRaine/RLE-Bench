"""Task06 sandbox and fixed-kernel scoring mechanics."""
import os
import textwrap

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import pytest

from harness import battery, reference, sandbox, scorer
from harness.checkpoints import (CHANNELS, checkpoint_list, kernel_score,
                                 pose_kernel_scores)


@pytest.fixture(scope="module")
def frame():
    from harness import episodes
    return episodes.single_frame(11)


@pytest.fixture(scope="module")
def mini_battery():
    return battery.build_battery(
        seeds=[11], stage_a_count=4, episode_max_frames=20)


def _write_estimator(tmp_path, body):
    (tmp_path / "estimator.py").write_text(textwrap.dedent(body))
    return str(tmp_path)


def test_kernel_matches_task04_formula():
    assert kernel_score(np.array([0.0]), 0.05)[0] == 1.0
    assert kernel_score(np.array([0.05]), 0.05)[0] == pytest.approx(np.exp(-1.0))
    assert kernel_score(np.array([1e-6]), 0.05)[0] < 1.0
    assert kernel_score(np.array([np.inf]), 0.05)[0] == 0.0


def test_pose_kernel_requires_zero_pose_error_for_full_credit():
    gt = np.zeros((2, 3))
    pred = gt.copy()
    assert np.all(pose_kernel_scores(pred, gt) == 1.0)
    pred[1, 0] = 1e-6
    scores = pose_kernel_scores(pred, gt)
    assert scores[0] == 1.0 and scores[1] < 1.0


def test_fixed_shape_contract_and_group_penalty():
    from harness import spec
    from harness.checkpoints import evaluate
    assert spec.BLOCK_SHAPES == ("tshape", "c_block", "f_shape")
    data = dict(gate_ok=True, a_preds=np.zeros((100, 3)),
                a_gts=np.zeros((100, 3)), a_shape_ids=np.arange(100) % 3,
                a_true_ids=np.arange(100) % 3,
                b_episode_preds=[np.zeros((60, 3)) for _ in range(10)],
                b_episode_gts=[np.zeros((60, 3)) for _ in range(10)],
                b_shape_ids=[np.full(60, i % 3) for i in range(10)],
                b_true_ids=[np.full(60, i % 3) for i in range(10)])
    assert evaluate("d", data) == {"G.load": 1., "A.pose": 1., "B.pose": 1.}
    data["b_shape_ids"][0][5] = 1
    assert evaluate("d", data)["B.pose"] == pytest.approx(.9)
    data["a_shape_ids"][5] = -1
    assert evaluate("d", data)["A.pose"] == pytest.approx(.9)
    data["a_shape_ids"] = (data["a_true_ids"] + 1) % 3
    assert evaluate("d", data)["A.pose"] == 0


def test_worst_five_mean_before_kernel():
    from harness.checkpoints import worst_frame_mean, group_score
    assert worst_frame_mean(np.arange(10)) == 7
    assert worst_frame_mean(np.arange(60)) == 57
    assert worst_frame_mean(np.arange(200)) == 197
    gt = np.zeros((60, 3))
    pred = gt.copy()
    pred[-1] = (.01, 0, .1)
    result = group_score(pred, gt, np.zeros(60), np.zeros(60))
    assert result["position_worst5_mean_m"] == .002
    assert result["angle_worst5_mean_rad"] == pytest.approx(.02)
    assert result["score"] == pytest.approx(np.exp(-.04))
    assert "position_p99_m" not in result and "angle_p99_rad" not in result


@pytest.mark.parametrize("errors", [[], [0, np.nan], [0, np.inf], [0, -np.inf]])
def test_worst_five_invalid_errors(errors):
    from harness.checkpoints import worst_frame_mean
    assert np.isinf(worst_frame_mean(errors))


def test_worst_five_short_and_unordered_groups():
    from harness.checkpoints import worst_frame_mean
    assert worst_frame_mean([3]) == 3
    assert worst_frame_mean([3, 1, 2]) == 2
    assert worst_frame_mean([0, 6, 2, 4, 1, 5, 3]) == 4


def test_worst_five_selects_translation_and_rotation_separately():
    from harness.checkpoints import group_score
    gt = np.zeros((10, 3))
    pred = gt.copy()
    pred[:5, 0] = .01
    pred[5:, 2] = .1
    group = group_score(pred, gt, np.zeros(10), np.zeros(10))
    assert group['score'] == pytest.approx(np.exp(-1))
    assert group['position_worst5_mean_m'] == pytest.approx(.01)
    assert group['angle_worst5_mean_rad'] == pytest.approx(.1)


@pytest.mark.parametrize('variant', CHANNELS)
def test_worst_five_scorer_preserves_groups_shapes_and_efficiency(tmp_path, monkeypatch, variant):
    from types import SimpleNamespace
    frame = SimpleNamespace(obs={}, gt=(0., 0., 0.), shape_id=0, occluded=False)
    bat = SimpleNamespace(stage_a=[frame] * 100,
                          episodes=[SimpleNamespace(frames=[frame] * 60) for _ in range(10)])
    (tmp_path / ('model.pt' if variant == 'c' else 'estimator.py')).touch()
    calls = []
    def run(sub, frames, channels, resets=None):
        n = len(frames)
        calls.append(resets)
        poses = np.zeros((n, 3))
        ids = np.zeros(n, dtype=int)
        if n == 100:
            poses[::10, 0] = .05  # Each group's worst-five mean is .01 m.
        elif n == 60:
            poses[0, 2] = .5  # Each episode's worst-five mean is .1 rad.
            if len(calls) == 3:
                ids[1] = 1  # Zero exactly one episode, even on a perfect-pose frame.
        return SimpleNamespace(ok=True, poses=poses, shape_ids=ids, wall_s=n / 5.5)
    monkeypatch.setattr(scorer, '_run', run)
    report = scorer.score_submission(str(tmp_path), variant, battery=bat)
    pose = .5 * (1 + np.exp(-1))
    assert report['accuracy_reward'] == pytest.approx((.3 + .7 * .9) * pose)
    assert report['reward'] == round((.3 + .7 * .9) * pose - .1, 4)
    assert len(report['stage_a_groups']) == len(report['stage_b_groups']) == 10
    assert calls[1] == [True] * 100
    assert all(reset == [True] + [False] * 59 for reset in calls[2:])


@pytest.mark.parametrize("hz,penalty", [(20, 0), (10, 0), (5.5, .1), (1, .2), (.5, .2)])
def test_efficiency_linear_in_hz(hz, penalty):
    from harness.checkpoints import efficiency_deduction
    assert efficiency_deduction(700, 700 / hz) == pytest.approx(penalty)


@pytest.mark.parametrize("value", ["(0,0,0)", "(0,0,0,0.5)", "(0,0,0,True)", "(0,0,0,3)"])
def test_sandbox_rejects_invalid_shape_output(frame, tmp_path, value):
    directory = _write_estimator(tmp_path, f"""
        class E:
            def reset(self): pass
            def update(self, **obs): return {value}
        def make_estimator(): return E()
    """)
    result = sandbox.run_job("code", [frame.obs], ("rgb",), submission_dir=directory)
    assert result.shape_ids.tolist() == [-1]


def test_minimal_checkpoint_surface():
    for variant in CHANNELS:
        cps = checkpoint_list(variant)
        assert [(c.id, c.weight) for c in cps] == [
            ("G.load", 0.0), ("A.pose", 0.3), ("B.pose", 0.7)]
        assert [c.id for c in cps if c.gate] == ["G.load"]


def test_sandbox_runs_stub(frame, tmp_path):
    directory = _write_estimator(tmp_path, """
        class E:
            def reset(self): pass
            def update(self, **obs): return (0.05, -0.02, 0.3, 0)
        def make_estimator(): return E()
    """)
    result = sandbox.run_job("code", [frame.obs], ("rgb", "depth"),
                             submission_dir=directory)
    assert result.ok
    assert np.allclose(result.poses[0], [0.05, -0.02, 0.3])
    assert result.shape_ids.tolist() == [0]


def test_shape_truth_stays_in_verifier(frame, tmp_path):
    from io import BytesIO
    from harness import training_service
    assert frame.shape_id in (0, 1, 2)
    with np.load(BytesIO(training_service._encode([frame], label_seed=11))) as data:
        assert "shape_id" not in data.files and "shape_ids" not in data.files
    sandbox.write_frames(str(tmp_path), [frame.obs], ("rgb",), [True])
    with np.load(tmp_path / "frames.npz") as data:
        assert set(data.files) == {"rgb", "K", "T_cam_table", "ts", "resets"}


def test_scorer_uses_parent_time_and_independent_stage_a(tmp_path, monkeypatch):
    from types import SimpleNamespace
    frame = SimpleNamespace(obs={}, gt=(0., 0., 0.), shape_id=0, occluded=False)
    bat = SimpleNamespace(stage_a=[frame] * 100,
                          episodes=[SimpleNamespace(frames=[frame] * 60) for _ in range(10)])
    (tmp_path / "estimator.py").touch()
    calls = []
    def run(sub, frames, channels, resets=None):
        n = len(frames)
        calls.append(resets)
        return SimpleNamespace(ok=True, poses=np.zeros((n, 3)),
                               shape_ids=np.zeros(n, dtype=int),
                               times=np.zeros(n), wall_s=n / 5.5)
    monkeypatch.setattr(scorer, "_run", run)
    report = scorer.score_submission(str(tmp_path), "d", battery=bat)
    assert report["accuracy_reward"] == 1
    assert report["reward"] == .9
    assert report["inference_hz"] == pytest.approx(5.5)
    assert calls[1] == [True] * 100
    assert all(reset == [True] + [False] * 59 for reset in calls[2:])


def test_sandbox_missing_estimator(frame, tmp_path):
    result = sandbox.run_job("code", [frame.obs], ("rgb",),
                             submission_dir=str(tmp_path))
    assert not result.ok and "missing" in result.error


def test_sandbox_crashing_update_becomes_invalid_pose(frame, tmp_path):
    directory = _write_estimator(tmp_path, """
        class E:
            def reset(self): pass
            def update(self, **obs): raise RuntimeError("boom")
        def make_estimator(): return E()
    """)
    result = sandbox.run_job("code", [frame.obs, frame.obs], ("rgb",),
                             submission_dir=directory)
    assert result.ok and np.isnan(result.poses).all()


def test_sandbox_timeout_kills(frame, tmp_path):
    directory = _write_estimator(tmp_path, """
        import time
        class E:
            def reset(self): pass
            def update(self, **obs):
                time.sleep(60)
                return (0, 0, 0)
        def make_estimator(): return E()
    """)
    result = sandbox.run_job("code", [frame.obs], ("rgb",),
                             submission_dir=directory,
                             frame_timeout_s=2.0, startup_timeout_s=15.0)
    assert not result.ok and "timeout" in result.error


def test_sandbox_input_masking(frame, tmp_path):
    directory = _write_estimator(tmp_path, """
        class E:
            def reset(self): pass
            def update(self, **obs):
                return (float("rgb" in obs), float("depth" in obs), 0.0, 0)
        def make_estimator(): return E()
    """)
    rgb = sandbox.run_job("code", [frame.obs], CHANNELS["a"],
                          submission_dir=directory)
    rgbd = sandbox.run_job("code", [frame.obs], CHANNELS["d"],
                           submission_dir=directory)
    assert np.allclose(rgb.poses[0, :2], [1, 0])
    assert np.allclose(rgbd.poses[0, :2], [1, 1])


def test_sandbox_child_cannot_import_harness(frame, tmp_path):
    directory = _write_estimator(tmp_path, """
        class E:
            def reset(self): pass
            def update(self, **obs):
                try:
                    import harness
                    return (1.0, 0.0, 0.0, 0)
                except ImportError:
                    return (-1.0, 0.0, 0.0, 0)
        def make_estimator(): return E()
    """)
    result = sandbox.run_job("code", [frame.obs], ("rgb",),
                             submission_dir=directory)
    assert result.ok and result.poses[0, 0] == -1.0


def test_empty_submission_fails_load_gate(tmp_path):
    report = scorer.score_submission(str(tmp_path), "d", seeds=[11])
    assert report["reward"] == 0.0
    assert report["gated"] and report["gate_failed"] == ["G.load"]


def test_reference_contract_on_mini_battery(mini_battery, tmp_path):
    reference.build_reference_submission("d", str(tmp_path))
    report = scorer.score_submission(str(tmp_path), "d", battery=mini_battery)
    assert not report["gated"]
    assert report["stage_a_shape_accuracy"] == 1.0
    assert report["stage_b_shape_accuracy"] == 1.0
    assert 0.0 < report["accuracy_reward"] <= 1.0
    assert set(report["checkpoints"]) == {"G.load", "A.pose", "B.pose"}
    assert report["checkpoints"]["B.pose"] == pytest.approx(
        np.mean([g["score"] for g in report["stage_b_groups"]]), abs=1e-4)
    assert report["reward"] == pytest.approx(
        max(0, report["accuracy_reward"] - report["efficiency_deduction"]), abs=1e-4)


def test_invalid_predictions_lose_kernel_credit_without_new_gate(mini_battery, tmp_path):
    _write_estimator(tmp_path, """
        class E:
            def reset(self): pass
            def update(self, **obs): return (5.0, 5.0, 0.0, 0)
        def make_estimator(): return E()
    """)
    report = scorer.score_submission(str(tmp_path), "d", battery=mini_battery)
    assert not report["gated"]
    assert report["checkpoints"]["A.pose"] == 0.0
    assert report["checkpoints"]["B.pose"] == 0.0
    assert report["reward"] == 0.0


def test_nondeterminism_is_not_a_gate(mini_battery, tmp_path):
    _write_estimator(tmp_path, """
        import random
        class E:
            def reset(self): pass
            def update(self, **obs): return (random.random(), 0.0, 0.0, 0)
        def make_estimator(): return E()
    """)
    report = scorer.score_submission(str(tmp_path), "d", battery=mini_battery)
    assert not report["gated"]
