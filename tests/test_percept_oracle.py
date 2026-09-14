"""Dev tests for the task06 reference estimators and baselines.

The golden-first bracket: each method-matched reference must succeed on its
own modality, and each naive baseline must clearly fail. Thresholds here are
looser than calibration will pin (they guard against regressions, not tune
scores).
"""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import pytest

# Every test here consumes real OSMesa renders (per-seed single frames or the
# fully rendered push episode): minutes on a CPU runner. PR CI skips `heavy`;
# heavy-nightly.yml runs the heavy tests.
pytestmark = pytest.mark.heavy

from harness import baselines, battery, checkpoints, episodes, spec
from harness import oracle_cv, oracle_fused, oracle_geom, oracle_unknown

SEEDS = (11, 23, 37, 101, 202, 303, 404)


def _errs(pose, gt):
    return (np.hypot(pose[0] - gt[0], pose[1] - gt[1]),
            abs(spec.wrap_angle(pose[2] - gt[2])))


@pytest.fixture(scope="module")
def frames():
    return [episodes.single_frame(s) for s in SEEDS]


@pytest.fixture(scope="module")
def episode23():
    return episodes.push_episode(23, render=True,
                                 max_frames=battery.EPISODE_MAX_FRAMES)


def _run_single(est, frames):
    out = []
    for f in frames:
        est.reset()
        out.append(_errs(est.update(**f.obs), f.gt))
    return np.array(out)


def _run_episode(est, ep):
    est.reset()
    return np.array([_errs(est.update(**f.obs), f.gt) for f in ep.frames])


# ---------------------------------------------------------------------------
# Single-frame accuracy (Stage-A regime)
# ---------------------------------------------------------------------------
def test_cv_single_frame_accuracy(frames):
    e = _run_single(oracle_cv.make_estimator(), frames)
    assert np.median(e[:, 0]) < 0.006
    assert np.median(e[:, 1]) < np.deg2rad(1.5)
    assert e[:, 0].max() < 0.015
    assert e[:, 1].max() < np.deg2rad(5.0)


def test_geom_single_frame_accuracy(frames):
    e = _run_single(oracle_geom.make_estimator(), frames)
    assert np.median(e[:, 0]) < 0.010
    assert np.median(e[:, 1]) < np.deg2rad(3.0)
    assert e[:, 0].max() < 0.020
    assert e[:, 1].max() < np.deg2rad(12.0)


def test_fused_single_frame_accuracy(frames):
    e = _run_single(oracle_fused.make_estimator(), frames)
    assert np.median(e[:, 0]) < 0.008
    assert e[:, 0].max() < 0.020


def test_estimators_deterministic(frames):
    f = frames[0]
    est = oracle_cv.make_estimator()
    est.reset()
    p1 = est.update(**f.obs)
    est.reset()
    p2 = est.update(**f.obs)
    assert p1 == p2


# ---------------------------------------------------------------------------
# Baselines must clearly fail (the difficulty mechanisms work)
# ---------------------------------------------------------------------------
def test_naive_color_fails(frames):
    e = _run_single(baselines.NaiveColorEstimator(), frames)
    assert np.median(e[:, 0]) > 0.02 or np.median(e[:, 1]) > np.deg2rad(15)


def test_raw_icp_fails(frames):
    e = _run_single(baselines.RawICPEstimator(), frames)
    assert np.median(e[:, 0]) > 0.02 or np.median(e[:, 1]) > np.deg2rad(15)


def test_reference_beats_baseline_by_construction(frames):
    ref = _run_single(oracle_cv.make_estimator(), frames)
    naive = _run_single(baselines.NaiveColorEstimator(), frames)
    assert np.median(ref[:, 0]) * 4 < np.median(naive[:, 0])


# ---------------------------------------------------------------------------
# Episode tracking (Stage-B regime)
# ---------------------------------------------------------------------------
def test_tracking_through_occlusion(episode23):
    flags = np.array(episode23.occluded_flags)
    assert flags.mean() > 0.2, "episode 23 should be heavily occluded"

    e_cv = _run_episode(oracle_cv.make_estimator(), episode23)
    e_gm = _run_episode(oracle_geom.make_estimator(), episode23)
    e_depth = _run_episode(oracle_unknown.make_estimator(), episode23)

    # visible frames: everyone competent
    for e in (e_cv, e_gm, e_depth):
        assert np.median(e[~flags][:, 0]) < 0.025

    # The shipped references stay bounded during off-center, occluded pushes.
    for e in (e_cv, e_depth):
        assert np.median(e[flags][:, 0]) < 0.05
        assert np.median(e[flags][:, 1]) < np.deg2rad(20)
    assert np.median(e_gm[flags][:, 0]) < 0.16
    assert np.median(e_gm[flags][:, 1]) < 2.2, "runaway/flip lock"


def test_last_pose_fails_on_moving_occluded_block(episode23):
    flags = np.array(episode23.occluded_flags)
    truth = np.asarray(episode23.gts)
    scores = []
    for est in (baselines.LastPoseEstimator(), oracle_fused.make_estimator()):
        est.reset()
        pred = np.asarray([est.update(**frame.obs) for frame in episode23.frames])
        scores.append(checkpoints.mean_pose_kernel(pred[flags], truth[flags]))
    assert scores[1] > scores[0] + 0.10
    errors = _run_episode(baselines.LastPoseEstimator(), episode23)
    assert np.median(errors[flags][:, 1]) > np.deg2rad(30)
