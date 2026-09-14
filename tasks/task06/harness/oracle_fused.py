"""Reference estimator for method-agnostic subtask: RGB-D fusion + temporal tracking.

Combines the RGB analysis-by-synthesis fit (oracle_cv) with the depth
top-surface registration (oracle_geom), weighted by their evidence sizes,
and leans on a constant-velocity motion model through occlusion. This is the
method-agnostic reference: use everything, fuse sensibly.
"""
from __future__ import annotations

import numpy as np

from . import spec
from .oracle_cv import CVEstimator, MIN_MASK_PIXELS, red_mask
from .oracle_geom import FIT_COST_MAX, _Tracker, top_points
from .oracle_unknown import fit_unknown_shape

_FULL_MASK = CVEstimator.FULL_VIEW_PIXELS
_FULL_TOP = 400        # top-surface points of an unoccluded block, roughly


def _blend(a, b, w_b):
    """Pose blend a->b with weight w_b (angles via shortest arc)."""
    dth = spec.wrap_angle(b[2] - a[2])
    return (a[0] + w_b * (b[0] - a[0]),
            a[1] + w_b * (b[1] - a[1]),
            spec.wrap_angle(a[2] + w_b * dth))


class FusedEstimator:
    """method-agnostic interface: reset() / update(rgb, depth, K, T_cam_table, t)."""

    REACQUIRE_MISSES = 8

    def __init__(self):
        self.tracker = _Tracker()
        self.cv = CVEstimator()
        self.misses = 0

    def reset(self):
        self.tracker = _Tracker()
        self.cv = CVEstimator()
        self.misses = 0

    @property
    def shape(self):
        return self.cv.shape

    def update(self, rgb=None, depth=None, K=None, T_cam_table=None,
               t=0.0, **_):
        pred = self.tracker.predict(t)
        init = None if self.misses >= self.REACQUIRE_MISSES else pred
        jump = (0.12 + 0.03 * self.misses, 1.2 + 0.15 * self.misses)

        # Keep the RGB motion history independent: a corrupted depth fit must
        # never become the prior for the robust visual anchor.
        n_mask = int(red_mask(rgb).sum())
        pose_cv = self.cv.update(rgb=rgb, K=K, T_cam_table=T_cam_table, t=t)
        pose_cv = pose_cv if n_mask >= MIN_MASK_PIXELS else None
        obs = top_points(depth)
        pose_gm, cost_gm, _ = fit_unknown_shape(
            obs, prior=init, shape=self.cv.shape, max_jump=jump)
        if cost_gm > FIT_COST_MAX:
            pose_gm = None
        n_top = len(obs)

        w_cv = min(n_mask / _FULL_MASK, 1.0) if pose_cv is not None else 0.0
        w_gm = min(n_top / _FULL_TOP, 1.0) if pose_gm is not None else 0.0

        if w_cv == 0.0 and w_gm == 0.0:
            # no measurement: never feed predictions back into the tracker
            self.misses += 1
            return pred if pred is not None else (0.0, 0.0, 0.0)
        self.misses = 0
        if pose_cv is None:
            meas = pose_gm
        elif pose_gm is None:
            meas = pose_cv
        elif np.hypot(pose_cv[0] - pose_gm[0], pose_cv[1] - pose_gm[1]) > 0.03 \
                or abs(spec.wrap_angle(pose_cv[2] - pose_gm[2])) > 0.26:
            # RGB is independently tracked, so it is the robust anchor when
            # the two modalities disagree.
            meas = pose_cv
        else:
            depth_weight = min(w_gm / (w_cv + w_gm + 1e-9), 0.25)
            meas = _blend(pose_cv, pose_gm, depth_weight)
        evidence = min(w_cv + w_gm, 1.0)
        if pred is not None and evidence < 1.0:
            pose = _blend(pred, meas, evidence)
        else:
            pose = meas
        self.tracker.push(t, pose)
        return pose


def make_estimator():
    return FusedEstimator()
