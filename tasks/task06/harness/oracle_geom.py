"""Reference estimator for rgb-depth subtask: depth-only 2D registration (verifier-side).

Pipeline: backproject the (noisy, dropout-ridden) depth image, keep the
block's TOP-surface height band — the side walls are exactly where the
incidence-noise model destroys the data, so they are never trusted — then
register the known T footprint against the surviving 2D points with a coarse
theta scan plus Nelder-Mead refinement on a symmetric chamfer cost.

Also exposes ``fit_top_points`` for reuse by the fused (method-agnostic) estimator.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.spatial import cKDTree

from . import footprint, sensor, spec

TOP_BAND = (0.028, 0.052)     # z window that isolates the block's top face
MAX_POINTS = 600              # deterministic subsample cap
MIN_POINTS = 25               # below this the fit is not attempted
_TEMPLATE = footprint.template_points()
_TEMPLATE_CENTROID = footprint.area_centroid()


def top_points(depth: np.ndarray) -> np.ndarray:
    """(N, 2) table-frame points on the block's top surface.

    Bounded to the region the block can occupy, and excluding a disc around
    the arm base — the Panda's link0 sits on the table inside the tabletop
    bounds and would otherwise contaminate the height band.
    """
    d = np.where(np.isnan(depth), np.inf, depth)
    pts = sensor.backproject(d)
    ok = np.isfinite(pts).all(axis=-1)
    z = pts[..., 2]
    # NOTE: no positional exclusion around the arm base — the block may spawn
    # right next to it, and the tall-structure shadow below already removes
    # the base's own height-band slice
    in_bounds = (np.abs(pts[..., 0]) < 0.40) & (np.abs(pts[..., 1]) < 0.32)
    sel = ok & (z > TOP_BAND[0]) & (z < TOP_BAND[1]) & in_bounds
    p = pts[sel][:, :2]
    # stick exclusion: the block is only 40 mm tall, so structure just above
    # the band at the same (x, y) is the pusher shaft — drop band points in
    # its immediate shadow. Keep the window low and the radius tight: the
    # wrist passes high above the block and must NOT cull block points.
    shaft = ok & in_bounds & (z > 0.072) & (z < 0.22)
    shaft_xy = pts[shaft][:, :2]
    if len(shaft_xy) > 0 and len(p) > 0:
        if len(shaft_xy) > 400:
            shaft_xy = shaft_xy[:: len(shaft_xy) // 400 + 1]
        dist, _ = cKDTree(shaft_xy).query(p, k=1)
        p = p[dist > 0.025]
    if len(p) > 8:
        # density filter: noisy far-table pixels leak into the height band as
        # sparse scatter; the block top is a dense contiguous cluster
        tree = cKDTree(p)
        neighbors = tree.query_ball_point(p, r=0.02, return_length=True)
        p = p[neighbors >= 6]
    if len(p) > MAX_POINTS:
        p = p[:: len(p) // MAX_POINTS + 1]
    return p


COVERAGE_CAP = 0.03   # m; only used WITH a motion prior (tracking mode)
PRIOR_WEIGHT = 0.02   # strength of the motion prior in tracking mode
PRIOR_ROT_SCALE = 0.05  # m per radian: converts dtheta into a distance


def _evidence_cost(pose, obs):
    return float(np.mean(footprint.outside_distance(obs, pose) ** 2))


def _cost(pose, obs, tree_obs, prior=None):
    """Registration cost.

    Without a prior (single-frame): full coverage term — the whole template
    must be explained, which a fully visible top face supports.
    With a prior (tracking): coverage capped so partial (occluded) fragments
    can register, disambiguated by the motion prior — a bare fragment is
    genuinely ambiguous inside a T, the prior is what breaks the tie.
    """
    t = footprint.transform(_TEMPLATE, pose)
    d_cov, _ = tree_obs.query(t, k=1)
    c = _evidence_cost(pose, obs)
    if prior is None:
        return c + float(np.mean(d_cov ** 2))
    d_cov = np.minimum(d_cov, COVERAGE_CAP)
    dx = pose[0] - prior[0]
    dy = pose[1] - prior[1]
    dth = spec.wrap_angle(pose[2] - prior[2])
    return c + float(np.mean(d_cov ** 2)) + PRIOR_WEIGHT * (
        dx * dx + dy * dy + (PRIOR_ROT_SCALE * dth) ** 2)


def fit_top_points(obs: np.ndarray, init=None, max_jump=(0.12, 1.2)):
    """Register the T footprint to observed top-surface points.

    Returns ((x, y, theta), cost) or (None, inf) when evidence is too thin.
    ``init`` warm-starts the search (tracking mode); ``max_jump`` bounds the
    accepted deviation from it (widen it as misses accumulate — after a long
    occlusion the block HAS legitimately moved).
    """
    if obs is None or len(obs) < MIN_POINTS:
        return None, float("inf")
    tree = cKDTree(obs)
    c_obs = obs.mean(axis=0)
    cands = []
    if init is not None:
        x0, y0, th0 = init
        cands += [(x0, y0, th0 + dth) for dth in (-0.15, 0.0, 0.15)]
    for th in np.linspace(-np.pi, np.pi, 72, endpoint=False):
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        t = c_obs - R @ _TEMPLATE_CENTROID
        cands.append((t[0], t[1], th))
    scored = sorted(cands, key=lambda p: _cost(p, obs, tree, init))[:3]
    best, best_cost = None, float("inf")
    for p0 in scored:
        res = optimize.minimize(_cost, np.asarray(p0), args=(obs, tree, init),
                                method="Nelder-Mead",
                                options={"maxiter": 250, "xatol": 1e-5,
                                         "fatol": 1e-10})
        if res.fun < best_cost:
            best, best_cost = res.x, float(res.fun)
    # trimmed re-fit: drop surviving outliers, polish on the inlier set
    keep = footprint.outside_distance(obs, best) < 0.03
    if keep.sum() >= MIN_POINTS and keep.sum() < len(obs):
        obs2 = obs[keep]
        tree2 = cKDTree(obs2)
        res = optimize.minimize(_cost, best, args=(obs2, tree2, init),
                                method="Nelder-Mead",
                                options={"maxiter": 200, "xatol": 1e-5,
                                         "fatol": 1e-10})
        best = res.x
        obs = obs2
    pose = (float(best[0]), float(best[1]), spec.wrap_angle(float(best[2])))
    if init is not None:
        # no-teleport: a tracked fit that leapt away from the prediction is a
        # misregistration of an ambiguous fragment, not the block moving
        if np.hypot(pose[0] - init[0], pose[1] - init[1]) > max_jump[0] \
                or abs(spec.wrap_angle(pose[2] - init[2])) > max_jump[1]:
            return None, float("inf")
    # gate on the EVIDENCE term only: "are the observed points on a T?" —
    # the coverage term legitimately stays high under partial occlusion
    return pose, _evidence_cost(pose, obs)


class _Tracker:
    """Constant-velocity prediction shared by the reference estimators."""

    def __init__(self):
        self.hist = []  # (t, pose)

    EXTRAPOLATION_HORIZON = 0.35  # s of velocity extrapolation past the
    #                               last real measurement, then hold pose

    def predict(self, t):
        if not self.hist:
            return None
        if len(self.hist) == 1:
            return self.hist[-1][1]
        (t0, p0), (t1, p1) = self.hist[-2], self.hist[-1]
        if t1 - t0 <= 1e-9 or t - t1 > self.EXTRAPOLATION_HORIZON:
            return self.hist[-1][1]
        a = (t - t1) / (t1 - t0)
        dth = spec.wrap_angle(p1[2] - p0[2])
        return (p1[0] + a * (p1[0] - p0[0]),
                p1[1] + a * (p1[1] - p0[1]),
                spec.wrap_angle(p1[2] + a * dth))

    def push(self, t, pose):
        """Record a REAL measurement. Never feed predictions back in — a
        motion model fed its own output extrapolates without bound."""
        self.hist.append((t, pose))
        if len(self.hist) > 4:
            self.hist.pop(0)


FIT_COST_MAX = 1.0e-4   # evidence-cost gate: observed points must lie on a T


class GeomEstimator:
    """rgb-depth interface: reset() / update(depth, K, T_cam_table, t) -> pose."""

    REACQUIRE_MISSES = 8      # after this many misses, allow a global refit
    REACQUIRE_POINTS = 150    # ... but only on rich evidence

    def __init__(self):
        self.tracker = _Tracker()
        self.misses = 0

    def reset(self):
        self.tracker = _Tracker()
        self.misses = 0

    def update(self, depth=None, K=None, T_cam_table=None, t=0.0, **_):
        pred = self.tracker.predict(t)
        obs = top_points(depth)
        init = pred
        if self.misses >= self.REACQUIRE_MISSES \
                and len(obs) >= self.REACQUIRE_POINTS:
            init = None   # the block is back in clear view: re-acquire
        jump = (0.12 + 0.03 * self.misses, 1.2 + 0.15 * self.misses)
        pose, cost = fit_top_points(obs, init=init, max_jump=jump)
        if pose is None or cost > FIT_COST_MAX:
            # no usable measurement: output the prediction, keep the motion
            # model's history untouched
            self.misses += 1
            return pred if pred is not None else (0.0, 0.0, 0.0)
        self.misses = 0
        if pred is not None and len(obs) < 80:
            # thin evidence (heavy occlusion): lean on the motion model
            w = len(obs) / 80.0
            dth = spec.wrap_angle(pose[2] - pred[2])
            pose = (pred[0] + w * (pose[0] - pred[0]),
                    pred[1] + w * (pose[1] - pred[1]),
                    spec.wrap_angle(pred[2] + w * dth))
        self.tracker.push(t, pose)
        return pose


def make_estimator():
    return GeomEstimator()
