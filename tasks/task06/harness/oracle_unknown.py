"""Reference estimator for Task 06's method-agnostic three unnamed block shapes.

The evaluator never supplies a shape id. The estimator jointly registers each
candidate footprint on the first visible depth frame, keeps the winning
hypothesis for the episode, and tracks pose through partial occlusion.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.spatial import cKDTree

from . import spec
from .oracle_geom import (
    COVERAGE_CAP,
    FIT_COST_MAX,
    MIN_POINTS,
    PRIOR_ROT_SCALE,
    PRIOR_WEIGHT,
    _Tracker,
    top_points,
)

SPACING = 0.012
MAX_SHAPE_COST = FIT_COST_MAX

_RECTS = {
    name: tuple(
        (center[0], center[1], half[0], half[1])
        for center, half in spec.BLOCK_COLLISION_BOXES[name]
    )
    for name in spec.BLOCK_SHAPES
}


def _template(shape: str, spacing: float = SPACING) -> np.ndarray:
    points = []
    for cx, cy, hx, hy in _RECTS[shape]:
        xs = np.arange(cx - hx + spacing / 2, cx + hx, spacing)
        ys = np.arange(cy - hy + spacing / 2, cy + hy, spacing)
        points.append(
            np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        )
    return np.unique(np.round(np.concatenate(points), 8), axis=0)


_TEMPLATES = {name: _template(name) for name in spec.BLOCK_SHAPES}


def _centroid(shape: str) -> np.ndarray:
    weighted = np.zeros(2)
    area = 0.0
    for cx, cy, hx, hy in _RECTS[shape]:
        rect_area = 4 * hx * hy
        weighted += rect_area * np.array([cx, cy])
        area += rect_area
    return weighted / area


_CENTROIDS = {name: _centroid(name) for name in spec.BLOCK_SHAPES}


def _to_block(points: np.ndarray, pose) -> np.ndarray:
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    delta = points - np.array([x, y])
    return delta @ np.array([[c, s], [-s, c]]).T


def _outside(points: np.ndarray, pose, shape: str) -> np.ndarray:
    local = _to_block(np.asarray(points, dtype=float), pose)
    distances = []
    for cx, cy, hx, hy in _RECTS[shape]:
        dx = np.maximum(np.abs(local[:, 0] - cx) - hx, 0.0)
        dy = np.maximum(np.abs(local[:, 1] - cy) - hy, 0.0)
        distances.append(np.hypot(dx, dy))
    return np.min(np.stack(distances), axis=0)


def _transform(points: np.ndarray, pose) -> np.ndarray:
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    return points @ rotation.T + np.array([x, y])


def _cost(pose, obs, tree, shape: str, prior=None) -> float:
    coverage, _ = tree.query(_transform(_TEMPLATES[shape], pose), k=1)
    evidence = float(np.mean(_outside(obs, pose, shape) ** 2))
    if prior is None:
        return evidence + float(np.mean(coverage ** 2))
    coverage = np.minimum(coverage, COVERAGE_CAP)
    dx = pose[0] - prior[0]
    dy = pose[1] - prior[1]
    dtheta = spec.wrap_angle(pose[2] - prior[2])
    return evidence + float(np.mean(coverage ** 2)) + PRIOR_WEIGHT * (
        dx * dx + dy * dy + (PRIOR_ROT_SCALE * dtheta) ** 2
    )


def _fit_shape(obs: np.ndarray, shape: str, prior=None):
    tree = cKDTree(obs)
    observed_centroid = obs.mean(axis=0)
    candidates = []
    if prior is not None:
        x0, y0, theta0 = prior
        candidates.extend(
            (x0, y0, theta0 + delta) for delta in (-0.15, 0.0, 0.15)
        )
    for theta in np.linspace(-np.pi, np.pi, 72, endpoint=False):
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s], [s, c]])
        xy = observed_centroid - rotation @ _CENTROIDS[shape]
        candidates.append((xy[0], xy[1], theta))
    starts = sorted(
        candidates,
        key=lambda pose: _cost(pose, obs, tree, shape, prior),
    )[:3]
    best = None
    best_cost = float("inf")
    for start in starts:
        result = optimize.minimize(
            _cost,
            np.asarray(start),
            args=(obs, tree, shape, prior),
            method="Nelder-Mead",
            options={"maxiter": 250, "xatol": 1e-5, "fatol": 1e-10},
        )
        if result.fun < best_cost:
            best = result.x
            best_cost = float(result.fun)
    pose = (
        float(best[0]),
        float(best[1]),
        spec.wrap_angle(float(best[2])),
    )
    evidence = float(np.mean(_outside(obs, pose, shape) ** 2))
    return pose, evidence


def fit_unknown_shape(obs: np.ndarray, prior=None, shape: str | None = None,
                      max_jump=(0.12, 1.2)):
    if obs is None or len(obs) < MIN_POINTS:
        return None, float("inf"), None
    names = (shape,) if shape is not None else spec.BLOCK_SHAPES
    fits = [
        (*_fit_shape(obs, name, prior=prior), name)
        for name in names
    ]
    pose, cost, winner = min(fits, key=lambda item: item[1])
    if prior is not None:
        translation = np.hypot(pose[0] - prior[0], pose[1] - prior[1])
        rotation = abs(spec.wrap_angle(pose[2] - prior[2]))
        if translation > max_jump[0] or rotation > max_jump[1]:
            return None, float("inf"), None
    return pose, cost, winner


class UnknownShapeEstimator:
    REACQUIRE_MISSES = 8
    REACQUIRE_POINTS = 150

    def __init__(self):
        self.reset()

    def reset(self):
        self.tracker = _Tracker()
        self.shape = None
        self.misses = 0

    def update(self, rgb=None, depth=None, K=None, T_cam_table=None,
               t=0.0, **_):
        prediction = self.tracker.predict(t)
        points = top_points(depth)
        prior = prediction
        shape = self.shape
        if self.misses >= self.REACQUIRE_MISSES \
                and len(points) >= self.REACQUIRE_POINTS:
            prior = None
            shape = None
        jump = (0.12 + 0.03 * self.misses, 1.2 + 0.15 * self.misses)
        pose, cost, winner = fit_unknown_shape(
            points, prior=prior, shape=shape, max_jump=jump)
        if pose is None or cost > MAX_SHAPE_COST:
            self.misses += 1
            return prediction if prediction is not None else (0.0, 0.0, 0.0)
        self.misses = 0
        self.shape = winner
        if prediction is not None and len(points) < 80:
            weight = len(points) / 80.0
            dtheta = spec.wrap_angle(pose[2] - prediction[2])
            pose = (
                prediction[0] + weight * (pose[0] - prediction[0]),
                prediction[1] + weight * (pose[1] - prediction[1]),
                spec.wrap_angle(prediction[2] + weight * dtheta),
            )
        self.tracker.push(t, pose)
        return pose


def make_estimator():
    return UnknownShapeEstimator()
