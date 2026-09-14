"""Naive baselines (verifier-side): the calibration bracket's failing end.

Each baseline embodies the obvious first attempt that the task's difficulty
mechanisms are designed to defeat. Calibration requires them to score far
below the reference estimators; if one stops failing, the task got too easy.
"""
from __future__ import annotations

import numpy as np

from . import footprint, sensor, spec
from .oracle_cv import pixel_to_plane, red_mask


class NaiveColorEstimator:
    """Color centroid + PCA on the mask, cast to the table plane.

    Ignores the 3D extrusion: the shaded side panels merge into the mask, so
    the centroid is biased toward the camera and the principal axis is both
    skewed and 180-degree ambiguous.
    """

    def reset(self):
        pass

    def update(self, rgb=None, K=None, T_cam_table=None, t=0.0, **_):
        mask = red_mask(rgb) if rgb is not None else None
        if mask is None or mask.sum() < 20:
            return (0.0, 0.0, 0.0)
        pix = np.argwhere(mask)
        pts = np.array([pixel_to_plane(u, v, spec.BLOCK_HEIGHT / 2)
                        for v, u in pix[:: max(1, len(pix) // 400)]])
        c = pts.mean(axis=0)
        d = pts - c
        cov = d.T @ d / len(d)
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, np.argmax(evals)]
        theta = float(np.arctan2(major[1], major[0]))
        # centroid of the T footprint is stem-ward of the block origin
        off = footprint.area_centroid()
        cc, ss = np.cos(theta), np.sin(theta)
        origin = c - np.array([[cc, -ss], [ss, cc]]) @ off
        return (float(origin[0]), float(origin[1]), spec.wrap_angle(theta))


class RawICPEstimator:
    """Vanilla point-to-point ICP of the raw noisy cloud against the block.

    No incidence filtering, no height band, single initialization: it locks
    onto the corrupted side-wall plumes and band-leaking table points.
    """

    ITERS = 15

    def __init__(self):
        self._template = footprint.surface_samples(0.02)

    def reset(self):
        pass

    def update(self, depth=None, K=None, T_cam_table=None, t=0.0, **_):
        from scipy.spatial import cKDTree
        d = np.where(np.isnan(depth), np.inf, depth)
        pts = sensor.backproject(d)
        ok = np.isfinite(pts).all(axis=-1) & (pts[..., 2] > 0.012) \
            & (np.abs(pts[..., 0]) < 0.42) & (np.abs(pts[..., 1]) < 0.33)
        cloud = pts[ok]
        if len(cloud) < 30:
            return (0.0, 0.0, 0.0)
        if len(cloud) > 800:
            cloud = cloud[:: len(cloud) // 800 + 1]
        tree = cKDTree(cloud)
        x, y, th = float(cloud[:, 0].mean()), float(cloud[:, 1].mean()), 0.0
        tmpl = self._template
        for _ in range(self.ITERS):
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            tw = tmpl @ R.T + np.array([x, y, 0.0])
            _, idx = tree.query(tw, k=1)
            target = cloud[idx]
            # 2D Kabsch on the xy components
            a = tw[:, :2] - tw[:, :2].mean(axis=0)
            b = target[:, :2] - target[:, :2].mean(axis=0)
            H = a.T @ b
            U, _, Vt = np.linalg.svd(H)
            sign = np.sign(np.linalg.det(Vt.T @ U.T))
            Rk = Vt.T @ np.diag([1, sign]) @ U.T
            dth = float(np.arctan2(Rk[1, 0], Rk[0, 0]))
            th = spec.wrap_angle(th + dth)
            c, s = np.cos(th), np.sin(th)
            R2 = np.array([[c, -s], [s, c]])
            centr_t = tmpl[:, :2].mean(axis=0)
            x, y = target[:, :2].mean(axis=0) - R2 @ centr_t
        return (float(x), float(y), float(th))


class LastPoseEstimator:
    """Estimates once, then freezes — the anti-gaming probe for Stage B.

    The block keeps moving while occluded, so this must score clearly worse
    than a real tracker on push episodes.
    """

    def __init__(self):
        from .oracle_geom import GeomEstimator
        self._inner = GeomEstimator()
        self._frozen = None

    def reset(self):
        self._inner.reset()
        self._frozen = None

    def update(self, **obs):
        if self._frozen is None:
            self._frozen = self._inner.update(**obs)
        return self._frozen
