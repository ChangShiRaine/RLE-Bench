"""Reference estimator for rgb-only subtask: RGB-only analysis-by-synthesis (verifier-side).

A naive color threshold merges the block's bright top face with its shaded
side panels, so the silhouette is NOT the 2D T footprint. This estimator
therefore renders the hypothesis forward: for a candidate pose it projects
the full 3D block (both prisms, side walls included) through the published
camera model and scores the predicted silhouette against the observed color
mask — mutual coverage, maximized over (x, y, theta).
"""
from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.spatial import ConvexHull

from . import spec

MIN_MASK_PIXELS = 60
MAX_MASK_SAMPLES = 2500
def _shape_geometry(name: str, spacing: float = 0.02):
    corners = []
    surface = []
    weighted = np.zeros(2)
    area = 0.0
    for center, half in spec.BLOCK_COLLISION_BOXES[name]:
        cx, cy = center[:2]
        hx, hy = half[:2]
        corners.append(np.asarray([
            [cx + sx * hx, cy + sy * hy, z]
            for z in (0.0, spec.BLOCK_HEIGHT)
            for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))
        ]))
        xs = np.arange(cx - hx + spacing / 2, cx + hx, spacing)
        ys = np.arange(cy - hy + spacing / 2, cy + hy, spacing)
        if len(xs) and len(ys):
            xy = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
            surface.append(np.column_stack(
                [xy, np.full(len(xy), spec.BLOCK_HEIGHT)]))
        rect_area = 4 * hx * hy
        weighted += rect_area * np.asarray(center[:2])
        area += rect_area
    return tuple(corners), np.concatenate(surface), weighted / area


_SHAPE_GEOMETRY = {name: _shape_geometry(name) for name in spec.BLOCK_SHAPES}
_CORNERS = {name: value[0] for name, value in _SHAPE_GEOMETRY.items()}
_SURFACE = {name: value[1] for name, value in _SHAPE_GEOMETRY.items()}
_CENTROID = {name: value[2] for name, value in _SHAPE_GEOMETRY.items()}


def red_mask(rgb: np.ndarray) -> np.ndarray:
    """Block segmentation by color ratio (robust to the lighting nuisance)."""
    r = rgb.astype(np.float32)
    return (r[..., 0] > 60) & (r[..., 0] > 1.35 * r[..., 1]) \
        & (r[..., 0] > 1.35 * r[..., 2])


def project(pts_world: np.ndarray) -> np.ndarray:
    """World points -> (N, 2) pixel coordinates via the published camera."""
    K = spec.intrinsics()
    R = spec.camera_rotation()
    pc = (np.asarray(pts_world) - np.asarray(spec.CAM_POS)) @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 2] + K[0, 0] * pc[:, 0] / -pc[:, 2]
        v = K[1, 2] - K[1, 1] * pc[:, 1] / -pc[:, 2]
    return np.column_stack([u, v])


def pixel_to_plane(u: float, v: float, z0: float) -> np.ndarray:
    """Back-cast one pixel to the horizontal plane z=z0 (table frame)."""
    K = spec.intrinsics()
    R = spec.camera_rotation()
    d_cam = np.array([(u - K[0, 2]) / K[0, 0],
                      -(v - K[1, 2]) / K[1, 1], -1.0])
    d = R @ d_cam
    t = (z0 - spec.CAM_POS[2]) / d[2]
    p = np.asarray(spec.CAM_POS) + t * d
    return p[:2]


def _pose_to_world(pose, pts_block3d):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pts_block3d @ R.T + np.array([x, y, 0.0])


def _hull_edges(px: np.ndarray):
    """Convex hull of projected points -> (edge origins, inward normals)."""
    hull = ConvexHull(px)
    v = px[hull.vertices]                     # CCW order
    e = np.roll(v, -1, axis=0) - v
    normals = np.column_stack([-e[:, 1], e[:, 0]])
    return v, normals


def _contains(v, normals, pts):
    d = pts[:, None, :] - v[None, :, :]
    return np.all(np.einsum("pen,en->pe", d, normals) >= 0.0, axis=1)


def silhouette_test(pose, pts_uv: np.ndarray, shape: str = "tshape") -> np.ndarray:
    """Which pixels fall inside the projected block silhouette at ``pose``."""
    inside = np.zeros(len(pts_uv), dtype=bool)
    for prism in _CORNERS[shape]:
        px = project(_pose_to_world(pose, prism))
        if not np.isfinite(px).all():
            continue
        v, n = _hull_edges(px)
        inside |= _contains(v, n, pts_uv)
    return inside


class _Evidence:
    def __init__(self, rgb):
        self.mask = red_mask(rgb)
        pix = np.argwhere(self.mask)          # (v, u)
        self.count = len(pix)
        if self.count > MAX_MASK_SAMPLES:
            pix = pix[:: self.count // MAX_MASK_SAMPLES + 1]
        self.pts_uv = pix[:, ::-1].astype(float)  # (u, v)
        self.h, self.w = self.mask.shape

    def coverage(self, pose, shape: str = "tshape") -> float:
        """Fraction of predicted block-surface probes landing on the mask."""
        px = project(_pose_to_world(pose, _SURFACE[shape]))
        u = np.clip(np.round(px[:, 0]).astype(int), 0, self.w - 1)
        v = np.clip(np.round(px[:, 1]).astype(int), 0, self.h - 1)
        return float(np.mean(self.mask[v, u]))

    def inlier_frac(self, pose, shape: str = "tshape") -> float:
        if len(self.pts_uv) == 0:
            return 0.0
        return float(np.mean(silhouette_test(pose, self.pts_uv, shape)))


PRIOR_WEIGHT = 20.0     # motion-prior strength in the (dimensionless) cost
PRIOR_ROT_SCALE = 0.05  # m per radian


def _cost(pose, ev: _Evidence, prior=None, shape: str = "tshape") -> float:
    c = (1.0 - ev.inlier_frac(pose, shape)) + (1.0 - ev.coverage(pose, shape))
    if prior is not None:
        # under partial occlusion a bar-only mask supports theta and
        # theta+pi almost equally — the motion prior breaks the tie
        dx = pose[0] - prior[0]
        dy = pose[1] - prior[1]
        dth = spec.wrap_angle(pose[2] - prior[2])
        c += PRIOR_WEIGHT * (dx * dx + dy * dy
                             + (PRIOR_ROT_SCALE * dth) ** 2)
    return c


def fit_mask(rgb: np.ndarray, init=None, max_jump=(0.12, 1.2),
             shape: str = "tshape"):
    """Fit (x, y, theta) to the observed color mask; (pose, cost) or None.

    ``max_jump`` bounds accepted deviation from ``init`` (widen with misses)."""
    ev = _Evidence(rgb)
    if ev.count < MIN_MASK_PIXELS:
        return None, float("inf"), ev.count
    cu, cv = ev.pts_uv.mean(axis=0)
    # cast the mask centroid onto the block's mid-height plane; correct for
    # the footprint centroid sitting stem-ward of the block-frame origin
    c_xy = pixel_to_plane(cu, cv, spec.BLOCK_HEIGHT / 2)
    cands = []
    if init is not None:
        x0, y0, th0 = init
        cands += [(x0, y0, th0 + d) for d in (-0.15, 0.0, 0.15)]
    tmpl_c = _CENTROID[shape]
    for th in np.linspace(-np.pi, np.pi, 72, endpoint=False):
        c, s = np.cos(th), np.sin(th)
        off = np.array([[c, -s], [s, c]]) @ tmpl_c
        cands.append((c_xy[0] - off[0], c_xy[1] - off[1], th))
    scored = sorted(cands, key=lambda p: _cost(p, ev, init, shape))[:3]
    best, best_cost = None, float("inf")
    for p0 in scored:
        res = optimize.minimize(_cost, np.asarray(p0), args=(ev, init, shape),
                                method="Nelder-Mead",
                                options={"maxiter": 250, "xatol": 1e-5,
                                         "fatol": 1e-9})
        if res.fun < best_cost:
            best, best_cost = res.x, float(res.fun)
    pose = (float(best[0]), float(best[1]), spec.wrap_angle(float(best[2])))
    if init is not None:
        # no-teleport: the block can't jump between frames; a leaping fit is
        # a misregistration of ambiguous (occluded) evidence
        if np.hypot(pose[0] - init[0], pose[1] - init[1]) > max_jump[0] \
                or abs(spec.wrap_angle(pose[2] - init[2])) > max_jump[1]:
            return None, float("inf"), ev.count
    return pose, best_cost, ev.count


class CVEstimator:
    """rgb-only interface: reset() / update(rgb, K, T_cam_table, t) -> pose."""

    FULL_VIEW_PIXELS = 3000   # mask size of an unoccluded block, roughly
    REACQUIRE_MISSES = 8
    REACQUIRE_PIXELS = 2200

    def __init__(self):
        from .oracle_geom import _Tracker
        self._tracker_cls = _Tracker
        self.tracker = _Tracker()
        self.misses = 0
        self.shape = None

    def reset(self):
        self.tracker = self._tracker_cls()
        self.misses = 0
        self.shape = None

    def update(self, rgb=None, K=None, T_cam_table=None, t=0.0, **_):
        pred = self.tracker.predict(t)
        init = pred
        if self.misses >= self.REACQUIRE_MISSES \
                and red_mask(rgb).sum() >= self.REACQUIRE_PIXELS:
            init = None   # block back in clear view: global re-acquisition
        jump = (0.12 + 0.03 * self.misses, 1.2 + 0.15 * self.misses)
        shapes = (self.shape,) if self.shape is not None else spec.BLOCK_SHAPES
        fits = [(*fit_mask(rgb, init=init, max_jump=jump, shape=shape),
                 shape) for shape in shapes]
        pose, cost, count, winner = min(fits, key=lambda item: item[1])
        if pose is None:
            # no usable measurement: don't feed predictions back into the
            # motion model — that extrapolates without bound
            self.misses += 1
            return pred if pred is not None else (0.0, 0.0, 0.0)
        self.misses = 0
        self.shape = winner
        if pred is not None and count < self.FULL_VIEW_PIXELS:
            w = min(count / self.FULL_VIEW_PIXELS, 1.0)
            dth = spec.wrap_angle(pose[2] - pred[2])
            pose = (pred[0] + w * (pose[0] - pred[0]),
                    pred[1] + w * (pose[1] - pred[1]),
                    spec.wrap_angle(pred[2] + w * dth))
        self.tracker.push(t, pose)
        return pose


def make_estimator():
    return CVEstimator()
