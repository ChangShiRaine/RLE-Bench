"""Stage B evidence videos: what the estimator saw, with truth and estimate drawn.

Written through rlebench.core.media once scoring is done, one mp4 per
evaluation episode: the RGB the sandboxed estimator received on the left and,
for the variants that expose depth to it, the depth image on the right. Both
panels carry the block's footprint outline at the true pose (green) and the
estimated pose (orange), a heading line, the estimate's centre and a HUD with
the numbers. Evidence for a human, never an input to the score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import episodes, spec
from .checkpoints import CHANNELS, pose_errors

TRUTH = (60, 220, 60)
ESTIMATE = (255, 140, 0)
WARN = (255, 90, 90)
DROPOUT = (110, 0, 0)            # NaN depth: grazing dropout or beyond range
LINE = 2


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def outline_segments(boxes) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Footprint outline of a union of axis-aligned boxes, as 2-D segments in
    the block frame: each box edge minus the parts that lie in another box, so
    the seams where boxes meet are not drawn."""
    rects = [(c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1]) for c, h in boxes]
    eps = 1e-9                     # touching boxes: the seam is interior
    segments = []
    for i, (x0, x1, y0, y1) in enumerate(rects):
        others = rects[:i] + rects[i + 1:]
        for along, fixed, lo, hi in ((0, y0, x0, x1), (0, y1, x0, x1),
                                     (1, x0, y0, y1), (1, x1, y0, y1)):
            pieces = [(lo, hi)]
            for r in others:
                p_lo, p_hi = (r[2], r[3]) if along == 0 else (r[0], r[1])
                a_lo, a_hi = (r[0], r[1]) if along == 0 else (r[2], r[3])
                if not (p_lo - eps <= fixed <= p_hi + eps):
                    continue
                cut = []
                for s0, s1 in pieces:
                    if a_hi <= s0 + eps or a_lo >= s1 - eps:
                        cut.append((s0, s1))
                        continue
                    if s0 < a_lo - eps:
                        cut.append((s0, a_lo))
                    if a_hi < s1 - eps:
                        cut.append((a_hi, s1))
                pieces = cut
            for s0, s1 in pieces:
                if s1 - s0 < 1e-9:
                    continue
                segments.append(((s0, fixed), (s1, fixed)) if along == 0
                                else ((fixed, s0), (fixed, s1)))
    return segments


def project(points_table: np.ndarray) -> np.ndarray:
    """(N, 3) table-frame points -> (N, 2) pixel coordinates; the inverse of
    sensor.backproject."""
    K = spec.intrinsics()
    cam = (np.asarray(points_table, dtype=float) - np.asarray(spec.CAM_POS)) @ spec.camera_rotation()
    depth = -cam[:, 2]
    u = K[0, 2] + K[0, 0] * cam[:, 0] / depth
    v = K[1, 2] - K[1, 1] * cam[:, 1] / depth
    return np.stack([u, v], axis=1)


def _pose_strokes(pose, shape: str) -> list[np.ndarray]:
    """Outline plus a centre-to-stem line for a block at (x, y, theta), as pixel
    polylines on the block's top face."""
    x, y, th = (float(p) for p in pose)
    boxes = spec.BLOCK_COLLISION_BOXES[shape]
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, -s], [s, c]])
    top = 2.0 * max(h[2] for _, h in boxes)

    def to_table(pts2):
        pts2 = np.asarray(pts2, dtype=float) @ rot.T + [x, y]
        return np.column_stack([pts2, np.full(len(pts2), top)])

    strokes = [project(to_table([a, b])) for a, b in outline_segments(boxes)]
    tip = min(c_[1] - h[1] for c_, h in boxes)        # the stem end, block -y
    strokes.append(project(to_table([(0.0, 0.0), (0.0, tip)])))
    return strokes


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _font(size: int):
    from PIL import ImageFont
    try:
        import matplotlib
        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono.ttf"
        return ImageFont.truetype(str(path), size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def depth_panel(depth: np.ndarray) -> np.ndarray:
    """Depth as a grey image, near bright, dropout dark red."""
    d = np.asarray(depth, dtype=np.float32)
    out = np.empty((*d.shape, 3), dtype=np.uint8)
    out[:] = DROPOUT
    finite = np.isfinite(d)
    if finite.any():
        lo, hi = np.percentile(d[finite], [2, 98])
        span = max(float(hi - lo), 1e-6)
        grey = (20 + 235 * np.clip((hi - d[finite]) / span, 0.0, 1.0)).astype(np.uint8)
        out[finite] = grey[:, None]
    return out


def _draw_poses(draw, gt, pred, shape: str) -> None:
    for stroke in _pose_strokes(gt, shape):
        draw.line([tuple(p) for p in stroke], fill=TRUTH, width=LINE)
    if np.all(np.isfinite(pred)):
        strokes = _pose_strokes(pred, shape)
        for stroke in strokes:
            draw.line([tuple(p) for p in stroke], fill=ESTIMATE, width=LINE)
        u, v = strokes[-1][0]
        draw.ellipse([u - 4, v - 4, u + 4, v + 4], fill=ESTIMATE)


def _hud(im, lines, legend_font) -> None:
    from PIL import Image, ImageDraw
    font = _font(15)
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width = max(draw.textlength(text, font=font) for text, _ in lines)
    draw.rectangle([4, 4, 12 + width, 10 + 18 * len(lines)], fill=(0, 0, 0, 170))
    for row, (text, color) in enumerate(lines):
        draw.text((8, 6 + 18 * row), text, fill=(*color, 255), font=font)
    h = im.size[1]
    draw.rectangle([4, h - 26, 250, h - 4], fill=(0, 0, 0, 200))
    draw.line([(12, h - 15), (32, h - 15)], fill=(*TRUTH, 255), width=3)
    draw.text((38, h - 24), "truth", fill=(255, 255, 255, 255), font=legend_font)
    draw.line([(92, h - 15), (112, h - 15)], fill=(*ESTIMATE, 255), width=3)
    draw.text((118, h - 24), "estimate", fill=(255, 255, 255, 255), font=legend_font)
    im.alpha_composite(overlay)


def compose_frame(frame, pred, channels, shape: str, label: str, k: int, n: int) -> np.ndarray:
    """One video frame: RGB panel, plus the depth panel when the estimator gets
    depth. `pred` is the estimated (x, y, theta); NaN marks an invalid estimate."""
    from PIL import Image, ImageDraw
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(frame.gt, dtype=float)
    panels = [np.ascontiguousarray(frame.obs["rgb"][:, :, :3])]
    if "depth" in channels:
        panels.append(depth_panel(frame.obs["depth"]))

    if np.all(np.isfinite(pred)):
        trans, rot = pose_errors(pred[None], gt[None])     # metres, radians
        est = (f"EST  x={pred[0] * 1000:7.1f} y={pred[1] * 1000:7.1f}mm "
               f"th={np.degrees(pred[2]):6.1f}deg")
        err = f"err {float(trans[0]) * 1000:6.1f} mm {np.degrees(float(rot[0])):6.2f} deg"
    else:
        est, err = "EST  invalid", "err  --"
    lines = [
        (label, (255, 255, 255)),
        (f"frame {k + 1:>3}/{n}  t={frame.t:5.2f}s", (255, 255, 255)),
        (f"TRUE x={gt[0] * 1000:7.1f} y={gt[1] * 1000:7.1f}mm th={np.degrees(gt[2]):6.1f}deg", TRUTH),
        (est, ESTIMATE),
        (err, (230, 230, 230)),
        ("occluded", WARN) if frame.occluded else ("visible", TRUTH),
    ]
    out = []
    for i, panel in enumerate(panels):
        im = Image.fromarray(panel).convert("RGBA")
        _draw_poses(ImageDraw.Draw(im), gt, pred, shape)
        if i == 0:
            _hud(im, lines, _font(13))
        out.append(np.asarray(im.convert("RGB")))
    return np.concatenate(out, axis=1)


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------
def render_episode(media, name: str, episode, preds, channels, shape: str, label: str) -> None:
    frames = [f for f in episode.frames if f.obs is not None]
    if not frames:
        raise ValueError("episode carries no observations")
    first = compose_frame(frames[0], preds[0], channels, shape, label, 0, len(frames))
    writer = media.video(name, size=(first.shape[1], first.shape[0]), fps=spec.FRAME_HZ)
    if writer is None:
        return
    try:
        writer.add(first)
        for k, (frame, pred) in enumerate(zip(frames[1:], preds[1:]), start=1):
            writer.add(compose_frame(frame, pred, channels, shape, label, k, len(frames)))
    finally:
        media.finish(writer)


def render_stage_b(media, battery, episode_preds, variant: str, shapes=None) -> None:
    """One mp4 per Stage B episode, named ep<i>_seed<seed>.mp4; each episode
    is guarded on its own so one failure cannot take the rest down."""
    channels = CHANNELS[variant]
    shapes = spec.BLOCK_SHAPES if shapes is None else shapes
    for i, (episode, preds) in enumerate(zip(battery.episodes, episode_preds)):
        shape = episodes.shape_for_seed(episode.seed, shapes)
        label = f"task06{variant}  seed {episode.seed}  ep {i}"
        media.run(f"ep{i}", render_episode, f"ep{i}_seed{episode.seed}.mp4",
                  episode, np.asarray(preds, dtype=float), channels, shape, label)
