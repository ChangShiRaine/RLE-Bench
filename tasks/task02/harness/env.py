"""The ONE place task02 constructs a RoboCasa environment.

Everything -- the metering daemon, the dev tools, the verifier -- goes through
make_env(). Three reasons:

1. The read-only asset mount needs `rlebench_ro_assets.install()` called before any
   env is built (RoboCasa writes a transient MJCF next to each object's XML). An
   explicit call at every call site was forgotten twice within minutes of being
   introduced in task01, so it is centralised here instead.
2. The RoboCasa test protocol must be honoured verbatim: development on
   split="pretrain", evaluation on split="target", `hard_reset` untouched, horizon
   left at the RoboCasa default. Funnelling construction through one function keeps
   that from drifting per call site.
3. RoboSuite only destroys the old simulator during a hard reset when its renderer is
   named "mujoco". Defaulting that selector here prevents discarded EGL contexts from
   accumulating across episodes while leaving headless offscreen rendering unchanged.

NO SCENE PINNING. Composite classes declare EXCLUDE_LAYOUTS / EXCLUDE_STYLES -- one
excludes layouts whose blender lid is broken, another every layout without a double-door
cabinet -- so a pinned (layout, style) pair is illegal for some tasks and there is no
general way to know which from outside RoboCasa. Evaluation therefore passes a seed and
lets RoboCasa's own sampler draw a legal scene. The pinning branch is deleted rather than
left unused: a code path nobody exercises is one nobody notices breaking.

This module deliberately does NOT import anything from the rest of the harness:
task02 pins its own mujoco/numpy/scipy, which conflict with the repo-wide pins.
"""

from __future__ import annotations

from typing import Any, Iterable

# RoboCasa's own defaults for the three robot cameras. Kept explicit because the
# camera set is part of the observation contract the agent is graded against.
DEFAULT_CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)

# The split names, the size every camera is baked at, the size an unsized observation
# delivers, and the ceiling a spec may ask for. Imported from config rather than
# duplicated, but defaulted here too so this module keeps its promise of not depending on
# the rest of the harness.
try:
    from .config import (DEV_SPLIT, EVAL_SPLIT, OBS_MAX_RESOLUTION, OBS_RESOLUTION,
                         RENDER_RESOLUTION)
except ImportError:  # pragma: no cover - only when used standalone
    DEV_SPLIT, EVAL_SPLIT = "pretrain", "target"     # keep in step with config.py
    OBS_RESOLUTION, OBS_MAX_RESOLUTION = 256, 512
    RENDER_RESOLUTION = OBS_MAX_RESOLUTION


def install_ro_assets_patch() -> str | None:
    """Make a read-only asset mount usable. Idempotent; no-op if unavailable.

    The patch module ships at /opt in the task01 image (PYTHONPATH=/opt). Against a
    writable asset tree (e.g. a dev venv) it is unnecessary, hence the soft import.
    """
    try:
        import rlebench_ro_assets
    except ImportError:
        return None
    return rlebench_ro_assets.install()


def _before_construction() -> None:
    """Everything that must happen before a RoboCasa env exists, in one call.

    Both are easy to forget and neither fails loudly when it is: a missing asset patch
    dies on a read-only mount, and a missing orientation setting silently hands the
    agent upside-down pictures. The module docstring records that the asset patch was
    already forgotten twice at separate call sites, which is why this is one function
    rather than two lines repeated per branch.
    """
    install_ro_assets_patch()
    _use_upright_images()
    _fix_dump_leftovers_predicate()


def _fix_dump_leftovers_predicate() -> None:
    """DumpLeftovers' shipped `_check_success` tests `leftover1` twice and never
    `leftover2`, so the bowl only had to shed one item. Require both, as the instruction
    reads. `stages.dump_leftovers` mirrors this corrected predicate. Idempotent; no-op
    off-simulator."""
    try:
        from robocasa.environments.kitchen.composite.washing_dishes.dump_leftovers import (
            DumpLeftovers,
        )
        import robocasa.utils.object_utils as OU
    except ImportError:
        return
    if getattr(DumpLeftovers, "_rlebench_fixed", False):
        return

    def _check_success(self):
        leftovers_dumped = not any(
            OU.check_obj_in_receptacle(self, name, "bowl")
            for name in ("leftover1", "leftover2")
        )
        bowl_in_sink = OU.obj_inside_of(self, "bowl", self.sink)
        gripper_far = OU.gripper_obj_far(self, obj_name="bowl")
        return leftovers_dumped and bowl_in_sink and gripper_far

    DumpLeftovers._check_success = _check_success
    DumpLeftovers._rlebench_fixed = True


def make_env(
    task: str,
    split: str = DEV_SPLIT,
    seed: int | None = None,
    camera_names: Iterable[str] = DEFAULT_CAMERAS,
    camera_height: int = RENDER_RESOLUTION,
    camera_width: int = RENDER_RESOLUTION,
    camera_depths: bool = False,
    **kwargs: Any,
):
    """Build a RoboCasa env under task02's conventions.

    CAMERAS ARE BAKED AT `RENDER_RESOLUTION`, the ceiling a spec may ask for, so that no
    render ever exceeds the offscreen buffer and forces robosuite to rebuild the live GL
    context (see `render_frames`). Everything smaller is resampled or rendered smaller.

    `split` selects RoboCasa's own train/test boundary. Reset stays `hard_reset=True`
    as shipped, so every reset resamples layout, style, object instances, placements
    and robot base pose, and the horizon stays at the RoboCasa default. The renderer
    selector defaults to "mujoco" so hard reset frees the previous simulator; callers
    may still override it explicitly. Scenes are never pinned -- see the module docstring.

    NOTE: `env.action_spec` is only valid AFTER the first `reset()` -- before it,
    `env.robots[0]` is None and action_spec raises AttributeError.
    """
    kwargs.setdefault("renderer", "mujoco")

    if split not in (DEV_SPLIT, EVAL_SPLIT):
        raise ValueError(
            f"split must be {DEV_SPLIT!r} (development) or {EVAL_SPLIT!r} "
            f"(evaluation), got {split!r}"
        )

    _before_construction()

    from robocasa.utils.env_utils import create_env

    return create_env(
        env_name=task,
        split=split,
        seed=seed,
        camera_names=list(camera_names),
        camera_heights=camera_height,
        camera_widths=camera_width,
        camera_depths=camera_depths,
        **kwargs,
    )


def _macros():
    """robosuite's macros module, or None off-simulator."""
    try:
        import robosuite.macros as macros
        return macros
    except ImportError:
        return None


def image_convention() -> int:
    """The vertical-flip convention, as a numpy step: +1 keeps, -1 flips.

    THE one place orientation is read. robosuite applies this same value to every camera
    observable, so an on-demand render agrees with the streamed observation by
    construction rather than by two flips happening to match.
    """
    macros = _macros()
    if macros is None:
        return 1
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

    return IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]


def _use_upright_images() -> None:
    """Make the simulator hand out UPRIGHT frames, before any env is built.

    MuJoCo reads pixels bottom-up and robosuite's default ("opengl") passes that through,
    so observations are upside down out of the box -- which nothing raises on, and which
    makes a perception task far harder than intended.

    Set at the source rather than flipped at each consumer, so every downstream reader
    (agent observations, render_frames, transcript, debug video) gets upright frames and
    none of them flips again. Assigned after import so it also overrides a
    macros_private.py, which robosuite applies at import time.
    """
    macros = _macros()
    if macros is not None:
        macros.IMAGE_CONVENTION = "opencv"


def allowed_cameras(requested) -> tuple[str, ...]:
    """Narrow a request to the cameras the observation contract publishes.

    ENFORCEMENT, not tidying. `sim.render` resolves any camera name present in the
    MuJoCo model, and a RoboCasa scene carries more of them than the three this task
    exposes -- so an unfiltered name would hand out a viewpoint the task never granted.
    Unknown names are dropped rather than rejected: the request is still honoured for
    whatever part of it was legitimate.
    """
    if requested is None:
        return DEFAULT_CAMERAS
    asked = set(requested)
    return tuple(cam for cam in DEFAULT_CAMERAS if cam in asked)


def resample(frame, width: int, height: int, *, point: bool = False):
    """Resample one rendered frame to (height, width).

    `point=True` takes nearest pixels and is what DEPTH must use: averaging across a
    depth discontinuity invents a surface halfway between the foreground and the
    background, at a range nothing occupies. Colour averages when the ratio divides
    exactly, which is what a native render of the smaller size would have looked like.
    """
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim < 2:
        return arr
    h, w = arr.shape[0], arr.shape[1]
    if (w, h) == (int(width), int(height)):
        return arr
    width, height = max(1, int(width)), max(1, int(height))
    if not point:
        # Pillow's BOX is the same box filter as the reshape below and about seven times
        # faster (1.4 ms against 10 ms for a 512 -> 256 frame), which matters because
        # this is on the observe path. Agreement with the fallback is exact to within
        # one level of rounding. Kept optional so this module still runs off-simulator.
        boxed = _box_filter(arr, width, height)
        if boxed is not None:
            return boxed
    if not point and w % width == 0 and h % height == 0:
        block = arr.reshape(height, h // height, width, w // width, *arr.shape[2:])
        mean = block.mean(axis=(1, 3), dtype=np.float32)
        return (np.rint(mean).astype(arr.dtype)
                if np.issubdtype(arr.dtype, np.integer) else mean.astype(arr.dtype))
    ys = np.minimum(((np.arange(height) + 0.5) * h / height).astype(int), h - 1)
    xs = np.minimum(((np.arange(width) + 0.5) * w / width).astype(int), w - 1)
    return arr[ys][:, xs]


def _box_filter(arr, width: int, height: int):
    """Pillow's box-filtered resize of a uint8 frame, or None if it does not apply."""
    import numpy as np

    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return None
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow ships in the image
        return None
    return np.asarray(Image.fromarray(arr).resize((width, height), Image.BOX))


def _target_size(spec, *, default: int, ceiling: int) -> tuple[int, int, bool]:
    """The size a spec asks for, clamped to `ceiling`, and whether it asked at all.

    Clamped rather than refused: a smaller picture is something an agent can work with,
    an exception is something it has to handle. The flag is what the reply reports the
    resolution as -- a bare number when the agent did not choose one, the pair it asked
    for when it did.
    """
    if spec is None or (spec.width is None and spec.height is None):
        return default, default, False
    w = min(int(spec.width or spec.height), ceiling)
    h = min(int(spec.height or spec.width), ceiling)
    if w <= 0 or h <= 0:
        return default, default, False
    return w, h, True


def apply_obs_spec(env, obs: dict, spec, *, default: int = OBS_RESOLUTION,
                   rendered: int = RENDER_RESOLUTION,
                   ceiling: int = OBS_MAX_RESOLUTION) -> tuple[dict, Any]:
    """Return `obs` shaped to `spec`, plus the resolution delivered.

    The single place a spec becomes pixels -- for `observe()` and for the observations
    streamed to a running controller alike, so the two cannot disagree.

    NOTHING HERE ASKS MUJOCO FOR MORE THAN `rendered`, because the cameras are already
    baked at the ceiling: a square colour frame is RESAMPLED from the one the step
    pipeline rendered (no render at all), and everything else renders at the size asked
    for, which is smaller than the buffer and so never rebuilds the GL context.
    """
    if not obs:
        # A step that was refused or exhausted has nothing to shape, and `None` is what
        # the callers of this path already hand on.
        return obs, default
    want_depth = bool(spec is not None and spec.depth)
    cameras = allowed_cameras(None if spec is None else spec.cameras)
    # `_depth` is dropped alongside `_image`, and for the same reason: both are camera
    # payloads, so both must obey the camera set and the size. Letting a depth key fall
    # through into `others` would smuggle a viewpoint past `allowed_cameras` and past
    # `cameras=()`.
    others = {k: v for k, v in obs.items()
              if not (k.endswith("_image") or k.endswith("_depth"))}
    if not cameras:
        return others, None

    # The buffer is the hard bound, whatever a caller passes as the policy ceiling.
    w, h, sized = _target_size(spec, default=default,
                               ceiling=min(int(ceiling), int(rendered)))
    delivered = [w, h] if sized else default
    # Depth is never in the step pipeline's output (`make_env` leaves camera_depths
    # False), and a non-square frame is a different projection from the square one the
    # pipeline renders rather than a resampling of it. Both render; neither grows the
    # buffer, because the size is bounded by what is baked.
    if want_depth or w != h:
        out = dict(others)
        out.update(render_frames(env, cameras, w, h, depth=want_depth))
        return out, delivered

    # A camera the observation does not carry is omitted rather than rendered: the
    # published set is exactly the baked set, so a missing frame means the episode is
    # over and there is nothing to render against.
    frames = {f"{cam}_image": resample(obs[f"{cam}_image"], w, h)
              for cam in cameras if f"{cam}_image" in obs}
    return {**others, **frames}, delivered


def metric_depth(sim, buffer):
    """Convert MuJoCo's normalised depth buffer to METRES.

    THE RESULT IS Z-DEPTH, NOT RANGE: the distance from the camera PLANE along the optical
    axis, not the distance from the camera along the pixel's own ray. The two differ by
    1/cos(angle from the axis), which is 25% at the edge of a 75-degree frame -- enough to
    put a deprojected point centimetres from the thing it is a point on. A pixel (u, v) at
    z-depth d sits at ((u - cx) * d / f, (v - cy) * d / f, d) in camera coordinates.

    THE RAW BUFFER IS NOT A DISTANCE EITHER. OpenGL depth is `[0, 1]` and non-linear -- almost
    all of its resolution sits near the near plane -- so handing it over unconverted
    would be handing over a number that looks like range and is not one. The inverse
    needs the model's near/far planes, which live on `sim`, so it has to happen here
    rather than in the agent's process.

    robosuite owns the formula (`utils/camera_utils.get_real_depth_map`); it is imported
    when available so a pin bump cannot leave this copy behind, and reproduced only as a
    fallback for running off-simulator.
    """
    import numpy as np

    # A stressed GL context can hand back NaN or out-of-range pixels; sanitise rather
    # than let robosuite's [0, 1] assert throw the whole map away (NaN reads as far).
    buffer = np.asarray(buffer, dtype=np.float32)
    buffer = np.clip(np.nan_to_num(buffer, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    try:
        from robosuite.utils.camera_utils import get_real_depth_map
    except ImportError:
        pass
    else:
        return np.asarray(get_real_depth_map(sim, buffer), dtype=np.float32)

    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    return np.asarray(near / (1.0 - buffer * (1.0 - near / far)), dtype=np.float32)


def render_frames(env, cameras: Iterable[str], width: int, height: int,
                  depth: bool = False, *, ceiling: int = RENDER_RESOLUTION) -> dict:
    """Render `cameras` off-screen at an explicit size, without stepping the sim.

    This is what makes an agent-chosen observation resolution possible: the cameras are
    baked at `RENDER_RESOLUTION` and MuJoCo re-renders the same scene at any SMALLER
    size for free.

    NEVER LARGER. A request above the offscreen buffer sends robosuite through
    `binding_utils.update_offscreen_size`, which frees the live GL context and builds a
    new one mid-run; that rebuild is where the daemon has died outright and where it has
    wedged a worker thread at 100% CPU until the trial was lost. The size is therefore
    clamped here as well as in `apply_obs_spec` -- a no-op for every legitimate caller,
    and the reason a future one cannot reintroduce the path by accident.

    Returns {"<camera>_image": HxWx3 uint8}, oriented exactly as the observation's own
    frames and keyed the same, so a caller can substitute one for the other without
    translating anything. With `depth=True` each camera also yields
    {"<camera>_depth": HxW float32}, in metres, flipped to match its own frame.

    DEPTH IS ALWAYS RENDERED HERE, never sourced from the observation dict: `make_env`
    leaves `camera_depths=False`, so an agent that does not ask for depth pays nothing
    for it on any step. Nothing steps the simulator between an observation being produced
    and this call, so the depth map describes the same state as the colour frame beside
    it.

    Cameras that fail to render are omitted rather than raising -- a missing frame is
    recoverable, a raised observation is not. A camera that renders in colour but not in
    depth is RETRIED ONCE first: see `_render_one`.
    """
    out: dict[str, Any] = {}
    sim = getattr(env, "sim", None)
    if sim is None:
        return out
    width, height = min(int(width), int(ceiling)), min(int(height), int(ceiling))
    convention = image_convention()
    for cam in cameras:
        rgb, dmap = _render_one(sim, cam, width, height, depth)
        # Asked for depth, got a picture and no depth: render that camera again before
        # handing back half of what was requested. It happens on the first depth read
        # against an offscreen context that has just been (re)created -- the first
        # observation of a freshly reset episode, and every observation of a finished one
        # -- and the second read finds the context warm. A retry is
        # cheap, it happens only on the failing path, and the alternative is a caller
        # discovering the gap as a `KeyError` several frames downstream.
        if depth and rgb is not None and dmap is None:
            rgb, dmap = _render_one(sim, cam, width, height, depth)
        if rgb is None:
            continue
        if dmap is not None:
            out[f"{cam}_depth"] = dmap[::convention]
        out[f"{cam}_image"] = rgb[::convention]
    return out


def _render_one(sim, cam: str, width: int, height: int, depth: bool):
    """One camera -> (colour frame, depth map in metres). Either may be None if it failed.

    Unflipped: `render_frames` owns the convention. Separated out so the depth retry is
    one call rather than a copy of the render.
    """
    try:
        frame = sim.render(camera_name=cam, width=int(width), height=int(height),
                           depth=bool(depth))
    except Exception as exc:  # noqa: BLE001
        _log_render_failure(cam, exc)
        return None, None
    if not depth:
        return frame, None
    # robosuite hands back (rgb, buffer) in this mode. Anything else means the depth read
    # did not happen, and it must NOT be unpacked on faith: destructuring a bare HxWx3
    # frame binds two ROWS OF PIXELS to (rgb, buffer), so the caller gets a two-pixel-tall
    # "image" and a "depth map" made of colour -- silently, and only when depth is broken.
    if not (isinstance(frame, tuple) and len(frame) == 2):
        return frame, None
    rgb, buffer = frame
    try:
        return rgb, metric_depth(sim, buffer)
    except Exception as exc:  # noqa: BLE001
        _log_render_failure(cam, exc)
        return rgb, None


def _log_render_failure(cam: str, exc: Exception) -> None:
    """Daemon-side stderr, never the agent: a dropped frame must leave a trace."""
    import sys

    print(f"[render] {cam}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
