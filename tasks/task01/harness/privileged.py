"""Ground truth, for the L3 harness only.

L1 hands the agent three cameras and proprioception: where things are has to be
perceived, as a real robot would. This module is the deliberate exception -- at L3 the
agent is *given* the poses instead, so a run can measure what perception was costing it.

IT SHIPS ONLY IN THE ROOT-ONLY TREE. `tasks/task01/build_assets.py` lists it in
FORBIDDEN_IN_AGENT and asserts the boundary at build time, at every level including L3:
the agent receives this state over the metered socket, never as a module it can import
and call on an env of its own.

WHERE THE POSES COME FROM, and why not from MuJoCo directly. RoboCasa's own
`_setup_observables` already writes every object and fixture pose into the observation
dict -- `<name>_pos`, `<name>_quat`, `<name>_to_robot0_eef_pos`, and the `object-state`
concatenation of them. `config.agent_visible_obs` is what strips them out. So the honest
definition of "privileged" is *exactly what that filter drops*, and taking it from there
means the numbers the L3 agent sees are the simulator's own, computed at the same instant
as the frame beside them, rather than a second derivation that could drift from it.

Facts the observation does NOT carry -- grasp state, fixture extents, which object the
task is about -- are read off the live env, each independently and each optional. A
missing one is omitted rather than raised: an incomplete privileged block still leaves
the agent better off than L1, whereas an exception would fail an otherwise good trial.

Everything is prefixed `priv_`, which cannot collide with the `robot\\d+_` keys
`agent_visible_obs` keeps.

QUATERNIONS ARE xyzw, matching every `_quat` key the agent already has. MuJoCo stores
them wxyz, and robosuite converts on the way out (`convert_quat(..., to="xyzw")` in every
`_quat` observable, robocasa's `obj_quat` included), so anything read straight off
`sim.data.body_xquat` must be converted here or the agent is handed two conventions in
one observation and told they are one.
"""

from __future__ import annotations

from typing import Any

# The concatenation of the per-object keys. Dropped rather than forwarded: it duplicates
# what `priv_object_state` already carries key by key, in an order the agent would have
# to reverse-engineer to use.
_CONCATENATED = "object-state"

# RoboCasa names the manipulated object of an atomic task `obj` in `object_cfgs`;
# everything else in the scene is a distractor. Checked by name rather than by position
# in the list, which is not stable.
_TARGET_NAME = "obj"


def _plain(value: Any) -> Any:
    """A JSON/ndarray-safe copy. `protocol.encode` handles ndarray and numpy scalars;
    anything else exotic is stringified rather than allowed to break a whole reply."""
    import numpy as np

    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return str(value)


def _object_state(obs: dict, keep: Any) -> dict:
    """The pose keys `keep` rejects -- which is the whole of RoboCasa's object state."""
    return {k: v for k, v in obs.items()
            if k != _CONCATENATED and not keep.match(k)}


def _target(env) -> dict:
    """Which object the task is about, and where it is.

    `object_cfgs` is RoboCasa's own scene manifest, so the target is looked up rather
    than guessed. Returns {} when the task manipulates no object (TurnOnStove and the
    other fixture-only tasks), which is correct: there is nothing to point at.
    """
    cfgs = getattr(env, "object_cfgs", None) or []
    for cfg in cfgs:
        if str(cfg.get("name", "")) != _TARGET_NAME:
            continue
        out = {"name": _TARGET_NAME}
        groups = cfg.get("obj_groups")
        if groups is not None:
            out["obj_groups"] = _plain(groups)
        placement = cfg.get("placement") or {}
        fixture = placement.get("fixture")
        if fixture is not None:
            out["placed_on"] = str(getattr(fixture, "name", fixture))
        return out
    return {}


def _xyzw(quat) -> Any:
    """A MuJoCo wxyz quaternion in the xyzw order the rest of the observation uses.

    The same reindex robosuite's `convert_quat(..., to="xyzw")` performs, open-coded so
    this module keeps working without robosuite imported.
    """
    import numpy as np

    q = np.asarray(quat, dtype=float).reshape(4)
    return q[[1, 2, 3, 0]]


def _body_pose(env, body: str) -> dict:
    """One body's world pose, straight from MuJoCo -- quaternion converted to xyzw."""
    sim = env.sim
    bid = sim.model.body_name2id(body)
    return {"pos": sim.data.body_xpos[bid].copy(),
            "quat": _xyzw(sim.data.body_xquat[bid])}


def _objects(env) -> dict:
    """Every object in the scene by name, with its world pose.

    The full list, not just the target: identifying which of them matters is left to
    the agent even at L3, so the privileged information is *localisation*, never
    task understanding.
    """
    out: dict[str, Any] = {}
    for name, obj in (getattr(env, "objects", None) or {}).items():
        root = getattr(obj, "root_body", None)
        if root is None:
            continue
        try:
            out[str(name)] = _body_pose(env, root)
        except Exception:  # noqa: BLE001 - one unreadable body must not lose the rest
            continue
    return out


def _fixtures(env) -> dict:
    """Fixtures -- the receptacles and appliances -- with pose and extents.

    Extents are what a pose alone cannot give you: the drawer's pose says nothing about
    where its opening is, and a place target needs the box.
    """
    out: dict[str, Any] = {}
    for name, fix in (getattr(env, "fixtures", None) or {}).items():
        entry: dict[str, Any] = {}
        for attr, key in (("pos", "pos"), ("quat", "quat"), ("size", "size")):
            value = getattr(fix, attr, None)
            if value is None:
                continue
            # The fixture's own attributes come off the MJCF, where quaternions are
            # wxyz. Converted on the way out for the same reason _body_pose is.
            entry[key] = _xyzw(value) if key == "quat" else _plain(value)
        root = getattr(fix, "root_body", None)
        if root is not None:
            try:
                entry.update(_body_pose(env, root))
            except Exception:  # noqa: BLE001
                pass
        if entry:
            out[str(name)] = entry
    return out


def _grasp(env) -> dict:
    """Whether the gripper is actually holding each object, and the finger contacts.

    `_check_grasp` is robosuite's own predicate (both fingers in opposing contact), not
    a distance heuristic -- so "grasped" here means the same thing the simulator means.
    """
    robot = (getattr(env, "robots", None) or [None])[0]
    gripper = getattr(robot, "gripper", None)
    if gripper is None:
        return {}
    # Multi-arm robots hand back a dict of grippers; the kitchen robots are single-arm,
    # so take the one arm's gripper and leave anything else alone.
    if isinstance(gripper, dict):
        gripper = next(iter(gripper.values()), None)
        if gripper is None:
            return {}

    held = []
    for name, obj in (getattr(env, "objects", None) or {}).items():
        try:
            if env._check_grasp(gripper=gripper,
                                object_geoms=obj.contact_geoms):
                held.append(str(name))
        except Exception:  # noqa: BLE001
            continue

    out: dict[str, Any] = {"grasped_objects": held, "grasping": bool(held)}
    fingers = getattr(gripper, "important_geoms", None) or {}
    for side in ("left_finger", "right_finger"):
        geoms = fingers.get(side)
        if not geoms:
            continue
        try:
            out[f"{side}_contact"] = bool(env.check_contact(geoms))
        except Exception:  # noqa: BLE001
            continue
    return out


def privileged_state(obs: dict, env) -> dict:
    """The `priv_*` block L3 attaches to an observation.

    `obs` must be the FULL observation, before `agent_visible_obs` narrows it -- the
    poses are read out of what that filter would have dropped.

    Every section is independent and optional. Returns {} when nothing could be read,
    which is what an L1-shaped observation looks like, so a failure here degrades the
    harness rather than the run.
    """
    from .config import _AGENT_OBS_RE

    out: dict[str, Any] = {}
    if isinstance(obs, dict):
        state = _object_state(obs, _AGENT_OBS_RE)
        if state:
            out["priv_object_state"] = state
    if env is None:
        return out

    for key, fn in (("priv_target", _target), ("priv_objects", _objects),
                    ("priv_fixtures", _fixtures), ("priv_grasp", _grasp)):
        try:
            value = fn(env)
        except Exception:  # noqa: BLE001 - see the module docstring
            continue
        if value:
            out[key] = value
    return out


def agent_view(obs: dict, env, level: str | None = None) -> Any:
    """THE agent's view of an observation, at every level. One definition, so the two
    kinds of reply -- one that IS an observation and one that CONTAINS a descriptor --
    cannot disagree about what a level means.

    `agent_visible_obs` narrows; L3 then widens by merging the block above. Both halves
    live here because splitting them is what let `EvaluationRun.descriptor` ship the
    narrowing without the widening, leaving `trial_info` and evaluation `reset`
    L1-shaped in an L3 container.

    `level` defaults to the container's, read from the one place it is defined; the
    daemon passes its own captured value so the wire cannot change level mid-run.
    """
    from . import config as C

    shown = C.agent_visible_obs(obs)
    if (level or C.level()) != C.PRIVILEGED_LEVEL or not isinstance(shown, dict):
        return shown
    return {**shown, **privileged_state(obs, env)}
