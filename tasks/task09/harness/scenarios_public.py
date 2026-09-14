"""Nominal model and analysis helpers for the GELLO co-design task."""
from __future__ import annotations

import itertools

import mujoco
import numpy as np

from . import spec as gspec


HOME = np.asarray(gspec.LEAD_HOME)
HW = np.asarray(gspec.WORKSPACE_HALFWIDTH)


def compose_lead(lead_xml_path: str):
    """Load a nominal design with the supplied stock hardware settings.

    Actuators and other simulation-only assistance declared in the submitted
    XML are ignored. This helper does not create physical-device variations.
    """
    spec = mujoco.MjSpec.from_file(lead_xml_path)
    for collection in (spec.actuators, spec.equalities, spec.tendons, spec.keys,
                       getattr(spec, "plugins", [])):
        for item in list(collection):
            spec.delete(item)
    spec.option.timestep = gspec.TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.gravity = gspec.GRAVITY
    spec.option.disableflags = 0
    spec.option.wind = [0.0, 0.0, 0.0]
    spec.option.density = 0.0
    spec.option.viscosity = 0.0
    model = spec.compile()

    model.dof_damping[:] = gspec.JOINT_DAMPING
    model.dof_frictionloss[:] = gspec.JOINT_FRICTIONLOSS
    model.dof_armature[:] = gspec.JOINT_ARMATURE
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    model.body_gravcomp[:] = 0.0

    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def workspace_grid(levels: int = 3) -> np.ndarray:
    """Nominal workspace sampling grid."""
    axes = [np.array([HOME[0]])]
    axes += [HOME[index] + np.linspace(-HW[index], HW[index], levels)
             for index in range(1, gspec.N_JOINTS)]
    return np.array([
        np.concatenate([np.atleast_1d(value) for value in combination])
        for combination in itertools.product(*axes)
    ])


def canonical_path(dt: float = 0.1) -> np.ndarray:
    """Nominal smooth path through the workspace."""
    alternating = np.where(np.arange(gspec.N_JOINTS) % 2 == 0, 1.0, -1.0)
    waypoints = np.vstack([
        HOME,
        HOME + HW * alternating,
        HOME - HW * alternating,
        HOME,
    ])
    duration_s = 24.0
    segment_s = duration_s / (waypoints.shape[0] - 1)
    steps = max(int(round(segment_s / dt)), 2)
    path = [waypoints[0][None, :]]
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        phase = np.linspace(0.0, 1.0, steps + 1)[1:]
        eased = (1.0 - np.cos(np.pi * phase)) / 2.0
        path.append(start + eased[:, None] * (end - start))
    return np.vstack(path)
