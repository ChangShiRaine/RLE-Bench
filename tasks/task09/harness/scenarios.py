"""Scene composition and deterministic workspace sampling.

compose_lead() strips actuators/equalities/tendons/keyframes from the
submitted model, re-pins the physics options and the stock-hardware joint
parameters, and disables all contacts, so what remains of the design is
exactly what the task grades: kinematic structure, masses/counterweights and
springs. Physical instances are drawn deterministically from (seed, envelope).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import mujoco
import numpy as np

from . import spec as gspec

HOME = np.asarray(gspec.LEAD_HOME)
HW = np.asarray(gspec.WORKSPACE_HALFWIDTH)


@dataclass
class GelloEnvelope:
    """Held-out instance envelope; concrete instances are drawn per seed."""
    link_mass_tol: float = 0.05        # printed-part mass tolerance, +-fraction
    n_hold_configs: int = 12           # hold grid size (3 fixed + seeded rest)
    n_residual_samples: int = 96       # seeded residual samples on top of the grid
    path_waypoints: int = 5            # teleop path
    path_duration_s: float = 24.0      # quasi-static: ~0.1 rad/s
    hold_time_s: float = 1.5


def compose_lead(lead_xml_path: str, seed: int = 0,
                 env: GelloEnvelope | None = None, perturb: bool = True):
    """Submitted model -> pinned, optionally mass-perturbed (model, data)."""
    env = env or GelloEnvelope()
    spec = mujoco.MjSpec.from_file(lead_xml_path)
    for lst in (spec.actuators, spec.equalities, spec.tendons, spec.keys,
                getattr(spec, "plugins", [])):
        for item in list(lst):
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
    model.body_gravcomp[:] = 0.0   # no engine-side gravity compensation

    if perturb:
        rng = np.random.default_rng([int(seed), 7])
        for b in range(1, model.nbody):
            f = 1.0 + rng.uniform(-env.link_mass_tol, env.link_mass_tol)
            model.body_mass[b] *= f
            model.body_inertia[b] *= f
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def workspace_grid(levels: int = 3) -> np.ndarray:
    """Seedless grid over the required workspace box. Joint 1 stays home: the
    other joints' gravity loads are invariant to q1."""
    axes = [np.array([HOME[0]])]
    axes += [HOME[i] + np.linspace(-HW[i], HW[i], levels) for i in range(1, gspec.N_JOINTS)]
    return np.array([np.concatenate([np.atleast_1d(a) for a in combo])
                     for combo in itertools.product(*axes)])


def sample_workspace(rng, n: int) -> np.ndarray:
    return HOME + rng.uniform(-1.0, 1.0, size=(n, gspec.N_JOINTS)) * HW


_ALT = np.where(np.arange(gspec.N_JOINTS) % 2 == 0, 1.0, -1.0)


def fixed_hold_configs() -> np.ndarray:
    """Seedless corner poses: home and the two alternating box corners."""
    return np.stack([HOME, HOME + HW * _ALT, HOME - HW * _ALT])


def hold_configs(seed: int, env: GelloEnvelope) -> np.ndarray:
    """Fixed corners + seeded fill."""
    fixed = fixed_hold_configs()
    rng = np.random.default_rng([int(seed), 1])
    n = max(env.n_hold_configs - fixed.shape[0], 0)
    return np.vstack([fixed, sample_workspace(rng, n)])


def teleop_path(seed: int, env: GelloEnvelope, dt: float = 0.1,
                waypoints: np.ndarray | None = None) -> np.ndarray:
    """Smooth quasi-static waypoint path through the workspace: cosine-eased
    segments, starting and ending at home."""
    if waypoints is None:
        rng = np.random.default_rng([int(seed), 2])
        waypoints = np.vstack([HOME,
                               sample_workspace(rng, env.path_waypoints - 2),
                               HOME])
    n_seg = waypoints.shape[0] - 1
    t_seg = env.path_duration_s / n_seg
    steps = max(int(round(t_seg / dt)), 2)
    path = [waypoints[0][None, :]]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        s = (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, steps + 1)[1:])) / 2.0
        path.append(a + s[:, None] * (b - a))
    return np.vstack(path)
