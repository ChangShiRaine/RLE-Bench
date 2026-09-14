"""Post-eval renders of a submitted GELLO lead device (verifier-only).

Written through rlebench.core.media after reward.json is on disk, so a
render can never affect scoring. Two views per variant: the device posed at
the variant's home configuration, and where it settles when the capped
servo holds that pose with no software trim -- the hardware-only hold the
co-design checkpoints measure, visible without loading the model.
"""
from __future__ import annotations

import mujoco
import numpy as np

from rlebench.core.media import frame_extent

from .balance import hold_sim
from .scenarios import compose_lead
from .variants import get_variant


def render_submission(media, lead_xml: str, variant: str) -> None:
    """Render `lead_xml` into `media` (a rlebench.core.media.Media).

    Raises on a model or GL failure; the caller's Media.run records it."""
    spec = get_variant(variant)
    model, data = compose_lead(lead_xml, perturb=False)
    for name, q in zip(spec.joint_names, spec.home):
        data.qpos[model.joint(name).qposadr[0]] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    lookat, distance = frame_extent(data)    # one camera for both views
    media.still(f"{variant}_home.png", model, data, lookat, distance,
                azimuth=130.0, elevation=-25.0)

    hold_sim(model, data, spec.home)    # servo on, trim off; data ends settled
    if not np.isfinite(data.qpos).all():
        raise ValueError("device diverged while holding home")
    media.still(f"{variant}_hold.png", model, data, lookat, distance,
                azimuth=130.0, elevation=-25.0)
