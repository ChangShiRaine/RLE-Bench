"""Post-eval renders of the submitted robot (verifier-only, informational).

Written through rlebench.core.media after reward.json is on disk, so a
render can never affect scoring or determinism; the images exist so a human
reading /logs/verifier can SEE what the agent built (free-floating wheels,
battery mast, buried arm, ...) without loading the model themselves.

Headless via OSMesa (the verifier image ships libosmesa6, no GPU). Views:
  robot_stow.png         scenario-A start pose, 3/4 view
  robot_extended.png     arm at full canonical extension (reach + footprint)
  robot_integration.png  the shelf scene, start pose
"""
from __future__ import annotations

import os


def render_submission(media, robot_xml: str, arm_reference_xml: str) -> None:
    """Render the submitted robot with the canonical Panda into `media` (a
    rlebench.core.media.Media).

    Raises on a scene or GL failure; the caller's Media.run records it."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco

    from ..sim.arm_variants import trusted_arm_specs
    from ..sim.scenarios import Envelope, compose_scene
    from .pickscene import compose_pick_scene, shelf_placement

    arm = trusted_arm_specs(arm_reference_xml)["panda"]
    model, data, _ = compose_scene(robot_xml, "A", 0, Envelope(), arm)
    mujoco.mj_forward(model, data)
    base = data.body("base").xpos
    media.still("robot_stow.png", model, data,
                lookat=[base[0], base[1], 0.35], distance=2.0, azimuth=140,
          elevation=-25)

    for i, q in enumerate(arm.extended(0.0)):
        data.qpos[model.joint(arm.joint(i)).qposadr[0]] = q
    mujoco.mj_forward(model, data)
    media.still("robot_extended.png", model, data,
                lookat=[base[0] + 0.25, base[1], 0.4], distance=2.4, azimuth=155,
          elevation=-20)

    _front_x, face_x = shelf_placement(robot_xml, arm_reference_xml, arm)
    pmodel, pdata, _names = compose_pick_scene(robot_xml, face_x, arm=arm)
    mujoco.mj_forward(pmodel, pdata)
    # camera on the robot's side so the pod's open bins face the viewer
    media.still("robot_integration.png", pmodel, pdata,
                lookat=[face_x / 2 + 0.1, 0.0, 0.6], distance=3.0, azimuth=-35,
          elevation=-12)
