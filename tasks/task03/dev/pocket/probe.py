"""Host-only pose-feedback probe, not a vision policy; no rollout cube edits."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from PIL import Image

from harness import env as shared_env
from harness.tabletop.pocket.scene import CAMERAS
from harness.tabletop.pocket.scene import PocketCube
from harness.tabletop.pocket import metrics
from rlebench.core.media import VideoWriter


def run(output, render=True, seed=17, scrambled=False, direction=1, one_move=False,
        grip_profile="baseline", hold=False, offset=0):
    shared_env._use_upright_images()
    variant = "pocket"
    env = PocketCube(seed=seed, use_camera_obs=False, has_offscreen_renderer=render,
                     grip_profile=grip_profile)
    env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    # Solved reset isolates mechanics. There are no cube writes after this setup.
    for body in (() if scrambled else env._cubie_ids):
        joint = model.body_jntadr[body]
        adr = model.jnt_qposadr[joint]
        if model.jnt_type[joint] == 1:
            data.qpos[adr:adr+4] = [1, 0, 0, 0]
        else:
            data.qpos[adr] = 0
    if one_move:
        poses = metrics.apply_moves(np.tile(np.eye(3), (8, 1, 1)), [(0, -1, -direction)])
        for body, pose in zip(env._cubie_ids, poses):
            adr = model.jnt_qposadr[model.body_jntadr[body]]
            data.qpos[adr:adr+4] = Rotation.from_matrix(pose).as_quat(scalar_first=True)
    env.sim.forward()
    core_id = model.body("cube_core").id
    initial = data.xmat[core_id].reshape(3, 3).T @ data.xmat[env._cubie_ids].reshape(8, 3, 3)
    expected = metrics.apply_moves(initial, [(0, -1, direction)])
    expected = expected[0].T @ expected
    output.mkdir(parents=True, exist_ok=True)
    video = VideoWriter(output / f"{variant}.mp4", fps=10) if render else None
    targets = np.array([[-.10, 0, 1.04], [.10, 0, 1.04]])
    rotations = [np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
                 np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])]
    if direction < 0:
        rotations[0] = Rotation.from_rotvec([-np.pi/2, 0, 0]).as_matrix() @ rotations[0]
    grip_frame = rotations[0].copy()
    records, count = [], 0
    hold_reference = None

    def move(grips, steps):
        nonlocal count
        for _ in range(steps):
            action = np.zeros(14)
            for i, robot in enumerate(env.robots):
                site = robot.eef_site_id["right"]
                base = np.diag([1, 1, 1] if i == 0 else [-1, -1, 1])
                delta = Rotation.from_matrix(rotations[i] @ data.site_xmat[site].reshape(3, 3).T).as_rotvec()
                action[7*i:7*i+3] = np.clip(base.T @ (targets[i]-data.site_xpos[site]) / .05, -.6, .6)
                action[7*i+3:7*i+6] = np.clip(base.T @ delta / .5, -.6, .6)
                action[7*i+6] = grips[i]
            env.step(action)
            count += 1
            if video and count % 2 == 0:
                raw = shared_env.render_frames(env, CAMERAS, 512, 512)
                video.add(np.concatenate([raw[f"{c}_image"] for c in CAMERAS], axis=1))

    def report(label):
        core = model.body("cube_core").id
        actual = data.xmat[core].reshape(3, 3).T @ data.xmat[env._cubie_ids].reshape(8, 3, 3)
        actual = actual[0].T @ actual
        error = np.arccos(np.clip((np.einsum("bij,bij->b", actual, expected)-1)/2, -1, 1))
        record = {"stage": label, "steps": count, "core_position": data.xpos[core].tolist(),
                  "expected_turn_error_deg": float(np.rad2deg(error.max())),
                  "cube": env.cube_state(), "success": env._check_success(),
                  "warnings": [int(w.number) for w in data.warning],
                  "eef": [data.site_xpos[r.eef_site_id["right"]].tolist() for r in env.robots],
                  "finger_positions": [data.geom_xpos[g].tolist() for g in range(model.ngeom)
                                       if "pad_collision" in model.geom(g).name],
                  "variant": "pocket"}
        pad_contacts = []
        for j, contact in enumerate(data.contact):
            names = [model.geom(int(g)).name for g in contact.geom]
            if any("pad_collision" in name for name in names) and any("cube_pocket" in name for name in names):
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, j, force)
                pad_contacts.append(dict(geoms=names, dim=int(contact.dim), force=force.tolist()))
        record["pad_contacts"] = pad_contacts
        record["grip_profile"] = grip_profile
        if hold_reference is not None:
            site = env.robots[1].eef_site_id["right"]
            relative = data.site_xmat[site].reshape(3, 3).T @ (data.xpos[core]-data.site_xpos[site])
            record["hold_slip_mm"] = float(1000*np.linalg.norm(relative-hold_reference))
        if render:
            raw = shared_env.render_frames(env, CAMERAS, 512, 512)
            Image.fromarray(np.concatenate([raw[f"{c}_image"] for c in CAMERAS], axis=1)).save(output / f"{variant}_{label}.png")
        records.append(record)
        print(json.dumps(record), flush=True)

    try:
        move([-1, -1], 60)
        targets[1, 0] = .016 - .0036 + offset
        move([-1, -1], 80)
        move([-1, 1], 100)
        report("right_grasp")
        if data.xpos[core_id, 2] < .9:
            return records
        if hold:
            for _ in range(80):
                targets[1, 2] += .0005
                move([-1, 1], 1)
            move([-1, 1], 40)
            site = env.robots[1].eef_site_id["right"]
            hold_reference = data.site_xmat[site].reshape(3, 3).T @ (data.xpos[core_id]-data.site_xpos[site])
            report("single_lift")
            for _ in range(10):
                move([-1, 1], 20)
                report("single_hold")
            return records
        targets[0, 0] = -.016 + .0036
        move([-1, 1], 80)
        move([1, 1], 100)
        report("dual_grasp")
        for _ in range(80):
            targets[:, 2] += .0005
            move([1, 1], 1)
        move([1, 1], 40)
        report("lift")
        # Privileged pose feedback isolates the model from perception/controller error.
        core = model.body("cube_core").id
        frame = data.xmat[env._cubie_ids[0]].reshape(3, 3) @ initial[0].T
        targets[0] = data.xpos[core] + frame @ np.array([-.0604, 0, 0])
        targets[1] = data.site_xpos[env.robots[1].eef_site_id["right"]].copy()
        rotations[1] = data.site_xmat[env.robots[1].eef_site_id["right"]].reshape(3, 3).copy()
        start = frame @ grip_frame
        rotations[0] = start
        move([1, 1], 60)
        for angle in np.linspace(0, direction*np.pi/2, 250):
            rotations[0] = Rotation.from_rotvec(-angle * frame[:, 0]).as_matrix() @ start
            move([1, 1], 1)
        move([1, 1], 40)
        report("turn")
        move([-1, 1], 40)
        targets[0] -= .08*frame[:, 0]
        move([-1, 1], 80)
        report("release")
    finally:
        if video:
            video.close()
        env.close()
        (output / f"{variant}.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/task03-pocket-physical"))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--scrambled", action="store_true")
    parser.add_argument("--one-move", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--grip-profile", choices=("baseline", "torsion", "elliptic", "wide"), default="baseline")
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--offset", type=float, default=0)
    args = parser.parse_args()
    run(args.output, not args.no_render, args.seed, args.scrambled, args.direction, args.one_move,
        args.grip_profile, args.hold, args.offset)
