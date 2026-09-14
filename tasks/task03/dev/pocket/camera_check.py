"""Static camera regression with deliberately delayed renderer collection."""

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from PIL import Image
from harness import env as camera
from harness.tabletop.pocket import scene
from harness.tabletop.pocket.scene import CAMERAS
from robosuite.utils.binding_utils import MjRenderContextOffscreen

# Pixel-footprint tolerances for anti-aliased, oblique planar surfaces.
DEPTH_TOLERANCE_M = {"workspace": .003, "cube_left": .001, "cube_right": .001}


def depth_alignment(env):
    """Compare camera-plane depth with independent rays through visible flat surfaces."""
    frames = camera.render_frames(env, CAMERAS, 512, 512, depth=True)
    model, data = env.sim.model._model, env.sim.data._data
    result = {}
    for name in CAMERAS:
        cid = model.camera(name).id
        focal = 256 / np.tan(np.deg2rad(model.cam_fovy[cid])/2)
        rotation = data.cam_xmat[cid].reshape(3, 3)
        errors = []
        for v in range(8, 504, 8):
            for u in range(8, 504, 8):
                ray = np.array([(u+.5-256)/focal, -(v+.5-256)/focal, -1])
                norm = np.linalg.norm(ray)
                gid = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(model, data, data.cam_xpos[cid], rotation@ray/norm,
                                        np.array([0, 1, 1, 1, 1, 1], dtype=np.uint8), True, -1, gid)
                if distance <= 0:
                    continue
                geom = model.geom(int(gid[0])).name
                if not ((name == "workspace" and geom == "table_visual") or
                        (name != "workspace" and geom.startswith("cube_sticker"))):
                    continue
                depth = frames[name+"_depth"]
                gradient_limit = .02 if name == "workspace" else .003
                if np.ptp(depth[v-1:v+2, u-1:u+2]) < gradient_limit:
                    errors.append(abs(distance/norm-depth[v, u]))
        result[name] = dict(samples=len(errors), max_error_m=max(errors, default=None))
    return result


def run(output, legacy=False, resets=5):
    camera._use_upright_images()
    output.mkdir(parents=True, exist_ok=True)
    original = scene.PocketRenderContext
    if legacy:
        class LegacyContext(MjRenderContextOffscreen):
            def close(self):
                pass
        scene.PocketRenderContext = LegacyContext
    env = scene.PocketCube(seed=139, renderer="mjviewer" if legacy else "mujoco")
    rows = []
    try:
        for reset in range(resets):
            retired = env.sim._render_context_offscreen
            obs = env.reset()
            state = (env.sim.data.qpos.copy(), env.sim.data.qvel.copy(), env.sim.data.time)
            del retired
            gc.collect()
            references = {}
            for width, height in ((512, 512), (384, 256), (511, 510), (640, 512), (512, 512)):
                for depth in (False, True, False, True):
                    frames = camera.render_frames(env, CAMERAS, width, height, depth=depth)
                    for name in CAMERAS:
                        rgb = frames[name+"_image"]
                        key = name, width, height
                        ref = references.setdefault(key, rgb.copy())
                        if (width, height) == (512, 512):
                            ref = obs[name+"_image"]
                        error = float(np.abs(rgb.astype(float)-ref).mean())
                        row = dict(reset=reset, camera=name, width=width, height=height,
                                   depth=depth, rgb_mae=error)
                        if depth:
                            z = frames[name+"_depth"]
                            row["depth_finite_positive"] = bool(np.isfinite(z).all() and (z > 0).all())
                            dkey = name, width, height, "depth"
                            dref = references.setdefault(dkey, z.copy())
                            row["depth_max_error_m"] = float(np.abs(z-dref).max())
                        rows.append(row)
                    assert np.array_equal(state[0], env.sim.data.qpos)
                    assert np.array_equal(state[1], env.sim.data.qvel)
                    assert state[2] == env.sim.data.time
                    if reset == 0 and (width, height) == (512, 512) and depth and len(rows) == 6:
                        for label, images in (("cached", obs), ("depth_rgb", frames)):
                            Image.fromarray(np.concatenate([images[c+"_image"] for c in CAMERAS], axis=1)).save(output / f"{label}.png")
        result = dict(legacy=legacy, comparisons=len(rows),
                      mismatches=sum(r["rgb_mae"] > 1e-4 for r in rows),
                      max_rgb_mae=max(r["rgb_mae"] for r in rows), rows=rows)
        if not legacy:
            result["depth_alignment"] = depth_alignment(env)
        (output / "camera_check.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
        return result
    finally:
        env.close()
        scene.PocketRenderContext = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    run(args.output, args.legacy)
