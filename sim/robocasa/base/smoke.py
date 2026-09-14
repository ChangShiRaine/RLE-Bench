"""Build-time smoke test for the shared RoboCasa simulator image.

Fails the Docker build unless:
  1. robocasa's env-construction path works with the REDUCED dependency set
     (no torch / lerobot / tianshou / wandb / tensorboard), and
  2. offscreen rendering works under the pinned MUJOCO_GL backend, and
  3. the pinned learning stack really is absent (so nobody silently re-adds
     4 GB by installing robocasa with its full install_requires).

Also prints the action/observation contract and per-step timing, which is what
sizes the interaction budgets of every task built on this image.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ENV_NAME = os.environ.get("SMOKE_ENV", "CloseDrawer")
N_STEP = int(os.environ.get("SMOKE_STEPS", "20"))

# --deps-only is what the Dockerfile runs at BUILD time. It checks everything that
# does not need the asset dataset or a GPU -- neither of which exists during
# `docker build`, by design (assets are a separate mounted dataset). The full
# env-construction and render checks run at RUN time -- see sim/robocasa/README.md.
DEPS_ONLY = "--deps-only" in sys.argv


def main() -> int:
    backend = os.environ.get("MUJOCO_GL", "<unset>")
    print(f"[smoke] MUJOCO_GL={backend} deps_only={DEPS_ONLY}")

    # (3) The learning stack must NOT be installed.
    leaked = []
    for mod in ("torch", "torchvision", "torchcodec", "lerobot", "tianshou", "wandb"):
        try:
            __import__(mod)
        except ImportError:
            pass
        else:
            leaked.append(mod)
    if leaked:
        print(f"[smoke] FAIL: learning stack present in image: {leaked}")
        return 1
    print("[smoke] reduced dependency set confirmed (no torch/lerobot/tianshou/wandb)")

    # numba JIT must stay off under OSMesa: libOSMesa links libLLVM.so.19.1 and
    # llvmlite ships its own LLVM, and two LLVM copies segfault on first JIT
    # compile. Fail loudly here rather than as a mystery SIGSEGV later.
    if backend == "osmesa" and os.environ.get("NUMBA_DISABLE_JIT") != "1":
        print("[smoke] FAIL: MUJOCO_GL=osmesa requires NUMBA_DISABLE_JIT=1 "
              "(libOSMesa/llvmlite LLVM clash segfaults numba's JIT)")
        return 1

    # MUJOCO_GL=egl asking for a GPU is not the same as GETTING one. libglvnd needs a
    # vendor ICD to find the NVIDIA driver; with only Mesa's 50_mesa.json it silently
    # renders on llvmpipe -- 315 ms/step against 28 ms/step, and DIFFERENT PIXELS
    # (std 72.75 vs 72.85), so a vision agent is scored against the wrong frames with
    # nothing raising. Checked here because the fallback is silent by construction.
    #
    # The .so half is deliberately not checked: at BUILD time the toolkit has injected
    # nothing yet, so only the file the image itself must carry is asserted.
    if backend == "egl":
        icd = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        if not os.path.isfile(icd):
            print(f"[smoke] FAIL: MUJOCO_GL=egl but {icd} is missing, so libglvnd "
                  "falls back to Mesa software rendering instead of the GPU. The "
                  "image must carry this file -- nvidia-container-toolkit injects "
                  "libEGL_nvidia.so but never a vendor .json.")
            return 1
        print(f"[smoke] NVIDIA EGL vendor ICD present ({icd})")

    # Import the env-construction path itself -- this is the check that the
    # --no-deps robocasa install plus the hand-picked runtime deps is actually
    # sufficient, and it needs no assets and no GPU.
    from robocasa.utils.env_utils import create_env

    print("[smoke] robocasa env-construction path imports cleanly")

    if DEPS_ONLY:
        print("[smoke] OK (deps-only; assets and EGL are verified at run time)")
        sys.stdout.flush()
        os._exit(0)

    # Everything below needs the mounted asset dataset and a working GL backend.
    # The asset mount is read-only, and RoboCasa writes a transient MJCF into the
    # asset tree per object, so this patch is mandatory -- without it env
    # construction dies with OSError [Errno 30] Read-only file system.
    import rlebench_ro_assets

    print(f"[smoke] ro-assets patch installed, scratch={rlebench_ro_assets.install()}")

    # UPRIGHT FRAMES. MuJoCo reads pixels bottom-up and robosuite's default convention
    # ("opengl") passes that straight through, so out of the box every camera
    # observation is upside down -- which went unnoticed in production because the
    # frames saved for humans were flipped on the way out while the agent's were not.
    #
    # MIRRORS tasks/task01/harness/env.py:_use_upright_images, which is the source of
    # truth. It is repeated rather than imported because this image is deliberately
    # simulator-only and has no harness code in it (see any task's build_assets.py).
    import robosuite.macros as macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

    macros.IMAGE_CONVENTION = "opencv"
    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    print(f"[smoke] image convention={macros.IMAGE_CONVENTION} (step {convention})")

    t0 = time.perf_counter()
    env = create_env(
        env_name=ENV_NAME,
        split="pretrain",
        seed=0,
        camera_widths=128,
        camera_heights=128,
    )
    construct = time.perf_counter() - t0

    t0 = time.perf_counter()
    obs = env.reset()
    reset_sec = time.perf_counter() - t0

    # NOTE: action_spec is only valid AFTER the first reset -- before it,
    # env.robots[0] is None and action_spec raises
    # AttributeError: 'NoneType' object has no attribute 'action_limits'.
    low, high = env.action_spec
    action_dim = int(np.asarray(low).shape[0])

    img_keys = sorted(k for k in obs if k.endswith("_image"))
    if not img_keys:
        print("[smoke] FAIL: no image observations -- offscreen rendering is broken")
        return 1
    for k in img_keys:
        arr = np.asarray(obs[k])
        if arr.ndim != 3 or arr.shape[2] != 3:
            print(f"[smoke] FAIL: {k} has unexpected shape {arr.shape}")
            return 1
    # A camera that renders a uniform frame usually means a dead GL context.
    spreads = {k: float(np.asarray(obs[k]).std()) for k in img_keys}
    if max(spreads.values()) < 1.0:
        print(f"[smoke] FAIL: all camera frames are flat under {backend}: {spreads}")
        return 1

    # ORIENTATION, the one thing a unit test cannot check: that an on-demand render and
    # the streamed observation agree in the real simulator. They disagreed in
    # production -- the agent's frames were inverted relative to every saved artifact --
    # and nothing raised, because an upside-down image is still a valid image.
    cam = img_keys[0][: -len("_image")]
    direct = np.asarray(env.sim.render(camera_name=cam, width=128, height=128,
                                       depth=False))[::convention]
    if not np.array_equal(direct, np.asarray(obs[img_keys[0]])):
        print(f"[smoke] FAIL: {cam} renders differently on demand than in the "
              "observation -- the two paths disagree about orientation")
        return 1
    print(f"[smoke] on-demand render matches the observation for {cam}")

    # And write one frame out so the orientation can actually be LOOKED at. Consistency
    # only proves the two paths agree; it cannot prove they are both the right way up.
    try:
        from PIL import Image

        out_png = os.environ.get("SMOKE_FRAME", "/tmp/smoke_upright.png")
        Image.fromarray(np.asarray(obs[img_keys[0]]).astype("uint8")).save(out_png)
        print(f"[smoke] wrote {out_png} -- open it; the kitchen must be the right "
              "way up")
    except ImportError:
        print("[smoke] PIL absent; skipped writing a frame to eyeball")

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(N_STEP):
        env.step(rng.uniform(low=low, high=high))
    step_sec = (time.perf_counter() - t0) / N_STEP

    try:
        lang = env.get_ep_meta().get("lang")
    except Exception:  # noqa: BLE001
        lang = None

    print(f"[smoke] env={ENV_NAME} action_dim={action_dim} lang={lang!r}")
    print(f"[smoke] image obs: {img_keys} (std {spreads})")
    print(
        f"[smoke] construct={construct:.1f}s reset={reset_sec:.2f}s "
        f"step={step_sec*1000:.1f}ms"
    )
    print("[smoke] OK")
    # Skip interpreter teardown: some GL backends raise in their free() path
    # after everything has already succeeded, which would fail the build.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
