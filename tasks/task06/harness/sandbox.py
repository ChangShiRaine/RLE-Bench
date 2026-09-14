"""Isolated execution of untrusted task06 submissions.

Threat model: rendering is deterministic, so an estimator that could read the
evaluation seed list could regenerate every scene and return perfect ground
truth. Layers:

1. Agent code runs in a FRESH subprocess (``subprocess.Popen`` of a
   self-contained runner script) — no verifier memory is inherited (a fork
   would copy module state; spawn-style is mandatory here).
2. The runner never imports the harness; the child's ``sys.path`` gets only the
   submission directory. PYTHONPATH is scrubbed from its environment.
3. When the verifier runs as root (the Harbor container), the child drops to
   uid/gid 65534 (nobody) pre-exec; test.sh chmods /tests to 700 first, so
   ground truth, the scoring code, and the evaluation seeds are unreadable
   even by absolute path. Dev runs (non-root) keep layers 1-2 and 4-5.
4. Evaluation seeds live in a JSON file (never an importable module) and are
   63-bit values — scene brute-force without the file is not feasible.
5. The parent trusts nothing back: only pose arrays are read, validated for
   shape/finiteness, and graded against parent-side ground truth. A parent
   wall-clock batch timeout prevents unbounded execution.

Input masking also happens parent-side: an rgb-only frames file contains no
depth array, so the sandboxed code cannot receive what the parent never
serialized.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np

from . import spec

DEFAULT_FRAME_TIMEOUT_S = 10.0   # aggregate batch allowance per frame
STARTUP_TIMEOUT_S = 90.0         # interpreter + torch import headroom

_RUNNER = r'''
import importlib.util, json, os, sys, time
import numpy as np


def _pack_torch(rgb, depth):
    import torch
    rgbf = torch.from_numpy(np.ascontiguousarray(rgb)).float() / 255.0
    d = torch.from_numpy(np.ascontiguousarray(depth)).float()
    valid = torch.isfinite(d) & (d > 0)
    d = torch.where(valid, d, torch.zeros_like(d))
    return torch.cat([rgbf.permute(2, 0, 1), d[None], valid[None].float()], 0)


def main():
    work = sys.argv[1]
    with open(os.path.join(work, "job.json")) as f:
        job = json.load(f)
    data = np.load(os.path.join(work, "frames.npz"), allow_pickle=False)
    n = int(data["ts"].shape[0])
    obs_static = {"K": data["K"], "T_cam_table": data["T_cam_table"]}
    resets = data["resets"]

    if job["mode"] == "torch":
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(0)
        model = torch.jit.load(job["model_path"], map_location="cpu")
        parameter_count = sum(
            int(tensor.numel()) for tensor in model.parameters()
        )
        parameter_count += sum(
            int(tensor.numel()) for tensor in model.buffers()
        )
        for node in model.inlined_graph.nodes():
            if (node.kind() == "prim::Constant"
                    and "value" in node.attributeNames()
                    and node.kindOf("value") == "t"):
                parameter_count += int(node.t("value").numel())
        parameter_limit = int(job["model_max_parameters"])
        if parameter_count > parameter_limit:
            raise ValueError(
                f"model parameter limit exceeded: "
                f"{parameter_count} > {parameter_limit}"
            )
        model.eval()
        est = None
    else:
        sub_dir = job["submission_dir"]
        sys.path.insert(0, sub_dir)
        spec = importlib.util.spec_from_file_location(
            "estimator", os.path.join(sub_dir, "estimator.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        est = mod.make_estimator()
        model = None

    poses = np.full((n, 3), np.nan, dtype=np.float64)
    shape_ids = np.full(n, -1, dtype=np.int64)
    times = np.zeros(n, dtype=np.float64)
    for i in range(n):
        obs = dict(obs_static)
        obs["t"] = float(data["ts"][i])
        if "rgb" in data.files:
            obs["rgb"] = data["rgb"][i]
        if "depth" in data.files:
            obs["depth"] = data["depth"][i]
        t0 = time.perf_counter()
        try:
            if model is not None:
                import torch
                with torch.no_grad():
                    out = model(_pack_torch(obs.get("rgb"),
                                            obs.get("depth"))[None])[0]
                v = out.detach().reshape(-1).float()
                if v.numel() != 7 or not bool(torch.isfinite(v).all()):
                    raise ValueError("expected seven finite model outputs")
                shape_id = int(torch.argmax(v[4:7]))
                pose = (float(v[0]), float(v[1]),
                        float(np.arctan2(float(v[3]), float(v[2]))))
            else:
                if bool(resets[i]):
                    est.reset()
                output = est.update(**obs)
                if len(output) != 4:
                    raise ValueError("expected (x, y, theta, shape_id)")
                pose, shape_id = output[:3], output[3]
                if isinstance(shape_id, (bool, np.bool_)) or not isinstance(shape_id, (int, np.integer)):
                    raise ValueError("shape_id must be an integer")
            p = np.asarray(pose, dtype=float).reshape(-1)
            if p.shape == (3,):
                poses[i] = p
                if shape_id in (0, 1, 2):
                    shape_ids[i] = shape_id
        except Exception:
            import traceback
            traceback.print_exc()
        times[i] = time.perf_counter() - t0
    np.savez(os.path.join(work, "result.npz"), poses=poses, shape_ids=shape_ids, times=times)


main()
'''


@dataclass
class SandboxResult:
    poses: np.ndarray | None = None       # (N, 3) with NaN rows on failure
    shape_ids: np.ndarray | None = None   # (N,) fixed IDs, -1 for invalid
    child_times: np.ndarray | None = None  # child-reported, informational only
    wall_s: float = 0.0                   # parent-measured, authoritative
    n_frames: int = 0
    error: str | None = None
    stderr_tail: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None and self.poses is not None


def _drop_privileges():
    """Pre-exec: become nobody so /tests (chmod 700) is unreadable."""
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)


def _world_readable(path: str) -> None:
    """Expose a verifier-owned, link-free staging tree to uid 65534."""
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            candidate = os.path.join(root, name)
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("submission staging tree contains a link")
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError("submission staging tree contains a special file")
        for name in dirs:
            os.chmod(os.path.join(root, name), 0o755)
        for name in files:
            candidate = os.path.join(root, name)
            mode = os.lstat(candidate).st_mode
            os.chmod(candidate, 0o555 if mode & 0o111 else 0o444)

    os.chmod(path, 0o755)


def write_frames(work: str, frames: list[dict], channels: tuple,
                 resets: list[bool]) -> None:
    """Serialize observations for the child. Channels not in ``channels``
    are never written — parent-side input masking."""
    arrays = {
        "ts": np.array([f["t"] for f in frames], dtype=np.float64),
        "K": np.asarray(frames[0]["K"], dtype=np.float64),
        "T_cam_table": np.asarray(frames[0]["T_cam_table"], dtype=np.float64),
        "resets": np.array(resets, dtype=bool),
    }
    if "rgb" in channels:
        arrays["rgb"] = np.stack([f["rgb"] for f in frames]).astype(np.uint8)
    if "depth" in channels:
        arrays["depth"] = np.stack([f["depth"] for f in frames]) \
            .astype(np.float32)
    np.savez(os.path.join(work, "frames.npz"), **arrays)


def run_job(mode: str, frames: list[dict], channels: tuple,
            resets: list[bool] | None = None,
            submission_dir: str | None = None,
            model_path: str | None = None,
            frame_timeout_s: float = DEFAULT_FRAME_TIMEOUT_S,
            startup_timeout_s: float = STARTUP_TIMEOUT_S,
            prepare_submission: bool = True) -> SandboxResult:
    """Execute one batch of frames through the submission.

    mode "code":  imports estimator.py from submission_dir (rgb-only/rgb-depth/method-agnostic)
    mode "torch": loads model_path via torch.jit, fixed inference (rgb-depth-model-training)
    """
    n = len(frames)
    resets = resets if resets is not None else [True] * n
    res = SandboxResult(n_frames=n)
    work = tempfile.mkdtemp(prefix="task06_")
    try:
        os.chmod(work, 0o777)
        write_frames(work, frames, channels, resets)
        job = {"mode": mode}
        if mode == "torch":
            if model_path is None or not os.path.exists(model_path):
                res.error = "model.pt missing"
                return res
            staged_model = os.path.join(work, "model.pt")
            shutil.copy(model_path, staged_model)
            os.chmod(staged_model, 0o644)
            job["model_path"] = staged_model
            job["model_max_parameters"] = spec.MODEL_MAX_PARAMETERS
        else:
            if submission_dir is None or not os.path.exists(
                    os.path.join(submission_dir, "estimator.py")):
                res.error = "estimator.py missing"
                return res
            if prepare_submission:
                _world_readable(submission_dir)
            job["submission_dir"] = os.path.abspath(submission_dir)
        with open(os.path.join(work, "job.json"), "w") as f:
            json.dump(job, f)
        runner = os.path.join(work, "runner.py")
        with open(runner, "w") as f:
            f.write(_RUNNER)
        for p in ("job.json", "frames.npz", "runner.py"):
            os.chmod(os.path.join(work, p), 0o644)

        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": work, "TMPDIR": work, "MPLCONFIGDIR": work,
            "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        }
        preexec = _drop_privileges if os.geteuid() == 0 else None
        timeout = startup_timeout_s + frame_timeout_s * n
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, runner, work], env=env, cwd=work,
                capture_output=True, text=True, timeout=timeout,
                preexec_fn=preexec)
        except subprocess.TimeoutExpired:
            res.wall_s = time.perf_counter() - t0
            res.error = f"sandbox timeout after {timeout:.0f}s"
            return res
        res.wall_s = time.perf_counter() - t0
        res.stderr_tail = (proc.stderr or "")[-2000:]
        result_path = os.path.join(work, "result.npz")
        if proc.returncode != 0 or not os.path.exists(result_path):
            detail = res.stderr_tail.strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            res.error = f"sandbox exited {proc.returncode}{suffix}"
            return res
        out = np.load(result_path, allow_pickle=False)
        poses = np.asarray(out["poses"], dtype=np.float64)
        if poses.shape != (n, 3):
            res.error = f"bad result shape {poses.shape}"
            return res
        res.poses = poses
        ids = np.asarray(out["shape_ids"], dtype=np.float64)
        if ids.shape != (n,):
            res.error = f"bad shape-ID array shape {ids.shape}"
            return res
        valid_ids = np.isfinite(ids) & np.isin(ids, (0, 1, 2))
        res.shape_ids = np.where(valid_ids, ids, -1).astype(np.int64)
        res.child_times = np.asarray(out["times"], dtype=np.float64)
        return res
    except Exception as e:  # sandbox infrastructure fault, not agent fault
        res.error = f"sandbox error: {e}"
        return res
    finally:
        shutil.rmtree(work, ignore_errors=True)
