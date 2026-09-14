"""Training utilities for rgb-depth-model-training subtask (agent-visible).

Provides seeded dataset generation from the public design seeds and an
epoch-controlled training loop. The Harbor agent phase supplies the overall
time limit; `fit()` does not impose a second wall-clock deadline.

Data generation uses the root-owned rendering service, which is started on
the same pinned OSMesa backend used for evaluation. Changing MUJOCO_GL in the
agent shell does not restart or reconfigure that service.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import episodes, spec, torch_contract


def generate_dataset(out_dir: str, seeds=None, frames_per_seed: int = 40,
                     episode_frames: int = 60, downsample: int = 1) -> dict:
    """Render a labeled dataset into ``out_dir`` as .npz shards.

    Per seed: ``frames_per_seed`` settled single frames with re-randomized
    block poses, plus one push episode truncated to ``episode_frames``
    (includes occluded frames — that's where amodal supervision comes from).
    Deterministic in (seeds, counts).
    """
    seeds = list(seeds if seeds is not None else spec.DESIGN_SEEDS)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"shards": [], "downsample": downsample}
    for seed in seeds:
        rgbs, depths, poses = [], [], []
        rng = np.random.default_rng([seed, 9000])
        for k in range(frames_per_seed):
            sub = int(rng.integers(0, 2**31 - 1))
            f = episodes.single_frame(sub)
            rgbs.append(f.obs["rgb"][::downsample, ::downsample])
            depths.append(f.obs["depth"][::downsample, ::downsample])
            poses.append(f.gt)
        ep = episodes.push_episode(seed, render=True,
                                   max_frames=episode_frames)
        for fr in ep.frames:
            rgbs.append(fr.obs["rgb"][::downsample, ::downsample])
            depths.append(fr.obs["depth"][::downsample, ::downsample])
            poses.append(fr.gt)
        shard = out / f"shard_{seed}.npz"
        np.savez_compressed(
            shard, rgb=np.stack(rgbs).astype(np.uint8),
            depth=np.stack(depths).astype(np.float16),
            pose=np.asarray(poses, dtype=np.float32))
        meta["shards"].append(shard.name)
    (out / "meta.json").write_text(json.dumps(meta))
    return meta


class ShardDataset(torch.utils.data.Dataset):
    """Loads the generated shards fully into RAM (uint8/f16 keeps it small)."""

    def __init__(self, data_dir: str):
        d = Path(data_dir)
        meta = json.loads((d / "meta.json").read_text())
        rgb, depth, pose = [], [], []
        for name in meta["shards"]:
            z = np.load(d / name)
            rgb.append(z["rgb"])
            depth.append(z["depth"])
            pose.append(z["pose"])
        self.rgb = np.concatenate(rgb)
        self.depth = np.concatenate(depth)
        self.pose = np.concatenate(pose)
        self.shape_labels = None

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, i):
        x = torch_contract.pack_observation(
            self.rgb[i], self.depth[i].astype(np.float32))
        y = torch_contract.encode_pose(*self.pose[i])
        if self.shape_labels is not None:
            y = torch.cat((y, torch.tensor([self.shape_labels[i]], dtype=y.dtype)))
        return x, y


def pose_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Huber on xy (meters) + cosine-pair loss on the angle channels."""
    xy = torch.nn.functional.huber_loss(pred[:, :2], target[:, :2],
                                        delta=0.02)
    cs = pred[:, 2:4]
    norm = torch.linalg.norm(cs, dim=1, keepdim=True).clamp_min(1e-6)
    ang = torch.nn.functional.mse_loss(cs / norm, target[:, 2:4])
    reg = ((torch.linalg.norm(cs, dim=1) - 1.0) ** 2).mean()
    return xy * 5.0 + ang + 0.1 * reg


def fit(model: torch.nn.Module, data_dir: str, epochs: int = 50,
        batch_size: int = 16, lr: float = 3e-4, seed: int = 0,
        device: str | None = None, report_path: str | None = None,
        shape_labeler=None) -> dict:
    """Train for a caller-selected number of epochs.

    There is no per-call wall-clock budget. The enclosing Harbor agent phase
    supplies the task-wide time limit. Returns and optionally writes a
    reproducibility report. Optional ``shape_labeler(rgb, depth)`` supplies
    caller-inferred IDs (0=T, 1=C, 2=F; -1 skips classification supervision).
    Without it, only pose is trained; the renderer supplies no shape labels.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    t0 = time.time()
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = ShardDataset(data_dir)
    if shape_labeler is not None:
        labels = np.asarray([shape_labeler(rgb, depth.astype(np.float32))
                             for rgb, depth in zip(ds.rgb, ds.depth)])
        if labels.shape != (len(ds),) or not np.isin(labels, (-1, 0, 1, 2)).all():
            raise ValueError("shape_labeler must return IDs in {-1, 0, 1, 2}")
        ds.shape_labels = labels.astype(np.int64)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0)
    parameter_count = torch_contract.validate_model_size(model)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    batches = 0
    last_losses = []
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = pose_loss(pred, y)
            if ds.shape_labels is not None:
                if pred.ndim != 2 or pred.shape[1] != 7:
                    raise ValueError("shape training requires seven model outputs")
                valid = y[:, 4] >= 0
                if valid.any():
                    loss = loss + torch.nn.functional.cross_entropy(
                        pred[valid, 4:7], y[valid, 4].long())
            loss.backward()
            opt.step()
            batches += 1
            last_losses.append(float(loss.detach()))
            if len(last_losses) > 50:
                last_losses.pop(0)
    report = {
        "train_wall_clock_s": round(time.time() - t0, 2),
        "device": device,
        "epochs": epochs,
        "batches": batches,
        "dataset_size": len(ds),
        "shape_labeled_frames": int(np.sum(ds.shape_labels >= 0)) if ds.shape_labels is not None else 0,
        "parameter_count": parameter_count,
        "parameter_limit": spec.MODEL_MAX_PARAMETERS,
        "final_loss_mean50": float(np.mean(last_losses)) if last_losses else None,
        "seed": seed,
    }
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2))
    model.eval()
    return report
