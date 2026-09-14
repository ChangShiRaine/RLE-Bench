"""The tracking MDP: reward terms, terminations, and adaptive sampling.

Ported from BeyondMimic. Every term is a pure function over batched tensors, so a
submission can reweight, replace or drop any of them without touching the
environment — reward shaping is explicitly the agent's to own.

The evaluator shares none of this. It scores with its own instrumentation
(metrics.py), so nothing here can move what a submission is measured against.
"""
from __future__ import annotations

import torch

from . import rotations as R


DEFAULT_WEIGHTS = {
    "anchor_pos": 0.5,
    "anchor_ori": 0.5,
    "body_pos": 1.0,
    "body_ori": 1.0,
    "body_lin_vel": 1.0,
    "body_ang_vel": 1.0,
    "action_rate": -0.1,
    "joint_limits": -10.0,
    "undesired_contacts": -0.1,
}
DEFAULT_STD = {
    "anchor_pos": 0.3,
    "anchor_ori": 0.4,
    "body_pos": 0.3,
    "body_ori": 0.4,
    "body_lin_vel": 1.0,
    "body_ang_vel": 3.14,
}


def _exp(sq_error: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-sq_error / std**2)


def anchor_position_reward(ref_pos, pos, std):
    return _exp((ref_pos - pos).square().sum(-1), std)


def anchor_orientation_reward(ref_quat, quat, std):
    return _exp(R.quat_error(ref_quat, quat).square(), std)


def body_position_reward(ref_pos, pos, std):
    return _exp((ref_pos - pos).square().sum(-1).mean(-1), std)


def body_orientation_reward(ref_quat, quat, std):
    return _exp(R.quat_error(ref_quat, quat).square().mean(-1), std)


def body_linear_velocity_reward(ref_vel, vel, std):
    return _exp((ref_vel - vel).square().sum(-1).mean(-1), std)


def body_angular_velocity_reward(ref_vel, vel, std):
    return _exp((ref_vel - vel).square().sum(-1).mean(-1), std)


def action_rate_l2(action, prev_action):
    return (action - prev_action).square().sum(-1)


def joint_pos_limits(joint_pos, lower, upper):
    return ((lower - joint_pos).clamp(min=0.0)
            + (joint_pos - upper).clamp(min=0.0)).sum(-1)


def undesired_contacts(contact_force, threshold):
    return (contact_force.norm(dim=-1) > threshold).float().sum(-1)


def terminated(ref_body_pos, body_pos, ref_anchor_quat, anchor_quat,
               anchor_idx, ee_idx, anchor_z, anchor_tilt, ee_z):
    fell = (ref_body_pos[:, anchor_idx, 2] - body_pos[:, anchor_idx, 2]).abs() > anchor_z
    tipped = (_gravity_tilt(ref_anchor_quat) - _gravity_tilt(anchor_quat)).abs() > anchor_tilt
    limb = ((ref_body_pos[:, ee_idx, 2] - body_pos[:, ee_idx, 2]).abs() > ee_z).any(-1)
    return fell | tipped | limb


def _gravity_tilt(quat):
    gravity = torch.zeros(quat.shape[:-1] + (3,), device=quat.device, dtype=quat.dtype)
    gravity[..., 2] = -1.0
    return R.quat_rotate_inverse(quat, gravity)[..., 2]


class AdaptiveSampler:

    def __init__(self, n_frames, fps, device, kernel_size=1, lam=0.8,
                 uniform_ratio=0.1, alpha=0.001):
        self.n_frames = n_frames
        self.n_bins = int(n_frames // fps) + 1
        self.uniform_ratio = uniform_ratio
        self.alpha = alpha
        self.failed = torch.zeros(self.n_bins, device=device)
        self.current = torch.zeros(self.n_bins, device=device)
        kernel = torch.tensor([lam**i for i in range(kernel_size)], device=device)
        self.kernel = (kernel / kernel.sum()).view(1, 1, -1)

    def probabilities(self):
        p = self.failed + self.uniform_ratio / self.n_bins
        p = torch.nn.functional.pad(p.view(1, 1, -1),
                                    (0, self.kernel.shape[-1] - 1), mode="replicate")
        p = torch.nn.functional.conv1d(p, self.kernel).view(-1)
        return p / p.sum()

    def sample(self, n, generator=None):
        bins = torch.multinomial(self.probabilities(), n, replacement=True,
                                 generator=generator)
        jitter = torch.rand(n, device=bins.device, generator=generator)
        return ((bins + jitter) / self.n_bins * (self.n_frames - 1)).long()

    def record(self, time_steps, failed_mask):
        if failed_mask.any():
            bins = torch.clamp((time_steps * self.n_bins) // max(self.n_frames, 1),
                               0, self.n_bins - 1)
            self.current = torch.bincount(bins[failed_mask],
                                          minlength=self.n_bins).float()

    def decay(self):
        self.failed = self.alpha * self.current + (1 - self.alpha) * self.failed
        self.current = torch.zeros_like(self.current)
