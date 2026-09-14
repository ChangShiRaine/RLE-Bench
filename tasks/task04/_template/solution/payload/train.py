"""Reference solution for task04: PPO on the tracking environment.

Both the Harbor Oracle and agent-facing starter code: a deliberately ordinary
baseline with the hyperparameters from BeyondMimic's rsl_rl config, an
asymmetric critic on the privileged observation, and empirical observation
normalisation. A submission is free to modify it or do something better.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn

from harness import export, progress, spec
from harness.env import TrackingEnv, TrackingEnvCfg

# rsl_rl_ppo_cfg.G1FlatPPORunnerCfg, verbatim.
STEPS_PER_ENV = 24
EPOCHS = 5
MINIBATCHES = 4
CLIP = 0.2
ENTROPY_COEF = 0.005
VALUE_COEF = 1.0
GAMMA = 0.99
LAM = 0.95
LR = 1.0e-3
DESIRED_KL = 0.01
MAX_GRAD_NORM = 1.0
HIDDEN = (512, 256, 128)


def mlp(inp, out, hidden=HIDDEN):
    layers, last = [], inp
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ELU()]
        last = h
    return nn.Sequential(*layers, nn.Linear(last, out))


class RunningNorm(nn.Module):
    """Welford statistics, frozen into the exported graph at the end."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        n = x.shape[0]
        delta = x.mean(0) - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var[:] = (self.var * self.count + x.var(0, unbiased=False) * n
                       + delta.square() * self.count * n / total) / total
        self.count += n

    def forward(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1e-8)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, priv_dim, act_dim):
        super().__init__()
        self.actor = mlp(obs_dim, act_dim)
        self.critic = mlp(priv_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.obs_norm = RunningNorm(obs_dim)
        self.priv_norm = RunningNorm(priv_dim)

    def distribution(self, obs):
        return torch.distributions.Normal(self.actor(self.obs_norm(obs)),
                                          self.log_std.exp())

    def value(self, priv):
        return self.critic(self.priv_norm(priv)).squeeze(-1)


def _stage(net, out_dir, iteration, elapsed, curve_dir=None):
    """Write the deliverable, then keep a dated copy for the learning curve.

    Staged repeatedly rather than once at the end: whatever is staged when the
    clock runs out is what gets scored, so a run that is cut short still submits
    its best-so-far instead of nothing.
    """
    meta = export.export(net.actor, out_dir, obs_history=1,
                         obs_mean=net.obs_norm.mean.cpu(),
                         obs_std=(net.obs_norm.var.sqrt() + 1e-8).cpu(),
                         extra={"iterations": iteration, "algorithm": "ppo",
                                "minutes": round(elapsed / 60.0, 2)})
    if curve_dir:
        snap = os.path.join(curve_dir, f"min_{elapsed / 60.0:06.1f}")
        os.makedirs(snap, exist_ok=True)
        for name in (spec.POLICY_FILE, spec.META_FILE):
            shutil.copy(os.path.join(out_dir, name), snap)
    return meta


def train(minutes, num_envs, out_dir, seed, robot_dir, motion, device="cuda",
          report_every=300.0, curve_dir=None, log_path=None):
    torch.manual_seed(seed)
    env = TrackingEnv(TrackingEnvCfg(num_envs=num_envs, seed=seed, device=device),
                      robot_dir, motion)
    obs = env.reset()
    priv = env.privileged_observation()
    net = ActorCritic(spec.OBS_DIM, spec.PRIVILEGED_DIM, spec.N_ACTIONS).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lr = LR

    log = progress.TrainingLog(log_path)
    start = time.time()
    deadline = start + minutes * 60
    next_report = start + report_every
    iteration = 0
    while time.time() < deadline:
        obs_buf, priv_buf, act_buf, logp_buf = [], [], [], []
        rew_buf, done_buf, val_buf, mu_buf, sigma_buf = [], [], [], [], []

        with torch.no_grad():
            for _ in range(STEPS_PER_ENV):
                dist = net.distribution(obs)
                action = dist.sample()
                obs_buf.append(obs)
                priv_buf.append(priv)
                act_buf.append(action)
                logp_buf.append(dist.log_prob(action).sum(-1))
                mu_buf.append(dist.mean)
                sigma_buf.append(dist.stddev.expand_as(dist.mean))
                val_buf.append(net.value(priv))
                obs, reward, done, info = env.step(action)
                priv = env.privileged_observation()
                # Bootstrap through a timeout: the episode ended because the clip
                # ran out, not because the policy failed.
                rew_buf.append(reward)
                done_buf.append(done & ~info["timeout"])
            last_value = net.value(priv)

        obs_b = torch.stack(obs_buf)
        priv_b = torch.stack(priv_buf)
        act_b = torch.stack(act_buf)
        logp_b = torch.stack(logp_buf)
        val_b = torch.stack(val_buf)
        rew_b = torch.stack(rew_buf)
        alive = 1.0 - torch.stack(done_buf).float()

        advantage = torch.zeros_like(rew_b)
        gae = torch.zeros_like(rew_b[0])
        for t in reversed(range(STEPS_PER_ENV)):
            nxt = last_value if t == STEPS_PER_ENV - 1 else val_b[t + 1]
            delta = rew_b[t] + GAMMA * alive[t] * nxt - val_b[t]
            gae = delta + GAMMA * LAM * alive[t] * gae
            advantage[t] = gae
        returns = advantage + val_b
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
        obs_f, priv_f = flat(obs_b), flat(priv_b)
        net.obs_norm.update(obs_f)
        net.priv_norm.update(priv_f)
        act_f, logp_f = flat(act_b), flat(logp_b)
        mu_f, sigma_f = flat(torch.stack(mu_buf)), flat(torch.stack(sigma_buf))
        adv_f, ret_f = flat(advantage), flat(returns)

        batch = obs_f.shape[0]
        for _ in range(EPOCHS):
            perm = torch.randperm(batch, device=device)
            for chunk in perm.chunk(MINIBATCHES):
                dist = net.distribution(obs_f[chunk])
                logp = dist.log_prob(act_f[chunk]).sum(-1)
                ratio = (logp - logp_f[chunk]).exp()
                surrogate = torch.min(
                    ratio * adv_f[chunk],
                    ratio.clamp(1 - CLIP, 1 + CLIP) * adv_f[chunk]).mean()
                value_loss = (net.value(priv_f[chunk]) - ret_f[chunk]).square().mean()
                loss = -surrogate + VALUE_COEF * value_loss \
                    - ENTROPY_COEF * dist.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                opt.step()

                # rsl_rl's adaptive learning rate, on the closed-form Gaussian
                # KL between the old and new policy.
                with torch.no_grad():
                    sigma_new = dist.stddev.expand_as(dist.mean)
                    kl = ((sigma_new / sigma_f[chunk]).log()
                          + (sigma_f[chunk].square()
                             + (mu_f[chunk] - dist.mean).square())
                          / (2.0 * sigma_new.square()) - 0.5).sum(-1).mean()
                lr = min(1e-2, lr * 1.5) if kl < DESIRED_KL / 2 else \
                    max(1e-5, lr / 1.5) if kl > DESIRED_KL * 2 else lr
                for group in opt.param_groups:
                    group["lr"] = lr

        iteration += 1
        if iteration % 25 == 0:
            log.log(iteration=iteration, reward=float(rew_b.mean()), lr=lr,
                    std=float(net.log_std.exp().mean().detach()),
                    alive=float(alive.mean()),
                    min_left=(deadline - time.time()) / 60.0,
                    **{k: float(v.mean()) for k, v in info["terms"].items()})
        if time.time() >= next_report:
            # Restage, then score the staged file on the design seeds. Reward is
            # whatever this recipe decided reward means; tracking_multi is what
            # the verifier will actually measure.
            _stage(net, out_dir, iteration, time.time() - start, curve_dir)
            log.log(iteration=iteration, staged=True,
                    **progress.evaluate(out_dir, robot_dir, env.motion))
            next_report = time.time() + report_every

    meta = _stage(net, out_dir, iteration, time.time() - start, curve_dir)
    print(f"exported after {iteration} iterations: {meta}", flush=True)
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--report-every", type=float, default=300.0,
                   help="seconds between restaging and a design-seed evaluation")
    p.add_argument("--log", default="/logs/artifacts/train_log.jsonl")
    p.add_argument("--curve-dir", default=None,
                   help="diagnostic: also keep a dated snapshot per report")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--out", default="/logs/artifacts/policy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--robot-dir", default=os.environ.get(
        "MOTIONTRACK_ROBOT_DIR", "/workspace/assets/unitree_g1"))
    p.add_argument("--motion", default=os.environ.get("MOTIONTRACK_MOTION"),
                   required="MOTIONTRACK_MOTION" not in os.environ)
    args = p.parse_args()
    train(args.minutes, args.num_envs, args.out, args.seed,
          args.robot_dir, args.motion, report_every=args.report_every,
          curve_dir=args.curve_dir, log_path=args.log)
