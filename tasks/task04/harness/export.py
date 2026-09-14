"""Checkpoint -> submission: policy.onnx plus its metadata.

ONNX is the deliverable because the evaluator must run the policy without
importing any of the submission's code — a graph of weights carries no training
framework, no custom classes and nothing to execute at load time.

Observation normalisation is baked INTO the graph. Almost every RL recipe
normalises observations by a running mean and variance, and those statistics are
part of the policy: a normaliser left outside would have to be described to the
evaluator and trusted, which is exactly the dependency the ONNX boundary exists
to remove.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from . import spec


def export(actor, path: str, obs_history: int = 1, obs_mean=None, obs_std=None,
           extra: dict | None = None) -> dict:
    import torch

    os.makedirs(path, exist_ok=True)
    obs_dim = spec.OBS_DIM * obs_history
    mean = torch.zeros(obs_dim) if obs_mean is None else torch.as_tensor(obs_mean).float().detach()
    std = torch.ones(obs_dim) if obs_std is None else torch.as_tensor(obs_std).float().detach()

    graph = _Normalised(copy.deepcopy(actor).cpu(), mean.cpu(), std.cpu()).eval()
    onnx_path = os.path.join(path, spec.POLICY_FILE)
    torch.onnx.export(
        graph, torch.zeros(1, obs_dim), onnx_path,
        input_names=[spec.OBS_INPUT], output_names=[spec.ACTION_OUTPUT],
        opset_version=18, export_params=True,
        external_data=False,
    )
    meta = {
        "obs_history_length": int(obs_history),
        "obs_dim": obs_dim,
        "action_dim": spec.N_ACTIONS,
        "n_params": count_parameters(onnx_path),
    }
    meta.update(extra or {})
    with open(os.path.join(path, spec.META_FILE), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


class _Normalised:

    def __new__(cls, actor, mean, std):
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = actor
                self.register_buffer("mean", mean)
                self.register_buffer("std", std)

            def forward(self, obs):
                return self.actor((obs - self.mean) / self.std)

        return Wrapper()


def count_parameters(onnx_path: str) -> int:
    import onnx

    model = onnx.load(onnx_path)
    return int(sum(np.prod(t.dims) for t in model.graph.initializer))


def load_meta(path: str) -> dict:
    with open(os.path.join(path, spec.META_FILE)) as f:
        return json.load(f)


def validate(submission_dir: str) -> list[str]:
    import onnx
    import onnxruntime

    problems = []
    onnx_path = os.path.join(submission_dir, spec.POLICY_FILE)
    if not os.path.exists(onnx_path):
        return [f"{spec.POLICY_FILE} missing"]
    try:
        meta = load_meta(submission_dir)
    except (OSError, ValueError) as exc:
        return [f"{spec.META_FILE} unreadable: {exc}"]

    history = meta.get("obs_history_length", 1)
    if not isinstance(history, int) or not 1 <= history <= spec.MAX_HISTORY:
        return [f"obs_history_length {history!r} outside 1..{spec.MAX_HISTORY}"]

    try:
        graph = onnx.load(onnx_path)
        onnx.checker.check_model(graph)
    except Exception as exc:
        return [f"policy.onnx will not load: {exc}"]

    n_params = int(sum(np.prod(t.dims) for t in graph.graph.initializer))
    if n_params > spec.MAX_PARAMS:
        problems.append(f"{n_params} parameters exceeds the {spec.MAX_PARAMS} budget")

    inputs = [i.name for i in graph.graph.input]
    outputs = [o.name for o in graph.graph.output]
    if inputs != [spec.OBS_INPUT]:
        problems.append(f"expected one input named {spec.OBS_INPUT!r}, found {inputs}")
    if outputs != [spec.ACTION_OUTPUT]:
        problems.append(f"expected one output named {spec.ACTION_OUTPUT!r}, found {outputs}")
    if problems:
        return problems

    try:
        session = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])
        out = session.run([spec.ACTION_OUTPUT], {
            spec.OBS_INPUT: np.zeros((1, spec.OBS_DIM * history), np.float32)})[0]
    except Exception as exc:
        return [f"policy.onnx will not run: {exc}"]

    if out.shape != (1, spec.N_ACTIONS):
        problems.append(f"actions have shape {out.shape}, expected (1, {spec.N_ACTIONS})")
    elif not np.isfinite(out).all():
        problems.append("policy returned non-finite actions on a zero observation")
    return problems
