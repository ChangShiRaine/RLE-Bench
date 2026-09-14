"""Tensor contract for rgb-depth-model-training subtask (learned pose estimation).

The deliverable for rgb-depth-model-training is a TorchScript module saved on CPU. This module is
the single source of truth for how observations become input tensors and how
output vectors become poses — the evaluation packs frames with exactly these
functions, so train against them.

Contract:
    input  [5, H, W] float32 — RGB (3 channels, scaled to 0..1),
                               depth (1 channel, meters, NaN -> 0),
                               validity mask (1 channel, 1 where depth valid)
    output [7] float32       — (x, y, cos(theta), sin(theta), T/C/F logits)
"""
from __future__ import annotations

import numpy as np
import torch

from . import spec

MODEL_FILENAME = "model.pt"


def parameter_count(module: torch.nn.Module) -> int:
    """Count serialized tensor state, including frozen graph constants.

    Counting parameters alone is insufficient: `torch.jit.freeze` moves
    weights into graph constants. Buffers and tensor constants therefore count
    toward the same public model-size limit.
    """
    count = sum(int(tensor.numel()) for tensor in module.parameters())
    count += sum(int(tensor.numel()) for tensor in module.buffers())
    graph = getattr(module, "inlined_graph", None)
    if graph is not None:
        for node in graph.nodes():
            if node.kind() != "prim::Constant" \
                    or "value" not in node.attributeNames() \
                    or node.kindOf("value") != "t":
                continue
            count += int(node.t("value").numel())
    return count


def validate_model_size(module: torch.nn.Module) -> int:
    """Enforce the public model-size contract and return the parameter count."""
    count = parameter_count(module)
    if count > spec.MODEL_MAX_PARAMETERS:
        raise ValueError(
            f"model has {count:,} tensor-state elements; "
            f"limit is {spec.MODEL_MAX_PARAMETERS:,}"
        )
    return count


def pack_observation(rgb: np.ndarray, depth: np.ndarray) -> torch.Tensor:
    """One RGB-D observation -> the [5, H, W] float32 input tensor."""
    rgbf = torch.from_numpy(np.ascontiguousarray(rgb)).float() / 255.0
    d = torch.from_numpy(np.ascontiguousarray(depth)).float()
    valid = torch.isfinite(d) & (d > 0)
    d = torch.where(valid, d, torch.zeros_like(d))
    return torch.cat([rgbf.permute(2, 0, 1), d[None], valid[None].float()],
                     dim=0)


def decode_output(vec: torch.Tensor) -> tuple[float, float, float, int]:
    """[7] output -> (x, y, theta, shape_id), with 0=T, 1=C, 2=F."""
    v = vec.detach().reshape(-1).float()
    if v.numel() != 7 or not bool(torch.isfinite(v).all()):
        raise ValueError("expected seven finite model outputs")
    theta = float(torch.atan2(v[3], v[2]))
    return float(v[0]), float(v[1]), spec.wrap_angle(theta), int(torch.argmax(v[4:7]))


def encode_pose(x: float, y: float, theta: float) -> torch.Tensor:
    """(x, y, theta) -> the [4] training target."""
    return torch.tensor([x, y, np.cos(theta), np.sin(theta)],
                        dtype=torch.float32)


def save_model(module: torch.nn.Module, path: str) -> None:
    """Serialize for submission: TorchScript, CPU weights."""
    module = module.eval().cpu()
    validate_model_size(module)
    scripted = torch.jit.script(module)
    validate_model_size(scripted)
    torch.jit.save(scripted, path)


def load_model(path: str) -> torch.jit.ScriptModule:
    """Load a submitted model for CPU inference (the evaluation's loader)."""
    m = torch.jit.load(path, map_location="cpu")
    validate_model_size(m)
    m.eval()
    return m


def infer(model: torch.jit.ScriptModule, rgb: np.ndarray,
          depth: np.ndarray) -> tuple[float, float, float, int]:
    """Deterministic single-frame CPU inference through the contract."""
    torch.set_num_threads(1)
    with torch.no_grad():
        out = model(pack_observation(rgb, depth)[None])
    return decode_output(out[0])
