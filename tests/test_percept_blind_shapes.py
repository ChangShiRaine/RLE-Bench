"""Regression tests for Task 06 family blind three-shape contract."""
from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import struct
import tempfile
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from harness import episodes, spec
from harness import (
    artifact_staging, design_evaluator, sandbox, training_client, training_service)

REPO = Path(__file__).resolve().parents[1]
TASK_NAMES = {
    "a": "rgb-only",
    "b": "rgb-depth",
    "c": "rgb-depth-model-training",
    "d": "method-agnostic",
}
TASK = REPO / "tasks" / "task06" / TASK_NAMES["d"]
MESH_DIR = REPO / "tasks" / "task06" / "harness" / "assets" / "tshape"


def _top_area_mm2(path: Path) -> float:
    raw = path.read_bytes()
    triangles = struct.unpack_from("<I", raw, 80)[0]
    area = 0.0
    for index in range(triangles):
        values = struct.unpack_from("<12fH", raw, 84 + 50 * index)
        normal = np.asarray(values[:3])
        if normal[2] <= 0.9:
            continue
        vertices = np.asarray(
            [values[3:6], values[6:9], values[9:12]], dtype=float)
        edge_a = vertices[1, :2] - vertices[0, :2]
        edge_b = vertices[2, :2] - vertices[0, :2]
        area += abs(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]) / 2
    return float(area)


def test_collision_decompositions_match_mesh_top_area():
    for name in spec.BLOCK_SHAPES:
        mesh_area = _top_area_mm2(MESH_DIR / f"{name}.stl")
        box_area = sum(
            4 * half[0] * half[1]
            for _, half in spec.BLOCK_COLLISION_BOXES[name]
        ) * 1e6
        np.testing.assert_allclose(mesh_area, box_area, rtol=0, atol=1e-3)


def test_hidden_shape_draw_is_deterministic_and_balanced():
    first = [episodes.shape_for_seed(seed, spec.BLOCK_SHAPES)
             for seed in range(120)]
    second = [episodes.shape_for_seed(seed, spec.BLOCK_SHAPES)
              for seed in range(120)]
    assert first == second
    assert set(first) == set(spec.BLOCK_SHAPES)
    counts = {name: first.count(name) for name in spec.BLOCK_SHAPES}
    assert min(counts.values()) >= 25, counts


def test_agent_payload_contains_no_3d_assets_or_shape_draw():
    agent = TASK / "environment" / "agent"
    for name in TASK_NAMES.values():
        public = REPO / "tasks" / "task06" / name / "environment" / "agent" / "harness"
        assert not (public / "reference_labels.py").exists()
    forbidden_suffixes = {".stl", ".obj", ".msh", ".xml"}
    assert not [
        path for path in agent.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    text = "\n".join(
        path.read_text(errors="ignore")
        for path in agent.rglob("*") if path.is_file()
    )
    for secret in (
        "c_block.stl",
        "f_shape.stl",
        "tshape.stl",
        "BLOCK_COLLISION_BOXES",
        "STREAM_SHAPE",
        "shape_for_seed",
    ):
        assert secret not in text
    assert (agent / "harness" / "evaluator.py").is_file()
    for private_source in ("design_evaluator.py", "training_service.py",
                           "sandbox.py"):
        assert not (agent / "harness" / private_source).exists()


def test_private_and_verifier_payloads_have_all_three_meshes():
    expected = {"tshape.stl", "c_block.stl", "f_shape.stl"}
    private = TASK / "environment" / "private" / "harness" / "assets" / "tshape"
    verifier = TASK / "tests" / "harness" / "assets" / "tshape"
    assert expected <= {path.name for path in private.iterdir()}
    assert expected <= {path.name for path in verifier.iterdir()}


def test_method_agnostic_runs_agent_non_root_and_locks_private_tree():
    config = tomllib.loads((TASK / "task.toml").read_text())
    assert config["agent"]["user"] == "agent"
    dockerfile = (TASK / "environment" / "Dockerfile").read_text()
    assert "COPY private/ /opt/private/" in dockerfile
    assert "chmod 700 /opt/private" in dockerfile
    assert 'ENTRYPOINT ["/opt/task06-entrypoint.sh"]' in dockerfile
    assert config["environment"]["healthcheck"]["command"] == (
        "test -f /run/rlebench/task06-ready"
    )


def test_all_subsets_share_private_boundary_and_requested_gpus():
    for variant, name in TASK_NAMES.items():
        task = REPO / "tasks" / "task06" / name
        config = tomllib.loads((task / "task.toml").read_text())
        assert config["agent"]["user"] == "agent"
        assert config["agent"]["timeout_sec"] == 7200.0
        assert config["environment"].get("gpus") == (1 if variant in "cd" else None)
        dockerfile = (task / "environment" / "Dockerfile").read_text()
        expected_modalities = "rgb" if variant == "a" else "rgb,depth"
        assert f"RLEBENCH_MODALITIES={expected_modalities}" in dockerfile
        assert f"RLEBENCH_VARIANT={variant}" in dockerfile
        assert "COPY private/ /opt/private/" in dockerfile
        agent = task / "environment" / "agent"
        assert not [path for path in agent.rglob("*")
                    if path.suffix.lower() in {".stl", ".obj", ".msh", ".xml"}]
        if variant in "cd":
            assert (task / "environment" / "docker-compose.yaml").is_file()
        else:
            assert not (task / "environment" / "docker-compose.yaml").exists()


def test_learned_variant_uses_parameter_limit_not_fit_timer():
    task = REPO / "tasks" / "task06" / TASK_NAMES["c"]
    instruction = (task / "instruction.md").read_text()
    agent_percept = task / "environment" / "agent" / "harness"
    public_spec = (agent_percept / "spec.py").read_text()
    harness = (agent_percept / "train_harness.py").read_text()

    assert spec.MODEL_MAX_PARAMETERS == 20_000_000
    assert "MODEL_MAX_PARAMETERS = 20_000_000" in public_spec
    assert "20,000,000" in instruction
    assert "parameters, buffers, and frozen constants" in " ".join(instruction.split())
    assert "two-hour agent budget" in instruction
    assert "budget_s" not in harness
    assert "900 s" not in instruction


def test_private_service_masks_training_modalities():
    obs = {
        "rgb": np.zeros((4, 5, 3), dtype=np.uint8),
        "depth": np.zeros((4, 5), dtype=np.float32),
        "K": np.eye(3),
        "T_cam_table": np.eye(4),
    }
    frame = SimpleNamespace(obs=obs, gt=(0.1, -0.2, 0.3),
                            occluded=False, t=0.0)
    with np.load(BytesIO(training_service._encode([frame], "rgb")),
                 allow_pickle=False) as arrays:
        assert "rgb" in arrays.files
        assert "depth" not in arrays.files
    with np.load(BytesIO(training_service._encode([frame], "rgb,depth")),
                 allow_pickle=False) as arrays:
        assert {"rgb", "depth"} <= set(arrays.files)


def test_private_service_adds_small_deterministic_label_noise():
    obs = {
        "rgb": np.zeros((4, 5, 3), dtype=np.uint8),
        "depth": np.zeros((4, 5), dtype=np.float32),
        "K": np.eye(3),
        "T_cam_table": np.eye(4),
    }
    frames = [SimpleNamespace(obs=obs, gt=(0.1, -0.2, 3.13),
                              occluded=False, t=float(i))
              for i in range(4)]
    payload_a = training_service._encode(
        frames, "rgb,depth", label_seed=11, operation="push_episode")
    payload_b = training_service._encode(
        frames, "rgb,depth", label_seed=11, operation="push_episode")
    with np.load(BytesIO(payload_a), allow_pickle=False) as arrays:
        noisy_a = arrays["gt"].copy()
    with np.load(BytesIO(payload_b), allow_pickle=False) as arrays:
        noisy_b = arrays["gt"].copy()
    truth = np.asarray([frame.gt for frame in frames])
    assert np.array_equal(noisy_a, noisy_b)
    assert not np.array_equal(noisy_a, truth)
    assert np.all(noisy_a[:, 2] > -np.pi)
    assert np.all(noisy_a[:, 2] <= np.pi)
    assert training_service.LABEL_XY_STD_M == 0.005
    assert training_service.LABEL_THETA_STD_RAD == np.deg2rad(3.0)
    samples = training_service._noisy_gts(frames * 2500, 11, "push_episode")
    errors = samples - np.tile(truth, (2500, 1))
    errors[:, 2] = (errors[:, 2] + np.pi) % (2 * np.pi) - np.pi
    expected = np.array([0.005, 0.005, np.deg2rad(3.0)])
    assert np.allclose(errors.std(axis=0), expected, rtol=0.03)
    assert np.all(np.abs(errors.mean(axis=0)) < 0.03 * expected)
    assert np.array_equal(np.asarray([frame.gt for frame in frames]), truth)


def test_design_error_statistics_include_requested_summaries():
    truth = np.zeros((3, 3), dtype=float)
    predictions = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.2],
        [np.nan, 0.0, 0.0],
    ])
    report = design_evaluator.error_statistics(predictions, truth)
    assert report["frames"] == 3
    assert report["valid_frames"] == 2
    assert report["ci_method"].startswith("cluster bootstrap")
    for name in ("translation_error_m", "rotation_error_rad"):
        assert set(report[name]) == {"mean", "std", "median", "mean_95_ci"}
        assert len(report[name]["mean_95_ci"]) == 2


def test_push_confidence_interval_resamples_whole_episodes():
    truth = np.zeros((200, 3), dtype=float)
    predictions = truth.copy()
    predictions[100:, 0] = 0.1
    clusters = np.repeat([0, 1], 100)
    report = design_evaluator.error_statistics(
        predictions, truth, clusters=clusters)
    low, high = report["translation_error_m"]["mean_95_ci"]
    assert high - low > 0.09



def test_evaluator_rpc_dispatch_returns_only_aggregate_json(monkeypatch):
    expected = {"ok": True, "all_frames": {"frames": 3}}
    monkeypatch.setenv("RLEBENCH_VARIANT", "b")
    monkeypatch.setattr(training_service, "_evaluator_calls", 0)
    monkeypatch.setattr(
        training_service.design_evaluator,
        "evaluate_submission",
        lambda variant: expected if variant == "b" else None,
    )
    actual = json.loads(training_service.dispatch({"op": "evaluate"}))
    assert actual == {**expected, "queries_remaining": 5}


def test_evaluator_rpc_has_a_hard_query_budget(monkeypatch):
    monkeypatch.setattr(training_service, "_evaluator_calls", 0)
    monkeypatch.setattr(training_service, "MAX_EVALUATOR_CALLS", 2)
    monkeypatch.setattr(
        training_service.design_evaluator,
        "evaluate_submission",
        lambda variant: {"ok": True, "mean": 0.123456},
    )
    first = json.loads(training_service.dispatch({"op": "evaluate"}))
    second = json.loads(training_service.dispatch({"op": "evaluate"}))
    assert first["mean"] == 0.1235 and first["queries_remaining"] == 1
    assert second["queries_remaining"] == 0
    with pytest.raises(RuntimeError, match="budget exhausted"):
        training_service.dispatch({"op": "evaluate"})


def test_artifact_staging_rejects_links_without_touching_the_target(tmp_path):
    source = tmp_path / "artifacts"
    source.mkdir()
    private = tmp_path / "private.txt"
    private.write_text("secret")
    private.chmod(0o600)
    (source / "estimator.py").symlink_to(private)
    with pytest.raises(ValueError, match="links are not allowed"):
        artifact_staging.stage_submission(str(source), str(tmp_path / "staged"))
    assert private.stat().st_mode & 0o777 == 0o600


def test_sandbox_permission_step_rejects_links_without_chmod_target(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    private = tmp_path / "private.txt"
    private.write_text("secret")
    private.chmod(0o600)
    (staging / "helper.py").symlink_to(private)
    with pytest.raises(ValueError, match="contains a link"):
        sandbox._world_readable(str(staging))
    assert private.stat().st_mode & 0o777 == 0o600


def test_private_evaluator_runs_candidate_in_sandbox():
    obs = {
        "rgb": np.zeros((4, 5, 3), dtype=np.uint8),
        "depth": np.ones((4, 5), dtype=np.float32),
        "K": np.eye(3),
        "T_cam_table": np.eye(4),
        "t": 0.0,
    }
    frame = SimpleNamespace(obs=obs, gt=(0.1, -0.2, 0.3),
                            occluded=False, t=0.0)
    diagnostic = design_evaluator.DesignBattery(
        single_frames=[frame],
        episodes=[SimpleNamespace(frames=[frame, frame])],
    )
    with tempfile.TemporaryDirectory(prefix="rb05_eval_", dir="/tmp") as raw:
        directory = Path(raw)
        os.chmod(directory, 0o755)
        estimator = directory / "estimator.py"
        estimator.write_text(
            "class E:\n"
            "    def reset(self): pass\n"
            "    def update(self, **obs): return (0.1, -0.2, 0.3, 0)\n"
            "def make_estimator(): return E()\n"
        )
        os.chmod(estimator, 0o644)
        report = design_evaluator.evaluate_submission(
            "d", str(directory), diagnostic=diagnostic)
    assert report["ok"]
    assert report["all_frames"]["valid_frames"] == 3
    assert report["all_frames"]["translation_error_m"]["mean"] == 0.0
    assert report["all_frames"]["rotation_error_rad"]["mean"] == 0.0


def test_client_reconstructs_only_public_observation_keys():
    arrays = {
        "rgb": np.zeros((1, 4, 5, 3), dtype=np.uint8),
        "depth": np.zeros((1, 4, 5), dtype=np.float32),
        "K": np.eye(3),
        "T_cam_table": np.eye(4),
        "t": np.array([0.0]),
        "gt": np.array([[0.1, -0.2, 0.3]]),
        "occluded": np.array([False]),
    }
    frame = training_client._frames(arrays, render=True)[0]
    assert set(frame.obs) == {"rgb", "depth", "K", "T_cam_table", "t"}
    assert frame.gt == (0.1, -0.2, 0.3)
    assert not hasattr(frame, "shape")
    assert not hasattr(frame, "shape_id")


def test_instructions_promise_the_same_blind_interface():
    for name in TASK_NAMES.values():
        instruction = (REPO / "tasks" / "task06" / name / "instruction.md").read_text()
        instruction = " ".join(instruction.split())
        assert "Shape identity is never an input" in instruction
        assert "shape labels are not provided" in instruction
        assert "0 = T, 1 = C, 2 = F" in instruction
        assert "Pose labels are noisy" in instruction
        assert all(term not in instruction for term in ("P99", "0.3/0.7", "penalty", "99%"))
        assert "evaluator.evaluate()" in instruction
        assert "At most six calls" in instruction
