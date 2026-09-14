"""Three-experiment submission contract and independent rigid-body checks."""
import json
from pathlib import Path
import tomllib

import pytest

from tasks.task03.tabletop.hidden_com.config import CASES, MAX_STEPS
from tasks.task03.tabletop.hidden_com.verify import score


@pytest.mark.parametrize("answer,expected", [("A", 1), ("B", 0), (None, 0), ("", 0)])
def test_score_uses_committed_answer(answer, expected):
    assert score({"trials": [dict(trial=1, quadrant="A", answer=answer, closed=True, steps=100)]}, cases=CASES[:1])["reward"] == pytest.approx(expected)


@pytest.mark.parametrize("patch", [{"closed": False}, {"steps": -1}, {"steps": True},
                                    {"steps": MAX_STEPS+1}, {"quadrant": "E"}])
def test_invalid_record_never_scores(patch):
    record = dict(trial=1, quadrant="A", answer="A", closed=True, steps=100)
    record.update(patch)
    assert score({"trials": [record]}, cases=CASES[:1])["reward"] == 0


def test_single_stage_template():
    root = Path(__file__).resolve().parents[1] / "tasks/task03/_template/hidden_com"
    config = tomllib.loads((root / "task.toml").read_text())
    assert "steps" not in config and "multi_step_reward_strategy" not in config
    assert config["agent"]["user"] == "agent"
    assert config["environment"]["network_mode"] == "public"
    assert config["agent"]["network_mode"] == "allowlist"
    assert config["agent"]["allowed_hosts"] == []
    assert config["verifier"]["network_mode"] == "no-network"
    assert not (root / "steps").exists()


def session(tmp_path, cases=CASES[:1]):
    pytest.importorskip("robosuite")
    from tasks.task03.tabletop.hidden_com.service import Session

    class Env:
        def __init__(self, quadrant, case):
            pass
        def reset(self):
            pass
        def close(self):
            pass

    return Session(tmp_path / "state.json", env_factory=Env, cases=cases)


def test_submission_is_final_and_does_not_reveal_correctness(tmp_path):
    s = session(tmp_path)
    response = s.handle({"op": "submit", "trial": 1, "quadrant": s.quadrant}, 1000)
    assert response == {"submitted": True, "quadrant": s.quadrant, "steps": 0, "trial": 1, "remaining_trials": 0, "done": True}
    for op in ("submit", "step", "move", "observe", "reset"):
        with pytest.raises(ValueError):
            s.handle({"op": op, "quadrant": "A"}, 1000)
    record = json.loads(s.state_path.read_text())
    assert score(record, cases=CASES[:1])["reward"] == 1
    assert s.state_path.stat().st_mode & 0o777 == 0o600


def test_invalid_requests_cannot_finalize_or_change_answer(tmp_path):
    s = session(tmp_path)
    for request in ({"op": "finalize"}, {"op": "reset"}, {"op": "submit", "trial": 1, "quadrant": ""},
                    {"op": "submit", "trial": 1, "quadrant": "AB"},
                    {"op": "step", "trial": 1, "action": [float("nan")]*7},
                    {"op": "step", "trial": 1, "action": [2]*7},
                    {"op": "step", "trial": 1, "action": [0]*7, "steps": True}):
        with pytest.raises(ValueError):
            s.handle(request, 1000)
    assert s.answer is None and not s.closed and s.steps == 0
    s.handle({"op": "finalize"}, 0)
    assert score(json.loads(s.state_path.read_text()), cases=CASES[:1])["reward"] == 0


def test_budget_exhaustion_still_allows_submission(tmp_path):
    s = session(tmp_path)
    s.steps = MAX_STEPS
    with pytest.raises(ValueError):
        s.handle({"op": "step", "trial": 1, "action": [0]*7}, 1000)
    s.handle({"op": "submit", "trial": 1, "quadrant": "A"}, 1000)
    assert s.closed


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("quadrant", list("ABCD"))
def test_mass_properties_against_independent_compound_body(quadrant, case):
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    import numpy as np
    from scipy.spatial.transform import Rotation
    from tasks.task03.tabletop.hidden_com.scene import mass_properties, SIGNS, HANDLE_PARTS
    x, y = np.array(SIGNS[quadrant]) * case.offset_xy
    handle_xml = "".join(f'<geom type="box" size="{" ".join(map(str, half))}" mass="{mass}" pos="{" ".join(map(str, pos))}"/>'
                         for _, mass, half, pos in HANDLE_PARTS)
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody><body>
      <freejoint/><geom type="box" size=".075 .075 .025" mass="{case.shell_mass}"/>
      <geom type="box" size="{' '.join(map(str, case.ballast_half))}" mass="{case.ballast_mass}" pos="{x} {y} 0"/>
    {handle_xml}</body></worldbody></mujoco>''')
    mass, center, inertia = mass_properties(quadrant, case)
    axes = Rotation.from_quat(model.body_iquat[1], scalar_first=True).as_matrix()
    np.testing.assert_allclose(mass, model.body_mass[1], atol=1e-12)
    np.testing.assert_allclose(center, model.body_ipos[1], atol=1e-12)
    np.testing.assert_allclose(inertia, axes @ np.diag(model.body_inertia[1]) @ axes.T, atol=2e-10)
    assert np.linalg.eigvalsh(inertia).min() > 0


def test_box_markings_preserve_inertia_and_cannot_collide():
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    import numpy as np
    import xml.etree.ElementTree as ET
    from tasks.task03.tabletop.hidden_com.scene import add_labels

    root = ET.fromstring('<mujoco><worldbody><body><freejoint/>'
                         '<geom type="box" size=".075 .075 .025" mass=".5"/>'
                         '</body></worldbody></mujoco>')
    original = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    add_labels(root.find("worldbody/body"))
    marked = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for field in ("body_mass", "body_ipos"):
        np.testing.assert_array_equal(getattr(marked, field), getattr(original, field))
    from scipy.spatial.transform import Rotation
    def inertia(model):
        axes = Rotation.from_quat(model.body_iquat[1], scalar_first=True).as_matrix()
        return axes @ np.diag(model.body_inertia[1]) @ axes.T
    np.testing.assert_allclose(inertia(marked), inertia(original), atol=1e-15)
    assert not marked.geom_contype[1:].any()
    assert not marked.geom_conaffinity[1:].any()


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("quadrant", list("ABCD"))
def test_rgb_only_physical_oracle(quadrant, case, tmp_path):
    pytest.importorskip("robosuite")
    import base64
    from io import BytesIO
    import numpy as np
    from PIL import Image
    from tasks.task03.tabletop.hidden_com.service import Session
    from tasks.task03.tabletop.hidden_com.scene import HiddenCOM
    from tasks.task03._template.hidden_com.solution.oracle import solve

    s = Session(tmp_path / "state.json", env_factory=lambda _, case: HiddenCOM(quadrant, case=case), cases=(case,))
    s.quadrant = quadrant

    class Client:
        def call(self, request):
            request["trial"] = 1
            result = s.handle(request, 1000)
            if "images" in result:
                result["images"] = {k: np.asarray(Image.open(BytesIO(base64.b64decode(v))))
                                    for k, v in result["images"].items()}
            return result
        def observe(self):
            return self.call({"op": "observe"})
        def move(self, position, quaternion, steps, gripper=1):
            return self.call(dict(op="move", trial=1, position=position, quaternion=quaternion, gripper=gripper, steps=steps))
        def submit(self, answer):
            return self.call(dict(op="submit", trial=1, quadrant=answer))

    try:
        result = solve(Client())
        assert result["quadrant"] == quadrant
        assert result["steps"] == case.oracle_steps
        assert score(json.loads(s.state_path.read_text()), cases=CASES[:1])["reward"] == pytest.approx(1)
        assert not any(w.number for w in s.env.sim.data._data.warning)
    finally:
        s.env.close()


@pytest.mark.parametrize("case", CASES)
def test_initial_rgb_does_not_reveal_quadrant(case):
    pytest.importorskip("robosuite")
    import numpy as np
    from tasks.task03.tabletop.hidden_com.scene import HiddenCOM
    from tasks.task03.tabletop.hidden_com.render import render
    reference = None
    for quadrant in "ABCD":
        env = HiddenCOM(quadrant, case=case)
        try:
            env.reset()
            pixels = np.asarray(render(env, "top"))
            if reference is None:
                reference = pixels.copy()
            else:
                np.testing.assert_array_equal(pixels, reference)
        finally:
            env.close()


def test_observations_survive_delayed_garbage_collection(tmp_path):
    pytest.importorskip("robosuite")
    import gc
    from tasks.task03.tabletop.hidden_com.service import Session
    s = Session(tmp_path / "state.json", cases=CASES[:1])
    try:
        before = s.observe()["images"]
        gc.collect()
        assert s.observe()["images"] == before
        s.handle(dict(op="move", trial=1, position=[-.13, 0, 1.0],
                      quaternion=[1, 0, 0, 0], steps=80), 1000)
        before = s.observe()["images"]
        gc.collect()
        assert s.observe()["images"] == before
    finally:
        s.env.close()


def test_recording_preserves_physics_and_observations(tmp_path, monkeypatch):
    pytest.importorskip("robosuite")
    import gc
    import numpy as np
    import imageio_ffmpeg
    from rlebench.core.media import ffmpeg_available
    from tasks.task03.tabletop.hidden_com.service import Session
    if not ffmpeg_available():
        pytest.skip("ffmpeg unavailable")
    monkeypatch.setenv("RLEBENCH_MEDIA", "1")
    snapshots = []
    for recording in (False, True):
        s = Session(tmp_path / "state.json", video_root=tmp_path / "media" if recording else None, cases=CASES[:1])
        try:
            s.handle(dict(op="move", trial=1, position=[-.13, 0, 1.0],
                          quaternion=[1, 0, 0, 0], steps=12), 1000)
            gc.collect()
            snapshots.append((s.env.sim.data.qpos.copy(), s.observe()))
            s.handle(dict(op="submit", trial=1, quadrant=s.quadrant), 1000)
            s.handle(dict(op="finalize"), 0)
            assert score(json.loads(s.state_path.read_text()), cases=CASES[:1])["reward"] == pytest.approx(1)
        finally:
            s.env.close()
    np.testing.assert_array_equal(snapshots[0][0], snapshots[1][0])
    assert snapshots[0][1] == snapshots[1][1]
    index = json.loads((tmp_path / "media/trial-01/index.json").read_text())
    assert index["files"] == ["interaction.mp4"] and not index["skipped"]
    frames = imageio_ffmpeg.read_frames(str(tmp_path / "media/trial-01/interaction.mp4"))
    metadata = next(frames)
    assert metadata["size"] == (1024, 576)
    assert metadata["fps"] == 10
    assert sum(1 for _ in frames) == 17  # Initial + six sampled steps + one-second final hold.


def test_recording_failure_does_not_change_submission(tmp_path, monkeypatch):
    pytest.importorskip("robosuite")
    from tasks.task03.tabletop.hidden_com.service import Session
    from tasks.task03.tabletop.hidden_com.video import InteractionVideo
    def fail(*args):
        raise RuntimeError("encoder/render unavailable")
    monkeypatch.setattr(InteractionVideo, "render", fail)
    s = Session(tmp_path / "state.json", video_root=tmp_path / "media", cases=CASES[:1])
    try:
        s.handle(dict(op="submit", trial=1, quadrant=s.quadrant), 1000)
        assert score(json.loads(s.state_path.read_text()), cases=CASES[:1])["reward"] == 1
    finally:
        s.env.close()


def committed(trial, steps, answer="A"):
    return dict(trial=trial, quadrant="A", answer=answer, closed=True, steps=steps)


@pytest.mark.parametrize("steps,expected", [(0, 1), (240, 1), (MAX_STEPS // 2, 1), (MAX_STEPS, 1), (MAX_STEPS+1, 0)])
def test_quality_score_with_budget_validation(steps, expected):
    record = {"trials": [committed(i, steps) for i in range(1, 4)]}
    assert score(record)["reward"] == pytest.approx(expected)


def test_average_includes_wrong_missing_and_duplicate_trials():
    assert score({"trials": [committed(1, 240)]})["reward"] == pytest.approx(1/3)
    assert score({"trials": [committed(1, 240), committed(2, MAX_STEPS), committed(3, 50, "B")]})["reward"] == pytest.approx(2/3)
    assert score({"trials": [committed(1, 240)] * 3})["reward"] == pytest.approx(1/3)


def test_three_trials_reset_budget_and_reject_stale_submissions(tmp_path):
    s = session(tmp_path, cases=CASES)
    for trial in range(1, 4):
        assert s.steps == 0
        s.steps = MAX_STEPS - 1
        with pytest.raises(ValueError, match="budget exhausted"):
            s.handle(dict(op="step", trial=trial, action=[0]*7, steps=2), 1000)
        assert s.steps == MAX_STEPS - 1
        s.steps = MAX_STEPS
        answer = s.quadrant
        response = s.handle(dict(op="submit", trial=trial, quadrant=answer), 1000)
        assert response["done"] == (trial == 3)
        assert "correct" not in response
        with pytest.raises(ValueError):
            s.handle(dict(op="submit", trial=trial, quadrant=answer), 1000)
    result = score(json.loads(s.state_path.read_text()))
    assert result["reward"] == pytest.approx(1)
    assert result["control_steps"] == 3 * MAX_STEPS
    assert result["correct"] == 3


def test_finalize_preserves_completed_trial_and_scores_unfinished_zero(tmp_path):
    s = session(tmp_path, cases=CASES)
    s.handle(dict(op="submit", trial=1, quadrant=s.quadrant), 1000)
    s.handle(dict(op="finalize"), 0)
    record = json.loads(s.state_path.read_text())
    assert len(record["trials"]) == 2
    assert record["active"] is None
    assert score(record)["reward"] == pytest.approx(1/3)


def test_exact_cap_counts_actual_control_advances(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    s = session(tmp_path)
    advances = []
    s.env.sim = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(1)))
    s.env.step = lambda action: advances.append(action.copy())
    s.observe = lambda: {"steps": s.steps, "remaining_steps": MAX_STEPS-s.steps}
    for _ in range(MAX_STEPS // 200):
        result = s.handle(dict(op="step", trial=1, action=[0]*7, steps=200), 1000)
    assert len(advances) == MAX_STEPS == result["steps"]
    assert result["remaining_steps"] == 0
    with pytest.raises(ValueError, match="budget exhausted"):
        s.handle(dict(op="step", trial=1, action=[0]*7), 1000)
    assert len(advances) == MAX_STEPS
    s.handle(dict(op="submit", trial=1, quadrant=s.quadrant), 1000)
    assert score(json.loads(s.state_path.read_text()), cases=CASES[:1])["reward"] == 1


@pytest.mark.parametrize("case", CASES)
def test_handle_supports_physical_lift_and_transport(case, tmp_path):
    pytest.importorskip("robosuite")
    import numpy as np
    from tasks.task03.tabletop.hidden_com.service import Session
    s = Session(tmp_path / "state.json", cases=(case,))
    try:
        for position, gripper in (
            ([-.13, -.25, 1.10], -1), ([0, 0, 1.05], -1),
            ([0, 0, .86], -1), ([0, 0, .86], 1),
            ([0, 0, 1.06], 1), ([.04, -.20, 1.06], 1),
        ):
            s.handle(dict(op="move", trial=1, position=position,
                          quaternion=[2**-.5, 2**-.5, 0, 0],
                          gripper=gripper, steps=120), 1000)
        box_position = s.env.sim.data.get_joint_qpos(s.env.box.joints[0])[:3]
        assert box_position[2] > .94
        for name in ("point_base", "point_stem", "point_tip", "edge_base", "edge_rail", "support_1", "support_2"):
            assert name not in s.env.sim.model.geom_names
        np.testing.assert_allclose(box_position[:2], [.04, -.20], atol=.025)
        assert not any(w.number for w in s.env.sim.data._data.warning)
    finally:
        s.env.close()
