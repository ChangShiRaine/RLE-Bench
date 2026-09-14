"""harness/verifier/recording.py: which scored episodes are filmed and which clip each task shows."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tasks" / "task05" / "harness" / "verifier" / "recording.py"
spec = importlib.util.spec_from_file_location("task05_recording", SCRIPT)
recording = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recording)


def test_designate_takes_the_first_episodes_of_each_task_in_order():
    clips = recording.designate({"task0": [("standard", "libero_10", 0, e) for e in range(50)],
                                 "KITCHEN SCENE/3": [("plus", "a.bddl", 7)]})
    assert list(clips.values()) == [f"task0__{i}" for i in range(recording.RECORDED)] + ["KITCHEN_SCENE_3__0"]
    assert json.loads(next(iter(clips))) == ["standard", "libero_10", 0, 0]


def test_choose_prefers_a_success_and_skips_unfilmed_tasks():
    clips = recording.designate({"a": [("a", 1), ("a", 2), ("a", 3)], "b": [("b", 1), ("b", 2)],
                                 "c": [("c", 1)]})
    outcomes = {json.dumps(["a", 2]): True, json.dumps(["b", 1]): False}
    filmed = {"a__0", "a__1", "a__2", "b__0", "b__1"}
    assert recording.choose(clips, outcomes, filmed) == ["a__1", "b__0"]


def test_undesignated_episodes_get_no_recorder():
    class Args:
        record, clips = "/tmp/media", {}
    assert recording.recorder(Args(), ("a", 1), every=2) is None
