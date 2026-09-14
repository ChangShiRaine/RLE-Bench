"""The evaluation must survive things that are not the agent's competence.

A real run scored 0.1 having driven zero graded trials: the agent's CLI died mid-phase,
the harness sealed the ledger from a root collect hook, and a flat "gate" component paid
out for the sealed ledger. The reward change removes the payout; these tests cover the
other half -- that the evaluation itself does not turn one broken trial, one raising
environment, or one missing `reset()` into a lost run.

None of this changes what a run is WORTH. A trial nobody drove and a trial driven and
failed both score zero. What it changes is whether the remaining trials still get their
chance, and whether a reader can afterwards tell the two apart.
"""

from __future__ import annotations

import pytest

from harness import config as C
from harness import evaluation as E
from harness import ledger as L
from harness.obs import ObsSpec
from harness.session import Budgets, MeteredSession
from test_speedrun_session import FakeEnv


def make_session(tmp_path, *, name="rb", plan=None, steps=10_000, submissions=1,
                 success_after=-1, max_episode_steps=20, factory=None):
    made = []

    def default_factory(task, split="pretrain", scene=None):
        env = FakeEnv(success_after=success_after, split=split)
        made.append(env)
        return env

    sess = MeteredSession(
        task="T",
        budgets=Budgets(interaction_steps=steps, submissions=submissions),
        ledger=L.Ledger(tmp_path / f"{name}.jsonl"),
        env_factory=factory or default_factory,
        max_episode_steps=max_episode_steps,
        eval_plan_fn=lambda t: plan or [(1, 1, 1, i) for i in range(4)],
    )
    return sess, made



# More actions than any episode here can absorb, so only the
# harness can stop it.
MANY = [[0.0] * 12] * 5000


# -- one broken trial must not cost the others -------------------------------

def test_a_scene_that_cannot_be_built_loses_only_its_own_trials(tmp_path):
    """Building a scene touches RoboCasa and can fail. Before, that exception escaped
    to the wire and the trials behind it were never attempted."""
    plan = [(1, 1, 1, 0), (2, 2, 2, 1), (3, 3, 3, 2)]
    built = []

    def factory(task, split="pretrain", scene=None):
        built.append(scene)
        if scene == (2, 2):
            raise RuntimeError("this scene is broken")
        return FakeEnv(success_after=1, split=split)

    sess, _ = make_session(tmp_path, name="badscene", plan=plan, factory=factory)
    sess.begin_evaluation()
    while sess.evaluating:
        sess.step(MANY)
        if sess.evaluating:
            sess.reset()
    summary = sess.close()

    # The broken scene was reached, not skipped over -- otherwise this would pass for
    # the wrong reason.
    assert (2, 2) in built
    # Three trials scored, and the two good scenes still succeeded.
    assert summary["trials_total"] == 3
    assert summary["best_success_rate"] == pytest.approx(2 / 3)


def test_an_environment_that_raises_mid_batch_ends_only_that_trial(tmp_path):
    """An environment that falls over loses its own trial, never the run: the trials
    behind it belong to the rest of the plan."""

    class ExplodingEnv(FakeEnv):
        def step(self, action):
            if self.split == "target" and len(self.draws) == 1:
                raise RuntimeError("mujoco fell over")
            return super().step(action)

    def factory(task, split="pretrain", scene=None):
        return ExplodingEnv(success_after=1, split=split)

    plan = [(1, 1, 1, 0), (1, 1, 1, 1)]
    sess, _ = make_session(tmp_path, name="boom", plan=plan, factory=factory)
    sess.begin_evaluation()

    res = sess.step(MANY)
    assert res["episode_over"] is True
    assert res["info"]["trial_ended"] == E.TRIAL_END_ERROR
    assert "mujoco fell over" in res["info"]["error"]

    # The evaluation is still alive and the next trial still runs.
    assert sess.evaluating
    sess.reset()
    sess.step(MANY)
    summary = sess.close()
    assert summary["trials_total"] == 2


# -- stepping never crosses a trial boundary ---------------------------------

def test_driving_on_after_a_trial_ends_is_refused_until_reset(tmp_path):
    """Only reset() advances. A controller stepping blindly past a scored trial
    fails loudly instead of landing in a trial it was never told about."""
    from harness.session import EpisodeOver

    plan = [(1, 1, 1, i) for i in range(3)]
    sess, _ = make_session(tmp_path, name="noreset", plan=plan, success_after=1)
    sess.begin_evaluation()

    sess.step(MANY)          # trial 0 scored
    with pytest.raises(EpisodeOver):
        sess.step(MANY)
    with pytest.raises(EpisodeOver):
        sess.step(MANY)
    sess.reset()
    sess.step(MANY)          # trial 1
    sess.reset()
    sess.step(MANY)          # trial 2
    summary = sess.close()

    assert summary["trials_total"] == 3
    assert summary["trials_attempted"] == 3
    assert summary["best_success_rate"] == 1.0


def test_reset_still_forfeits_a_live_trial(tmp_path):
    """The give-up move must keep costing what it costs, or skipping to a favourable
    trial would be free."""
    plan = [(1, 1, 1, i) for i in range(3)]
    sess, _ = make_session(tmp_path, name="forfeit", plan=plan, success_after=-1)
    sess.begin_evaluation()
    for _ in range(3):
        out = sess.reset()
    assert out["evaluation_done"] is True
    assert out["trials"] == 3
    assert out["success_rate"] == 0.0


def test_running_past_the_last_trial_is_refused_not_silently_looped(tmp_path):
    """Driving on past the end must not quietly re-open trials or re-score anything.

    `end_development()` first, because that is the real deployed state: by the time an
    agent is in the scored phase the development env is closed, so there is nothing for
    a stray run_segment to fall back onto.
    """
    plan = [(1, 1, 1, 0)]
    sess, _ = make_session(tmp_path, name="past", plan=plan, success_after=1)
    sess.end_development()
    sess.begin_evaluation()
    sess.step(MANY)
    sess.reset()                               # finishes the evaluation
    assert not sess.evaluating

    with pytest.raises(Exception):
        sess.step(MANY)

    summary = sess.close()
    assert summary["trials_total"] == 1        # still exactly one trial, scored once


# -- working in pieces is not counted ----------------------------------------

def test_a_long_chain_of_separate_calls_is_not_truncated(tmp_path):
    """Working in pieces IS the interface -- send a few actions, look, send more -- so
    nothing may cap how many times the agent comes back, and no segment counter may
    silently stop it."""
    sess, _ = make_session(tmp_path, name="chain", steps=10_000,
                           max_episode_steps=200)
    sess.reset()
    spent = 0
    for _ in range(200):
        res = sess.step([0.0] * 12)
        spent += res["steps"]
        if res["episode_over"]:
            break
    # 200 separate one-action calls, not 16.
    assert spent == 200


def test_sending_no_actions_costs_nothing_and_changes_nothing(tmp_path):
    """The degenerate call an agent's loop can easily make. It must be free rather than
    an error, and must not end the episode."""
    sess, _ = make_session(tmp_path, name="idle", steps=10_000,
                           max_episode_steps=1_000)
    sess.reset()
    spent = sess.state.steps_used
    res = sess.step([])
    assert res["steps"] == 0
    assert res["episode_over"] is False
    assert sess.state.steps_used == spent


# -- attempted vs failed -----------------------------------------------------

def test_unattempted_trials_are_recorded_as_such(tmp_path):
    """The distinction the failed run needed: zero because nobody drove it, versus
    zero because driving it did not work."""
    plan = [(1, 1, 1, i) for i in range(4)]
    sess, _ = make_session(tmp_path, name="undriven", plan=plan, success_after=-1)
    sess.begin_evaluation()
    sess.step(MANY)           # exactly one trial driven
    summary = sess.close()                     # the rest are never reached

    assert summary["trials_total"] == 4
    assert summary["trials_attempted"] == 1


def test_a_trial_reset_straight_past_still_counts_as_attempted(tmp_path):
    """Observed in a real run: the agent wrote `sim.reset()` twice in one expression,
    so a graded trial was opened and closed without a single action. It reached that
    trial and chose to skip it -- a result, not a trial the harness lost, and reporting
    it as unattempted read as a truncated evaluation."""
    plan = [(1, 1, 1, i) for i in range(4)]
    sess, _ = make_session(tmp_path, name="skipped", plan=plan, success_after=-1)
    sess.begin_evaluation()
    run = sess._eval
    sess.reset()              # trial 0 abandoned unread
    sess.reset()              # trial 1 too
    sess.step(MANY)           # trial 2 driven
    sess.reset()
    sess.step(MANY)           # trial 3 driven
    summary = sess.close()

    assert summary["trials_total"] == 4
    assert summary["trials_attempted"] == 4
    skipped = run.results[0]
    assert skipped.steps == 0 and skipped.ended_by == E.TRIAL_END_RESET


def test_a_run_that_drove_nothing_reports_zero_attempts(tmp_path):
    sess, _ = make_session(tmp_path, name="nothing")
    sess.begin_evaluation()
    summary = sess.close()
    assert summary["trials_attempted"] == 0
    assert summary["best_success_rate"] == 0.0


def test_unreached_trials_are_not_scored_against_a_stale_env(tmp_path):
    """Asking the PREVIOUS scene's env could report success for a trial nobody
    entered."""
    plan = [(1, 1, 1, 0), (2, 2, 2, 1), (3, 3, 3, 2)]
    sess, _ = make_session(tmp_path, name="stale", plan=plan, success_after=1)
    sess.begin_evaluation()
    sess.step(MANY)           # trial 0 succeeds
    summary = sess.close()

    assert summary["trials_total"] == 3
    # Exactly one success: the two unreached trials must not inherit it.
    assert summary["best_success_rate"] == pytest.approx(1 / 3)


def test_the_seal_records_how_the_run_ended(tmp_path):
    sess, _ = make_session(tmp_path, name="reason")
    sess.close(end_reason="phase_timeout")
    records = L.load_verified(sess.ledger.path)
    assert L.summarize(records)["end_reason"] == "phase_timeout"


# -- observe() ---------------------------------------------------------------

def test_observe_is_free_in_development(tmp_path):
    sess, _ = make_session(tmp_path, name="obsdev")
    sess.reset()
    before = sess.state.steps_used
    out = sess.observe()
    assert sess.state.steps_used == before
    assert out["obs"], "observe must return an observation"


def test_observe_is_free_during_evaluation(tmp_path):
    """Unlike `step`, which is development-only: an agent must not have to spend a
    graded trial's steps in order to look at it."""
    sess, _ = make_session(tmp_path, name="obseval", success_after=-1)
    sess.begin_evaluation()
    before_steps = sess.state.steps_used
    trial_steps_before = sess._eval.trial_steps

    out = sess.observe()

    assert sess.state.steps_used == before_steps
    assert sess._eval.trial_steps == trial_steps_before
    assert out["obs"]


def test_observe_does_not_advance_or_end_a_trial(tmp_path):
    sess, _ = make_session(tmp_path, name="obsnoadv", success_after=-1)
    sess.begin_evaluation()
    position = sess._eval.position
    for _ in range(5):
        sess.observe()
    assert sess._eval.position == position
    assert sess._eval.trial_live is True


def test_observe_before_any_reset_says_there_is_nothing_to_look_at(tmp_path):
    """Looking must work as the first thing an agent does -- answered, never an error.

    Answering by opening an episode would be wrong: a reset costs a step, so that would
    be a charge the agent never asked for and could not see, so `observe` reports the
    absence instead: `live` False and an empty obs.
    """
    sess, _ = make_session(tmp_path, name="obsfirst")
    out = sess.observe()
    assert out["obs"] == {} and out["live"] is False
    assert sess.state.steps_used == 0 and sess.state.episodes == 0


def test_observe_ledger_records_nothing(tmp_path):
    sess, _ = make_session(tmp_path, name="obsledger")
    sess.reset()
    sess.observe()
    sess.observe(ObsSpec(width=64))
    summary = sess.close()
    assert summary["interaction_steps_total"] == 1      # the reset, and nothing else


def test_requested_resolution_is_clamped_to_the_ceiling(tmp_path, monkeypatch):
    """A request above the configured ceiling is clamped, not refused: degrading the
    resolution is recoverable, failing the call is not.

    The ceiling is policy, not mechanism -- robosuite grows the offscreen framebuffer
    itself when a render exceeds it, so nothing pre-sizes anything.
    """
    sess, _ = make_session(tmp_path, name="clamp")
    sess.reset()

    captured = {}

    def fake_render(env, cameras, w, h, depth=False):
        captured["size"] = (w, h)
        return {}

    monkeypatch.setattr("harness.env.render_frames", fake_render)
    out = sess.observe(ObsSpec(width=4096, height=4096))
    assert captured["size"] == (C.OBS_MAX_RESOLUTION, C.OBS_MAX_RESOLUTION)
    assert out["resolution"] == [C.OBS_MAX_RESOLUTION, C.OBS_MAX_RESOLUTION]


def test_a_single_dimension_squares_the_request(tmp_path, monkeypatch):
    sess, _ = make_session(tmp_path, name="square")
    sess.reset()
    captured = {}
    monkeypatch.setattr("harness.env.render_frames",
                        lambda env, cams, w, h, depth=False:
                        captured.update(size=(w, h)) or {})
    sess.observe(ObsSpec(width=300))
    assert captured["size"] == (300, 300)


# -- one ObsSpec, three places -----------------------------------------------

class SpyEnv(FakeEnv):
    """An env whose observations carry images, so specs have something to shape."""

    CAMERAS = ("robot0_agentview_left", "robot0_agentview_right",
               "robot0_eye_in_hand")

    def _images(self):
        import numpy as np
        return {f"{cam}_image": np.zeros((C.OBS_RESOLUTION, C.OBS_RESOLUTION, 3),
                                         dtype="uint8")
                for cam in self.CAMERAS}

    def reset(self, seed=None):
        return {**super().reset(seed=seed), **self._images()}

    def step(self, action):
        obs, r, d, i = super().step(action)
        return {**obs, **self._images()}, r, d, i


def spy_session(tmp_path, name, **kw):
    return make_session(tmp_path, name=name,
                        factory=lambda task, split="pretrain", scene=None:
                            SpyEnv(success_after=-1, split=split), **kw)


def test_the_spec_shapes_what_comes_back(tmp_path):
    """The agent chooses what an observation contains, per call."""
    sess, _ = spy_session(tmp_path, "picky")
    sess.reset()
    res = sess.step([0.0] * 12,
                    obs_spec=ObsSpec(cameras=("robot0_eye_in_hand",)))
    assert sorted(k for k in res["obs"] if k.endswith("_image")) == \
        ["robot0_eye_in_hand_image"]


def test_no_cameras_means_no_images_at_all(tmp_path):
    """The cheap path: an agent servoing on proprioception should not pay to render,
    encode and decode three pictures it ignores."""
    sess, _ = spy_session(tmp_path, "blind")
    sess.reset()
    res = sess.step([0.0] * 12, obs_spec=ObsSpec(cameras=()))
    assert all(not k.endswith("_image") for k in res["obs"])
    assert res["obs"], "dropping the pictures must not drop the robot's own state"


def test_the_spec_is_per_call_and_does_not_stick(tmp_path):
    """Nothing is remembered between calls: an agent that looked cheaply once must not
    find itself blind for the rest of the episode."""
    sess, _ = spy_session(tmp_path, "sticky")
    sess.reset()
    sess.step([0.0] * 12, obs_spec=ObsSpec(cameras=()))
    res = sess.step([0.0] * 12)
    assert any(k.endswith("_image") for k in res["obs"])


def test_the_default_path_renders_nothing_extra(tmp_path, monkeypatch):
    """No spec, or a spec that matches what the step pipeline already produced, must not
    trigger a second render. The common case has to stay free."""
    called = []
    monkeypatch.setattr("harness.env.render_frames",
                        lambda *a, **k: called.append(1) or {})

    sess, _ = spy_session(tmp_path, "plain")
    sess.reset()
    sess.step([0.0] * 12)
    sess.step([0.0] * 12, obs_spec=ObsSpec(width=C.OBS_RESOLUTION))
    assert not called


def test_a_batch_renders_only_the_observation_it_hands_back(tmp_path, monkeypatch):
    """The wall-clock reason batching pays: fifteen actions cost one render, not
    fifteen, because only the final observation is shaped and returned."""
    rendered = []
    monkeypatch.setattr(
        "harness.env.render_frames",
        lambda env, cams, w, h, depth=False:
        rendered.append((tuple(cams), w, h)) or {})

    sess, _ = spy_session(tmp_path, "batchrender", max_episode_steps=50)
    sess.reset()
    sess.step([[0.0] * 12] * 15, obs_spec=ObsSpec(width=384))
    assert len(rendered) == 1


def test_the_harness_still_sees_everything_a_narrowed_agent_does_not(tmp_path):
    """Narrowing what the AGENT is shown must never narrow what the harness checks:
    success is the environment's predicate, computed from the real observation."""
    sess, _ = make_session(
        tmp_path, name="harnesssees",
        factory=lambda task, split="pretrain", scene=None:
            SpyEnv(success_after=1, split=split))
    sess.reset()
    res = sess.step([0.0] * 12, obs_spec=ObsSpec(cameras=()))
    assert res["success"] is True


def test_observe_without_a_size_does_not_re_render(tmp_path, monkeypatch):
    """The default path must stay free of rendering work entirely."""
    sess, _ = make_session(tmp_path, name="norerender")
    sess.reset()
    called = []
    monkeypatch.setattr("harness.env.render_frames",
                        lambda *a, **k: called.append(1) or {})
    out = sess.observe()
    assert not called
    assert out["resolution"] == C.OBS_RESOLUTION


# -- range (depth) ------------------------------------------------------------
#
# Depth is a SENSOR reading, not object state: it says how far a surface is, never what
# it is or where the task's objects are, so it stays on the agent's side of the
# observation boundary. What these pin down is that it obeys the same boundary the
# colour frames do -- the camera set, the size ceiling, and the ban on reaching a
# viewpoint the task never published.

def test_depth_is_off_unless_asked_for(tmp_path, monkeypatch):
    """It roughly doubles what an observation costs, so the default path must not pay
    for it -- and must not re-render merely to decide that."""
    sess, _ = make_session(tmp_path, name="nodepth")
    sess.reset()
    called = []
    monkeypatch.setattr("harness.env.render_frames",
                        lambda *a, **k: called.append(k.get("depth")) or {})
    sess.observe()
    assert not called
    assert ObsSpec().as_wire() == {}          # nothing on the wire either


def test_asking_for_depth_renders_even_at_the_native_size(tmp_path, monkeypatch):
    """The step pipeline never produces range, so `observe(depth=True)` at the size the
    frames already are still has to render. The size shortcut must not swallow it."""
    sess, _ = make_session(tmp_path, name="depthnative")
    sess.reset()
    asked = []
    monkeypatch.setattr(
        "harness.env.render_frames",
        lambda env, cams, w, h, depth=False: asked.append((w, h, depth)) or {})
    sess.observe(ObsSpec(width=C.OBS_RESOLUTION, depth=True))
    assert asked == [(C.OBS_RESOLUTION, C.OBS_RESOLUTION, True)]


def test_depth_obeys_the_published_camera_set(tmp_path, monkeypatch):
    """`cameras=()` means no camera payload AT ALL. A depth map that survived it would
    be a viewpoint reaching the agent past the filter that exists to stop exactly that."""
    sess, _ = make_session(
        tmp_path, name="depthcams",
        factory=lambda task, split="pretrain", scene=None:
            SpyEnv(success_after=-1, split=split))
    sess.reset()
    monkeypatch.setattr(
        "harness.env.render_frames",
        lambda env, cams, w, h, depth=False:
            {f"{c}_depth": [[0.0]] for c in cams} if depth else {})

    out = sess.observe(ObsSpec(cameras=(), depth=True))
    assert not [k for k in out["obs"] if k.endswith(("_image", "_depth"))]

    out = sess.observe(ObsSpec(cameras=("robot0_eye_in_hand",), width=256, depth=True))
    assert set(out["obs"]) & {"robot0_eye_in_hand_depth"}
    assert "robot0_agentview_left_depth" not in out["obs"]


def test_a_stale_depth_key_cannot_ride_through_the_filter(tmp_path):
    """`_depth` is dropped alongside `_image` when a spec is applied. Were it treated as
    proprioception, a depth map for an unrequested camera would pass straight through --
    the same hole `allowed_cameras` exists to close on the colour side."""
    from harness import env as ENV

    obs = {"robot0_proprio-state": [0.0],
           "robot0_agentview_left_image": "img",
           "robot0_agentview_left_depth": "range"}
    out, _ = ENV.apply_obs_spec(None, obs, ObsSpec(cameras=()),
                                native=C.OBS_RESOLUTION,
                                ceiling=C.OBS_MAX_RESOLUTION)
    assert out == {"robot0_proprio-state": [0.0]}


def test_depth_crosses_the_wire_only_when_set():
    """Omitted when false, so a peer that predates range decodes these messages
    unchanged -- and `depth` survives the round trip when it is asked for."""
    assert ObsSpec(width=8).as_wire() == {"width": 8}
    assert ObsSpec(width=8, depth=True).as_wire() == {"width": 8, "depth": True}
    assert ObsSpec.from_wire({"width": 8}).depth is False
    assert ObsSpec.from_wire({"width": 8, "depth": True}).depth is True


def test_metric_depth_converts_the_buffer_to_metres(monkeypatch):
    """MuJoCo's raw buffer is [0, 1] and non-linear. Handing it over unconverted would
    be handing over a number that LOOKS like range and is not one, so the inverse -- which
    needs the model's near/far planes -- happens on the harness's side."""
    import numpy as np

    from harness import env as ENV

    class FakeSim:
        class model:
            class stat:
                extent = 2.0

            class vis:
                class map:
                    znear = 0.05      # -> near = 0.1 m
                    zfar = 5.0        # -> far  = 10.0 m

    near, far = 0.1, 10.0
    buffer = np.array([[0.0, 1.0]], dtype="float32")
    out = ENV.metric_depth(FakeSim, buffer)

    expected = near / (1.0 - buffer * (1.0 - near / far))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-6)
    # The endpoints are the planes themselves: 0 -> near, 1 -> far.
    assert out[0, 0] == pytest.approx(near)
    assert out[0, 1] == pytest.approx(far)

    # The line above went through robosuite's own get_real_depth_map, which owns this
    # formula. The local copy is only a fallback for running off-simulator, so it is
    # worth proving the two agree rather than trusting that they drifted together:
    # a fallback nobody compares is a second formula, not a fallback.
    import builtins

    real_import = builtins.__import__

    def no_robosuite(name, *args, **kwargs):
        if name == "robosuite.utils.camera_utils":
            raise ImportError("robosuite is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_robosuite)
    fallback = ENV.metric_depth(FakeSim, buffer)
    assert fallback.dtype == np.float32
    np.testing.assert_allclose(fallback, out, rtol=1e-6)
