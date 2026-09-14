"""Calibration candidates are complete, isolated, and published only on success."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from dev import calibrate
from harness import config


@pytest.fixture
def values():
    return {key: getattr(config, key) for key in calibrate.THRESHOLD_NAMES}


@pytest.fixture
def measurements():
    seeds = (*config.HIDDEN_SEEDS, *config.DESIGN_SEEDS)
    scenarios = {
        sid: {str(seed): dict(
            scenario=sid, tip=False, liftoff_events=[], min_fasm=60., min_fasm_robust=60.,
            max_tilt_deg=0., twist_cmd=[[1., 0., 0.]], twist_meas=[[.9, 0., 0.]])
              for seed in seeds}
        for sid in ("A", "B", "C", "D1", "D2", "E")}
    record = dict(
        model_info=dict(ok=True, reach=1., reach_floor=.9, reach_beyond=.5,
                        battery=dict(ok=True), arm=dict(ok=True), actuators=dict(ok=True),
                        resource_constraints=dict(score=1.), design_efficiency=dict(score=.2)),
        battery=dict(validity=dict(ok=True, loads=True, structure=dict(ok=True),
                                   inertia={"base": dict(ok=True)}, mecanum=dict(cases=[dict(ok=True)]),
                                   settle=dict(drift_xy=0., dz=0., tilt_deg=0., end_speed=0., max_penetration=0.)),
                     scenario_a_min_ssm={str(seed): .06 for seed in seeds}, scenarios=scenarios))
    return {name: deepcopy(record) for name in ("panda", "ur5e", "xarm7")}


def test_canonical_arm_reference_exists():
    assert calibrate.ARM_REFERENCE == calibrate.REPO / "assets/robots/franka_emika_panda/panda_nohand.xml"
    assert calibrate.ARM_REFERENCE.is_file()


def test_measurements_use_release_reference_controller_and_envelope(monkeypatch, measurements):
    import build_assets
    from harness.base_design import checkpoints, golden_controller
    from harness.sim import runner

    calls = []
    monkeypatch.setattr(build_assets, "ensure_reference_robot", lambda: None)

    def analyze(robot, arm):
        assert Path(robot) == calibrate.TASK / "reference/robot.xml"
        return measurements[arm.name]["model_info"]

    def run(robot, *, controller, seeds, env, arm):
        assert Path(robot) == calibrate.TASK / "reference/robot.xml"
        assert controller is golden_controller.make_base_controller
        assert seeds == (*config.HIDDEN_SEEDS, *config.DESIGN_SEEDS)
        assert env.require_wheel_center_attachment
        calls.append(arm.name)
        return deepcopy(measurements[arm.name]["battery"])

    monkeypatch.setattr(checkpoints, "analyze_model", analyze)
    monkeypatch.setattr(runner, "run_all", run)
    assert calibrate.measure_golden() == measurements
    assert calls == ["panda", "ur5e", "xarm7"]


def test_derivation_uses_worst_arm_and_seed(measurements):
    measurements["ur5e"]["model_info"]["reach_beyond"] = .4
    measurements["xarm7"]["battery"]["scenario_a_min_ssm"]["307"] = .05
    measurements["panda"]["battery"]["scenarios"]["B"]["11"]["min_fasm_robust"] = 40.
    measurements["ur5e"]["battery"]["scenarios"]["D2"]["23"]["liftoff_events"] = [{"duration": .4}]
    assert calibrate.derive(measurements) == dict(
        REACH_BEYOND_MIN_M=.24, SSM_THRESHOLD_M=.03, SSM_SAT_M=.05, FASM_SAT=30.,
        D2_MAX_LIFTOFF_EVENTS=2, D2_MAX_LIFTOFF_DURATION_S=.6,
        TRACKING_RMS_TOL=.15, S5_TRACKING_GATE=.2)


def test_successful_candidates_preserve_fixed_configuration(tmp_path, values, measurements):
    original = {key: value for key, value in vars(config).items() if key.isupper()}
    installed = calibrate.THRESHOLDS_PATH.read_bytes()
    output = tmp_path / "thresholds.py"
    calibrate.publish(values, measurements, output)
    first = output.read_bytes()
    calibrate.publish(values, measurements, output)
    assert output.read_bytes() == first
    assert {key: value for key, value in vars(config).items() if key.isupper()} == original
    assert calibrate.THRESHOLDS_PATH.read_bytes() == installed
    # A changed candidate must be visible in the worker, despite imported host constants.
    values["FASM_SAT"] = 120.
    with pytest.raises(subprocess.CalledProcessError):
        calibrate.publish(values, measurements, output)
    assert output.read_bytes() == first


@pytest.mark.parametrize("fault", ["tip", "validity", "static", "tracking", "missing_arm", "missing_seed", "nan"])
def test_failed_calibration_preserves_existing_file(tmp_path, monkeypatch, values, measurements, fault):
    output = tmp_path / "thresholds.py"
    output.write_text("existing configuration\n")
    battery = measurements["ur5e"]["battery"]
    if fault == "tip":
        battery["scenarios"]["A"]["101"]["tip"] = True
    elif fault == "validity":
        battery["validity"]["ok"] = False
    elif fault == "static":
        battery["scenario_a_min_ssm"]["101"] = -.1
    elif fault == "tracking":
        battery["scenarios"]["B"]["101"]["twist_meas"] = [[0., 0., 0.]]
    elif fault == "missing_arm":
        del measurements["ur5e"]
    elif fault == "missing_seed":
        del battery["scenarios"]["D1"]["37"]
    else:
        battery["scenarios"]["E"]["101"]["min_fasm_robust"] = float("nan")
    monkeypatch.setattr(calibrate, "measure_golden", lambda: measurements)
    monkeypatch.setattr(calibrate, "THRESHOLDS_PATH", output)
    assert calibrate.main(["--write"]) == 1
    assert output.read_text() == "existing configuration\n"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("fault", ["write", "replace"])
def test_io_failure_leaves_existing_file_untouched(tmp_path, monkeypatch, values, measurements, fault):
    output = tmp_path / "thresholds.py"
    output.write_text("existing configuration\n")

    def fail(*args):
        raise OSError("injected failure")

    monkeypatch.setattr(calibrate.os, "fsync" if fault == "write" else "replace", fail)
    with pytest.raises(OSError, match="injected failure"):
        calibrate.publish(values, measurements, output)
    assert output.read_text() == "existing configuration\n"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("name,value", [("FASM_SAT", float("nan")), ("SSM_SAT_M", .01),
                                       ("S5_TRACKING_GATE", 0.), ("D2_MAX_LIFTOFF_EVENTS", 1.5)])
def test_invalid_thresholds_rejected(values, name, value):
    values[name] = value
    with pytest.raises(ValueError):
        calibrate.validate_values(values)


def test_default_check_does_not_write(tmp_path, monkeypatch, measurements):
    output = tmp_path / "thresholds.py"
    output.write_text("untouched\n")
    monkeypatch.setattr(calibrate, "measure_golden", lambda: measurements)
    monkeypatch.setattr(calibrate, "THRESHOLDS_PATH", output)
    assert calibrate.main([]) == 0
    assert output.read_text() == "untouched\n"


@pytest.mark.parametrize("mode", ["--output", "--write"])
def test_explicit_output_publishes_a_validated_candidate(tmp_path, monkeypatch, measurements, mode):
    output = tmp_path / "thresholds.py"
    monkeypatch.setattr(calibrate, "measure_golden", lambda: measurements)
    if mode == "--write":
        monkeypatch.setattr(calibrate, "THRESHOLDS_PATH", output)
    fixed = calibrate.TASK / "harness/config.py"
    original = fixed.read_bytes()
    assert calibrate.main([mode] + ([str(output)] if mode == "--output" else [])) == 0
    assert "SSM_SAT_M = 0.06" in output.read_text()
    assert fixed.read_bytes() == original


def test_output_cannot_overwrite_fixed_configuration(monkeypatch):
    def unexpected_measurement():
        pytest.fail("invalid output must be rejected before running physics")

    monkeypatch.setattr(calibrate, "measure_golden", unexpected_measurement)
    for path in (calibrate.THRESHOLDS_PATH, calibrate.TASK / "harness/config.py"):
        original = path.read_bytes()
        assert calibrate.main(["--output", str(path)]) == 1
        assert path.read_bytes() == original


def test_candidate_configuration_retains_all_fixed_settings(values):
    released = json.loads((Path(__file__).parent / "fixtures/task08_release_config.json").read_text())["config"]
    assert set(values) < set(released)
    assert {key for key in vars(config) if key.isupper()} == set(released)
