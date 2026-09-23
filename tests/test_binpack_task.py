"""Task11 staging, isolation and packaging: the agent tree holds no ground
truth, both images share one set of pins, and the Harbor contract holds."""
import re

from tasks.task11 import build_assets as build

TASK = build.TASK


def pins(dockerfile: str) -> dict:
    return dict(re.findall(r"([A-Za-z0-9_.-]+)==([^\s\\]+)", dockerfile))


def test_agent_tree_holds_no_ground_truth():
    build.main()
    agent = TASK / "environment" / "assets"
    names = {p.name for p in agent.rglob("*")}
    for private in ("golden.py", "baseline.py", "scorer.py",
                    "eval_seeds.json", "score_task.py", "calibrate.py"):
        assert private not in names
    build.scan(agent)                        # raises on any forbidden token
    public = (agent / "harness" / "config.py").read_text()
    for secret in ("W_UTIL", "UTIL_CAP", "GATE_CAP", "REF_SCORE"):
        assert secret not in public
    assert "DAMAGE_FORCE_N" in public


def test_verifier_tree_is_complete():
    build.main()
    tests = TASK / "tests"
    for rel in ("Dockerfile", "test.sh", "score_task.py", "harness/scorer.py",
                "harness/eval_seeds.json", "rlebench/core/media.py",
                "assets/franka_emika_panda/panda_nohand.xml"):
        assert (tests / rel).is_file(), rel
    assert "chmod -R go-rwx /tests" in (tests / "test.sh").read_text()


def test_images_share_pins_base_and_apt_snapshot():
    agent = (TASK / "environment" / "Dockerfile").read_text()
    verifier = (TASK / "tests" / "Dockerfile").read_text()
    assert agent == build.AGENT_DOCKERFILE and verifier == build.VERIFIER_DOCKERFILE
    assert pins(agent) == pins(verifier)
    assert {"mujoco", "numpy", "scipy", "torch"} <= pins(agent).keys()
    base = [line for line in agent.splitlines() if line.startswith("FROM")]
    assert base == [line for line in verifier.splitlines()
                    if line.startswith("FROM")]
    assert "@sha256:" in base[0]
    assert f"snapshot.debian.org/archive/debian/{build.SNAPSHOT}" in agent


def test_oracle_payload_is_the_baseline_policy():
    build.main()
    payload = TASK / "solution" / "payload" / "policy" / "policy.py"
    assert payload.read_text() == (build.SRC / "baseline.py").read_text()
    assert "def make_policy" in payload.read_text()


def test_task_toml_contract():
    toml = (TASK / "task.toml").read_text()
    assert toml.split("[task]")[0].strip().endswith(
        'artifacts = ["/logs/artifacts"]')
    assert 'environment_mode = "separate"' in toml
    assert toml.count('network_mode = "no-network"') == 3
    agent, verifier = toml.split("[verifier]", 1)
    assert "timeout_sec = 7200.0" in agent
    assert "timeout_sec = 10800.0" in verifier
    assert 'docker_image = "rlebench-task11-agent:dev"' in agent
    assert 'docker_image = "rlebench-task11-verifier:dev"' in verifier
