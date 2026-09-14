"""Source and staged task08 must retain the released verifier configuration."""
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
TASK = REPO / "tasks/task08"
SNAPSHOT = Path(__file__).parent / "fixtures/task08_release_config.json"


def _configuration(path):
    script = '''import json
from dataclasses import asdict
from harness import config
from harness.sim.scenarios import Envelope
print(json.dumps(dict(config={k: v for k, v in vars(config).items() if k.isupper()},
                     envelope=asdict(Envelope()))))
'''
    return json.loads(subprocess.check_output(
        [sys.executable, "-c", script], cwd=REPO,
        env=dict(os.environ, PYTHONPATH=str(path)), text=True))


def test_source_and_staged_configuration_match_release():
    expected = json.loads(SNAPSHOT.read_text())
    assert _configuration(TASK) == expected
    assert _configuration(TASK / "tests") == expected


def test_verifier_modules_are_staged_without_source_changes():
    source = TASK / "harness"
    staged = TASK / "tests/harness"
    files = {path.relative_to(source) for path in source.rglob("*.py")}
    assert files == {path.relative_to(staged) for path in staged.rglob("*.py")}
    for path in files:
        assert (source / path).read_bytes() == (staged / path).read_bytes(), path
