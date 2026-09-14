import subprocess

import pytest

from rlebench import lint


@pytest.fixture
def checkout(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "tasks/task08").mkdir(parents=True)
    (tmp_path / "tasks/task08/build_assets.py").write_text(
        "def solution_controller_source():\n    return 'value = 1\\n'\n")
    return tmp_path


def test_lint_checks_tracked_and_new_sources_but_excludes_generated(checkout, capsys):
    for name in (
        "tasks/task01/harness/tracked.py", "tests/new source.py",
        "tasks/task08/solution/controller.py",
        "tasks/task08/solution/payload/controller.py",
    ):
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("missing_name\n")
    (checkout / ".gitignore").write_text("tasks/task08/solution/payload/\n")
    subprocess.run(["git", "add", "tasks/task01"], cwd=checkout, check=True)

    assert lint.check(checkout) == 1
    output = capsys.readouterr().out
    assert "tracked.py:1:1: undefined name" in output
    assert "new source.py:1:1: undefined name" in output
    assert output.count("undefined name") == 2

    (checkout / "tasks/task01/harness/tracked.py").write_text("value = 1\n")
    (checkout / "tests/new source.py").write_text("value = 1\n")
    assert lint.check(checkout) == 0


def test_lint_rejects_errors_in_assembled_controller(checkout, capsys):
    (checkout / "tasks/task08/build_assets.py").write_text(
        "def solution_controller_source():\n    return 'missing_helper()\\n'\n")
    assert lint.check(checkout) == 1
    assert "controller.py (assembled):1:1: undefined name 'missing_helper'" in capsys.readouterr().out


def test_lint_rejects_controller_syntax_errors(checkout, capsys):
    (checkout / "tasks/task08/build_assets.py").write_text(
        "def solution_controller_source():\n    return 'def broken(\\n'\n")
    assert lint.check(checkout) == 1
    assert "(assembled)" in capsys.readouterr().err
