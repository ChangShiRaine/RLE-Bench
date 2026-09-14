"""A timed-out agent must not contaminate the next evaluation trial."""

import signal
import time
from unittest.mock import MagicMock

import pytest

from harness import control


@pytest.fixture
def boundary(monkeypatch):
    cleanup = MagicMock()
    monkeypatch.setattr(control, "stop_agent_processes", cleanup)
    client = MagicMock()
    monkeypatch.setattr(control, "ControlClient", lambda _: client)
    return cleanup, client.__enter__.return_value


@pytest.mark.parametrize("failure", [None, RuntimeError("daemon unavailable")])
def test_cleanup_precedes_handoff(boundary, failure):
    cleanup, client = boundary

    def advance():
        cleanup.assert_called_once()
        if failure:
            raise failure
        return {"trials_done": 1, "total_trials": 5}

    client.next_trial.side_effect = advance
    assert control.main(["next-trial"]) == int(failure is not None)


def test_stuck_handoff_exits(boundary, monkeypatch):
    _, client = boundary
    client.next_trial.side_effect = lambda: time.sleep(5)
    alarm = signal.alarm
    monkeypatch.setattr(control.signal, "alarm", lambda n: alarm(1 if n else 0))
    assert control.main(["next-trial"]) == 1


def test_cleanup_failure_never_advances(boundary):
    cleanup, client = boundary
    cleanup.side_effect = RuntimeError("agent survived")
    assert control.main(["next-trial"]) == 1
    client.next_trial.assert_not_called()

