#!/bin/bash
# Harbor uploads this to /tests and runs it in the AGENT'S OWN container, as root, once
# per step.
#
# NOTHING SENSITIVE LIVES HERE, on purpose. This file is world-readable inside a container
# the agent also runs in, so it names paths and never contents: the scorer, the stage
# functions and the evaluation split all sit in /opt/private (root:root 0700) and the
# ledger in /var/lib/rlebench (root:root 0700). The agent can read this script and learn
# nothing it did not already know.
#
# The explicit PYTHONPATH is the point: the scorer must be imported from the root-only
# tree, never from anywhere the agent's uid can write. PYTHONSAFEPATH=1 is what makes
# that hold: Harbor runs this from /workspace, and `python -m` would otherwise search the
# CWD -- which the agent owns -- before PYTHONPATH.
#
# Every step writes a CUMULATIVE reward read off the ledger as it then stands, and
# task.toml's `multi_step_reward_strategy = "final"` keeps the last one that ran. That is
# what makes an aborted chain score what it earned instead of nothing.
set -uo pipefail

mkdir -p /logs/verifier
PYTHONSAFEPATH=1 PYTHONPATH=/opt/private /usr/local/bin/python -m harness.verify_main
status=$?

if [ ! -f /logs/verifier/reward.json ]; then
    # The scorer died before writing. Emit an explicit zero rather than leaving Harbor to
    # guess. Flat and numeric: Harbor parses this into dict[str, float | int], and a string
    # or bool here fails validation -- recorded as a step exception, which aborts every
    # remaining step. The reason goes next door, where non-numeric fields are allowed.
    printf '{"reward": 0.0, "success_rate": 0.0}\n' > /logs/verifier/reward.json
    printf '{"ledger_ok": false, "ledger_reason": "scorer did not produce a reward file"}\n' \
        > /logs/verifier/diagnosis.json
fi
exit "${status}"
