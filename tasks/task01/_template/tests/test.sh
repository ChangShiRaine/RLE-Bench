#!/bin/bash
# Harbor uploads this to /tests and runs it in the AGENT'S OWN container, as root.
#
# NOTHING SENSITIVE LIVES HERE, on purpose. This file is world-readable inside a
# container the agent also runs in, so it names paths and never numbers: the scorer,
# the thresholds and the evaluation seeds all sit in /opt/private (root:root 0700) and
# the ledger in /var/lib/rlebench (root:root 0700). The agent can read this script and
# learn nothing it did not already know.
#
# The explicit PYTHONPATH is the point: the scorer must be imported from the root-only
# tree, never from anywhere the agent's uid can write.
set -uo pipefail

mkdir -p /logs/verifier
PYTHONPATH=/opt/private /usr/local/bin/python -m harness.verify_main
status=$?

if [ ! -f /logs/verifier/reward.json ]; then
    # The scorer died before writing. Emit an explicit zero rather than leaving Harbor
    # to guess. Flat and numeric: Harbor parses this into dict[str, float | int], and a
    # string or bool here would fail validation -- turning a bad run into a lost one.
    # The reason goes next door, where non-numeric fields are allowed.
    printf '{"reward": 0.0, "success_rate": 0.0, "component_outcome": 0.0, "component_efficiency": 0.0}\n' \
        > /logs/verifier/reward.json
    printf '{"ledger_ok": false, "ledger_reason": "scorer did not produce a reward file"}\n' \
        > /logs/verifier/diagnosis.json
fi
exit "${status}"
