#!/bin/bash
# Harbor verifier entry point (task11); only score_task.py writes rewards.
set -u
mkdir -p /logs/verifier
# Lock the verifier tree before any submitted code runs: sandboxed policies
# execute as uid 65534 and must not read the scorer or the hidden seeds.
chmod -R go-rwx /tests
PYTHONPATH=/tests python3 /tests/score_task.py
status=$?
if [ ! -f /logs/verifier/reward.json ]; then
    echo '{"reward": 0.0, "harness_error": 1}' > /logs/verifier/reward.json
    status=1
fi
exit $status
