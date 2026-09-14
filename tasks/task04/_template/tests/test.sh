#!/bin/bash
# Harbor verifier entry point (task04). The ONLY writer of
# /logs/verifier/reward.json.
set -u

mkdir -p /logs/verifier

# Lock the verifier context before scoring. The submission is an ONNX graph and
# never executes here, so this is defence in depth rather than a sandbox.
chmod -R o-rwx,g-rwx /tests || true

export PYTHONPATH=/tests
python3 /tests/score_task.py
status=$?

if [ ! -f /logs/verifier/reward.json ]; then
    echo '{"reward": 0.0, "harness_crash": 1}' > /logs/verifier/reward.json
fi

exit $status
