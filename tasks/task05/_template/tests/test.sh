#!/bin/bash
set -u
mkdir -p /logs/verifier
export PYTHONPATH=/opt/nanovla-verifier
python3 /tests/score_task.py
status=$?
pkill -KILL -u 65534 2>/dev/null || true
if [ ! -f /logs/verifier/reward.json ]; then echo '{"reward": 0.0}' > /logs/verifier/reward.json; fi
exit $status
