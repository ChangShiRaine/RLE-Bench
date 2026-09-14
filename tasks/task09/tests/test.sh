#!/bin/bash
# Harbor verifier entry point; runs in the separate verifier container and is
# the only writer of /logs/verifier/reward.json.
set -u

mkdir -p /logs/verifier

export PYTHONPATH=/tests
# software renderer for the post-eval device renders (physics never touches GL)
export MUJOCO_GL=osmesa
python3 /tests/score_task.py
status=$?

# score_task.py always writes reward.json itself, including on scoring
# errors; this fallback only covers a catastrophic interpreter failure.
if [ ! -f /logs/verifier/reward.json ]; then
    echo '{"reward": 0.0, "harness_crash": 1}' > /logs/verifier/reward.json
fi

exit $status
