#!/bin/bash
# Harbor verifier entry point. Runs in the SEPARATE verifier container.
# The only writer of /logs/verifier/reward.json.
set -u

mkdir -p /logs/verifier
chmod 700 /tests /logs/verifier
umask 022

export PYTHONPATH=/tests
# mujoco binds its GL backend at import time: select the software renderer
# up front so the post-eval robot renders work headless (libosmesa6 is in
# this image; physics itself never touches GL)
export MUJOCO_GL=osmesa
python3 /tests/score_task.py
status=$?

# score_task.py always writes reward.json itself, including on scoring
# errors; this fallback only covers a catastrophic interpreter failure.
if [ ! -f /logs/verifier/reward.json ]; then
    echo '{"reward": 0.0, "harness_crash": 1}' > /logs/verifier/reward.json
fi

exit $status
