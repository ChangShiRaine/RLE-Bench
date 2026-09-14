#!/bin/sh
set -eu
test "${RLEBENCH_LEVEL:-L1}" = L1
test "${RLEBENCH_TASK:-RubikCube}" = RubikCube
mkdir -p /run/rlebench /var/lib/rlebench /workspace /logs/artifacts
chmod 700 /var/lib/rlebench
chmod 755 /run/rlebench
rm -f /run/rlebench/ready

PYTHONPATH=/opt/private /usr/local/bin/python -m harness.tabletop.pocket.recovery_daemon \
    --task RubikCube --eval-plan 1x1 --socket /run/rlebench/speedrun.sock \
    --ledger /var/lib/rlebench/cost.jsonl &
daemon_pid=$!
echo "$daemon_pid" > /run/rlebench/daemon.pid
trap 'kill -TERM "$daemon_pid" 2>/dev/null || true; wait "$daemon_pid" || true' TERM INT EXIT

attempt=0
while [ ! -S /run/rlebench/speedrun.sock ]; do
    kill -0 "$daemon_pid" 2>/dev/null || exit 1
    attempt=$((attempt + 1))
    test "$attempt" -lt 3000 || exit 1
    sleep 0.1
done
PYTHONPATH=/opt/private /usr/local/bin/python -c 'from harness.control import ControlClient; c=ControlClient(); c._request("end_development"); c.close()'
PYTHONPATH=/opt/private /usr/local/bin/python -m harness.control open-evaluation
touch /run/rlebench/ready
"$@" &
wait "$!"
