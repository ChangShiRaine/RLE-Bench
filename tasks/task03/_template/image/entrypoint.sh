#!/bin/sh
set -eu
TASK="${RLEBENCH_TASK:?RLEBENCH_TASK must name a tabletop task}"
SOCKET="/run/rlebench/speedrun.sock"
mkdir -p /run/rlebench
rm -f /run/rlebench/ready
cd /
python -I -c 'import sys; sys.path.insert(0, "/opt/private"); from harness.tabletop.daemon_main import main; main()' \
    --task "$TASK" --socket "$SOCKET" --ledger /var/lib/rlebench/cost.jsonl &
DAEMON_PID=$!
echo "$DAEMON_PID" > /run/rlebench/daemon.pid
i=0
while [ ! -S "$SOCKET" ]; do
    kill -0 "$DAEMON_PID" 2>/dev/null || exit 1
    i=$((i + 1))
    [ "$i" -lt 3000 ] || exit 1
    sleep 0.1
done
: > /run/rlebench/ready
cd /workspace
exec "$@"
