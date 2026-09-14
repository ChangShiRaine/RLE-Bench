#!/bin/sh
# Start the metering daemon as root, then hand control to the container command (which
# Harbor runs as the unprivileged agent user).
#
# The daemon must be running before any agent does anything, because it is the only route
# to the simulator. If it fails to start, fail the container loudly rather than leaving
# every agent in the run with a dead socket and no explanation.
set -eu

SOCKET="${RLEBENCH_TOOLSMITH_SOCKET:-/run/rlebench/toolsmith.sock}"
LEDGER="${RLEBENCH_LEDGER:-/var/lib/rlebench/cost.jsonl}"

# The asset dataset is a separate versioned mount; a stale or missing one changes the
# scenes silently, so refuse to start rather than produce a quietly wrong run.
python /opt/verify_assets.py

echo "[entrypoint] starting metering daemon"
# Package ROOT is /opt/private (the package itself is /opt/private/harness), and /opt
# carries verify_assets/rlebench_ro_assets. Pointing at the package directory instead of
# its parent yields "No module named 'harness'". PYTHONSAFEPATH=1 keeps the CWD off
# sys.path, so nothing under /workspace can answer this import.
PYTHONSAFEPATH=1 PYTHONPATH=/opt/private:/opt \
    python -m harness.daemon_main \
        --socket "${SOCKET}" \
        --ledger "${LEDGER}" &
DAEMON_PID=$!

# Record the pid so the seal hook can signal exactly this process. A pidfile rather than
# pgrep: a pattern match can find the hook's own shell and orphan the wait loop. Readable
# by the agent, which is harmless -- uid 1000 cannot signal a root process.
mkdir -p /run/rlebench
echo "${DAEMON_PID}" > /run/rlebench/daemon.pid

# Wait for the socket, so the agent never races a half-started daemon.
i=0
while [ ! -S "${SOCKET}" ]; do
    i=$((i + 1))
    if [ "${i}" -gt 900 ]; then
        echo "[entrypoint] daemon did not bind ${SOCKET} in 90s" >&2
        exit 1
    fi
    if ! kill -0 "${DAEMON_PID}" 2>/dev/null; then
        echo "[entrypoint] daemon exited before binding ${SOCKET}" >&2
        wait "${DAEMON_PID}" || true
        exit 1
    fi
    sleep 0.1
done
echo "[entrypoint] daemon ready on ${SOCKET}"

# NOTHING TO UNSET: no daemon-only value travels by environment, so there is no run-time
# channel to scrub.

exec "$@"
