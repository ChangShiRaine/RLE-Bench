#!/bin/sh
set -eu

socket="${RLEBENCH_TRAINING_SOCKET:-/run/rlebench/task06-training.sock}"
rm -f "${socket}" /run/rlebench/task06-ready

echo "[task06-entrypoint] starting private training renderer"
PYTHONPATH=/opt/private python -P -m harness.training_service \
    --socket "${socket}" &
daemon_pid=$!

i=0
while [ ! -S "${socket}" ]; do
    i=$((i + 1))
    if [ "${i}" -gt 600 ]; then
        echo "[task06-entrypoint] renderer did not become ready" >&2
        exit 1
    fi
    if ! kill -0 "${daemon_pid}" 2>/dev/null; then
        echo "[task06-entrypoint] renderer exited during startup" >&2
        wait "${daemon_pid}" || true
        exit 1
    fi
    sleep 0.1
done

: > /run/rlebench/task06-ready
exec "$@"
