#!/bin/sh
# Start the metering daemon as root, then hand control to the container command
# (which Harbor runs as the unprivileged agent user).
#
# The daemon must be running before the agent does anything, because it is the only
# route to the simulator. If it fails to start, fail the container loudly rather than
# leaving the agent with a dead socket and no explanation.
set -eu

TASK="${RLEBENCH_TASK:?RLEBENCH_TASK must name a RoboCasa task}"

# THE IMAGE AND THE TASK MUST AGREE ON THE HARNESS. The level is a fact about this image
# -- which modules its build put in the agent-readable tree -- and task.toml declares the
# same thing from the directory the run was launched from. Nothing else cross-checks
# them, so an L2 directory pointed at the L1 image would run happily, hand the agent no
# skills, and be scored as an L2 result.
IMAGE_LEVEL="${RLEBENCH_IMAGE_LEVEL:-L1}"
LEVEL="${RLEBENCH_LEVEL:-${IMAGE_LEVEL}}"
if [ "${LEVEL}" != "${IMAGE_LEVEL}" ]; then
    echo "[entrypoint] task declares harness ${LEVEL} but this image ships ${IMAGE_LEVEL}." >&2
    echo "[entrypoint] Run tasks/task01/${IMAGE_LEVEL}/, or rebuild: make task01-${LEVEL}" >&2
    exit 1
fi
mkdir -p /run/rlebench
rm -f /run/rlebench/ready
SOCKET="${RLEBENCH_SPEEDRUN_SOCKET:-/run/rlebench/speedrun.sock}"
LEDGER="${RLEBENCH_LEDGER:-/var/lib/rlebench/cost.jsonl}"
PERCEPTION_SOCKET="${RLEBENCH_PERCEPTION_SOCKET:-/run/rlebench/perception.sock}"

# The asset dataset is a separate versioned mount; a stale or missing one changes the
# scenes silently, so refuse to start rather than produce a quietly wrong run.
python /opt/verify_assets.py

# BOTH SERVICES ARE LAUNCHED BEFORE EITHER IS WAITED FOR. They are independent, and both
# have to be up before the agent runs; starting them in series would add the model load
# time (~30-60 s) to the env warm-up (~40 s).
echo "[entrypoint] starting metering daemon for task=${TASK} harness=${LEVEL}"
# Package ROOT is /opt/private (the package itself is /opt/private/harness), and
# /opt carries verify_assets/rlebench_ro_assets. Pointing at the package directory
# instead of its parent yields "No module named 'harness'".
PYTHONPATH=/opt/private:/opt \
    python -m harness.daemon_main \
        --task "${TASK}" \
        --socket "${SOCKET}" \
        --ledger "${LEDGER}" &
DAEMON_PID=$!

# Record the pid so the seal hook can signal exactly this process. A pidfile rather
# than pgrep: a pattern match can find the hook's own shell and orphan the wait loop.
# Readable by the agent, which is harmless -- uid 1000 cannot signal a root process.
echo "${DAEMON_PID}" > /run/rlebench/daemon.pid

# Models take ~30-60 s to load, which is exactly why they are a service: the agent runs
# many short scripts and an in-process import would pay that on every one of them.
if [ -d /opt/perception ]; then
    echo "[entrypoint] starting perception service"
    PYTHONPATH=/opt/private:/opt \
        python -m harness.perception_service \
            --socket "${PERCEPTION_SOCKET}" &
    PERCEPTION_PID=$!
    echo "${PERCEPTION_PID}" > /run/rlebench/perception.pid
fi

# Wait for the socket, so the agent never races a half-started daemon. 300s: under
# the task's cpu limit the warm-up shares the cores with the perception model
# load, and the two together can push the first env build well past a minute.
i=0
while [ ! -S "${SOCKET}" ]; do
    i=$((i + 1))
    if [ "${i}" -gt 3000 ]; then
        echo "[entrypoint] daemon did not bind ${SOCKET} in 300s" >&2
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

# THE PERCEPTION SERVICE, at the levels that ship the harness library and nowhere else.
# Presence of the vendored tree is what decides: it is copied in by the L2/L3 image build
# and absent from L1's, so this needs no second source of truth about the level.
#
# It touches no simulator, holds no env and cannot advance an episode, so nothing it does
# is metered -- it is a second SERVICE, not a second channel to the first one.
if [ -d /opt/perception ]; then
    echo "[entrypoint] waiting for perception service"
    i=0
    while [ ! -S "${PERCEPTION_SOCKET}" ]; do
        i=$((i + 1))
        if [ "${i}" -gt 3000 ]; then
            echo "[entrypoint] perception service did not bind ${PERCEPTION_SOCKET} in 300s" >&2
            exit 1
        fi
        if ! kill -0 "${PERCEPTION_PID}" 2>/dev/null; then
            echo "[entrypoint] perception service exited before binding" >&2
            wait "${PERCEPTION_PID}" || true
            exit 1
        fi
        sleep 0.1
    done
    echo "[entrypoint] perception ready on ${PERCEPTION_SOCKET}"
fi

# THE READINESS MARKER, and the healthcheck tests THIS rather than a socket.
#
# At L2 and L3 the daemon binds before the perception service has loaded its models, so
# a socket test would let Harbor `docker exec` the agent into a container where
# `segment_by_text` raises for the next half minute. One marker written after everything
# is up also keeps the healthcheck from having to know which level it is running on.
: > /run/rlebench/ready
echo "[entrypoint] ready"

# NOTHING TO UNSET: no daemon-only value travels by environment, so there is no run-time
# channel to scrub.

exec "$@"
