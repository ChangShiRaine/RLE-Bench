#!/bin/sh
# Finalize the simulator-owned result before collecting artifacts.
set -u

PIDFILE=/run/rlebench/daemon.pid
# The AUTHORITATIVE ledger -- root:root 0700, the file the verifier scores. Deliberately
# not the /logs/artifacts export: that copy is for humans and lives in a directory Harbor
# makes agent-writable.
LEDGER="${RLEBENCH_LEDGER:-/var/lib/rlebench/cost.jsonl}"

if [ ! -f "$PIDFILE" ]; then
    echo "[seal] no pidfile at $PIDFILE -- daemon never started?" >&2
    exit 1
fi

# Has the daemon finished? `kill -0` is not enough on its own: the daemon is a child
# of PID 1, PID 1 here does not reap, so an exited daemon lingers as a ZOMBIE -- and
# kill -0 succeeds on a zombie. Waiting on that alone always burns the full timeout.
daemon_finished() {
    kill -0 "$1" 2>/dev/null || return 0              # reaped or never existed
    # Field 3 of /proc/pid/stat is the state; Z means it exited and is awaiting reap.
    [ "$(awk '{print $3}' /proc/"$1"/stat 2>/dev/null)" = "Z" ]
}

pid=$(cat "$PIDFILE")
if ! daemon_finished "$pid"; then
    echo "[seal] signalling daemon pid=$pid"
    kill -TERM "$pid" 2>/dev/null
    # Wait for it to actually finish, so the export is complete before Harbor reads
    # it. Bounded: a hung daemon must not stall collection.
    i=0
    while ! daemon_finished "$pid"; do
        i=$((i + 1))
        if [ "$i" -gt 300 ]; then
            echo "[seal] daemon still alive after 30s; collecting whatever exists" >&2
            break
        fi
        sleep 0.1
    done
    echo "[seal] daemon finished"
else
    # Already exited -- its SIGTERM handler (or a clean close) will have sealed.
    echo "[seal] daemon pid=$pid not running; assuming it already sealed"
fi

if [ -f "$LEDGER" ]; then
    # The seal is the last record, so its presence is what the scorer's gate checks.
    if tail -1 "$LEDGER" | grep -q '"kind":"seal"'; then
        echo "[seal] ok: $LEDGER is sealed"
        exit 0
    fi
    echo "[seal] WARNING: $LEDGER exists but has no trailing seal record" >&2
    exit 1
fi

echo "[seal] FAILED: no ledger at $LEDGER" >&2
exit 1
