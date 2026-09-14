#!/bin/sh
# Seal the ledger and export it, as ROOT, after the agent phase ends.
#
# Wired as a Harbor collect hook on the last evaluation step, which runs inside the agent
# container BEFORE artifacts are downloaded and before it is stopped. That ordering is the
# point: Harbor collects /logs/artifacts from the *running* container, and sealing must not
# depend on the agent remembering to call close(). It is the harness's job, not the agent's.
#
# Mechanism: SIGTERM the daemon, whose handler calls seal_if_open() -> close() +
# export_artifacts(). close() guards on an already-closed session, so this is idempotent.
#
# MUST BE WIRED ON THE LAST EVALUATION STEP AND NOWHERE ELSE. Sealing kills the daemon, and
# one daemon serves every step: on `develop` it would end the run before a trial was graded,
# and on any middle step it would lose the trials after it. The earlier steps do not need
# it -- task02's scorer reads an unsealed ledger happily (see scoring.py), which it must,
# since every verifier but the last runs before the seal exists.
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
