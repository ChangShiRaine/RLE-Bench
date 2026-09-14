#!/bin/sh
# Host-safety check. Run INSIDE the agent container AS THE AGENT USER:
#
#   docker run --rm -u agent --entrypoint /opt/check_safety.sh \
#       -v "$ROBOCASA_ASSET_DIR:/opt/src/robocasa/robocasa/models/assets:ro" \
#       rlebench-task01-l1-agent:dev
#
# Add --self-test first (proves the probe can detect a writable directory, so a PASS
# is not simply a broken check), and repeat with -u root --mount-only to confirm the
# read-only mount is the last barrier standing.
#
# The question this answers: the agent runs arbitrary code in this container, so what
# can it destroy OUTSIDE it? Bind mounts are the only path from container to host, and
# the RoboCasa asset tree is a large host directory that would be painful to lose.
#
# NON-DESTRUCTIVE BY CONSTRUCTION. It never issues rm/truncate against real host data.
# The only write it attempts is creating a NEW probe file, which it then removes -- so
# the worst case if a check "succeeds unexpectedly" is a stray empty file, never lost
# data. Proving a mount read-only does not require deleting something from it: on a
# read-only mount the kernel rejects unlink/rename/truncate for every uid, root
# included, so "cannot create" plus the ro mount flag settles it.
#
# To prove the DETECTOR works, run --self-test, which builds a throwaway writable
# directory and asserts the same probes correctly report it as unsafe.
#
# Exits non-zero if any host data is reachable for writing.
set -u

fail=0
ok()     { echo "  ok      $1"; }
unsafe() { echo "  UNSAFE  $1" >&2; fail=1; }

ASSETS="${ROBOCASA_ASSETS:-/opt/src/robocasa/robocasa/models/assets}"

# Is DIR writable by this process? Creates and removes one new file; touches nothing
# that already exists. Returns 0 when writable.
probe_writable() {
    _p="$1/.rlebench_safety_probe.$$"
    if touch "$_p" 2>/dev/null; then
        rm -f "$_p" 2>/dev/null
        return 0
    fi
    return 1
}

# Does /proc/mounts carry the `ro` flag for this exact mount point?
mount_is_ro() {
    awk -v t="$1" '$2 == t { split($4, o, ","); for (i in o) if (o[i] == "ro") { print "ro"; exit } }' \
        /proc/mounts 2>/dev/null | grep -q ro
}

if [ "${1:-}" = "--self-test" ]; then
    # Confirm the probe actually detects a writable directory, so a PASS below means
    # "read-only" and not "the probe is broken".
    d=$(mktemp -d)
    if probe_writable "$d"; then
        echo "[safety] self-test ok: probe detects a writable directory"
        rmdir "$d"; exit 0
    fi
    echo "[safety] SELF-TEST FAILED: probe cannot detect a writable directory" >&2
    rmdir "$d"; exit 2
fi

# Root mode exists because the agent cannot even traverse to the asset tree
# (/opt/src is root-only 0700), which would let the mount check silently skip. As
# root the permission barrier is gone and the ro mount flag is the ONLY thing left
# standing between arbitrary code and the host dataset -- so that is the case worth
# testing directly. Nothing in the task runs as root; this is a check on the mount.
ROOT_MODE=0
[ "${1:-}" = "--mount-only" ] && ROOT_MODE=1

if [ "$ROOT_MODE" -eq 1 ]; then
    echo "[safety] mount check as uid=$(id -u) (permissions bypassed; testing the mount itself)"
else
    echo "[safety] uid=$(id -u) user=$(id -un)"
    if [ "$(id -u)" = "0" ]; then
        unsafe "running as root -- the agent phase must use an unprivileged user"
    else
        ok "running unprivileged"
    fi
fi

echo "[safety] the host asset dataset must be unwritable"
# /proc/mounts is world-readable even when the path itself cannot be traversed, so
# this distinguishes "no mount configured" from "mounted but out of reach".
if awk -v t="$ASSETS" '$2 == t {found=1} END {exit !found}' /proc/mounts 2>/dev/null; then
    mounted=1
else
    mounted=0
fi

if [ "$mounted" -eq 0 ]; then
    echo "  skip    no asset mount at $ASSETS (run with the bind to check it)"
elif [ ! -d "$ASSETS" ]; then
    # Mounted, but an ancestor directory denies traversal. That is STRONGER than a
    # read-only mount: the agent cannot open the path at all. Do not call it a skip.
    ok "asset mount present but unreachable for this uid (ancestor is 0700)"
    if mount_is_ro "$ASSETS"; then
        ok "and mounted ro, so it stays unwritable if those permissions ever loosen"
    else
        unsafe "mounted WITHOUT ro -- only directory permissions protect host data"
    fi
else
    if probe_writable "$ASSETS"; then
        unsafe "the asset mount accepts new files -- host data is DELETABLE from here"
        echo "          fix the mount (read_only: true) before running any agent" >&2
    else
        ok "cannot create files in the asset mount"
    fi

    if mount_is_ro "$ASSETS"; then
        ok "mounted ro -- the kernel refuses unlink/truncate here for every uid"
    else
        # Not fatal alone: the path may sit under a ro parent mount, and the write
        # probe above is authoritative. Report rather than assert.
        echo "  note    no ro flag on $ASSETS itself; relying on the write probe"
    fi

    # Read access must still work, or the simulator cannot load its assets.
    if [ -n "$(ls -A "$ASSETS" 2>/dev/null)" ]; then
        ok "asset tree is readable and non-empty"
    else
        unsafe "asset tree is empty or unreadable -- the simulator will not run"
    fi
fi

echo "[safety] no route to the container runtime"
sock_found=0
for sock in /var/run/docker.sock /run/docker.sock \
            /var/run/containerd/containerd.sock /run/containerd/containerd.sock; do
    if [ -S "$sock" ]; then
        unsafe "container runtime socket exposed: $sock -- this is full host root"
        sock_found=1
    fi
done
[ "$sock_found" -eq 0 ] && ok "no docker/containerd socket in the container"

echo "[safety] no host namespaces or privileged capabilities"
if [ -r /proc/1/cgroup ] && grep -qE 'docker|kubepods|containerd' /proc/1/cgroup 2>/dev/null; then
    ok "pid 1 is containerised"
elif [ -f /.dockerenv ]; then
    ok "pid 1 is containerised (/.dockerenv)"
else
    echo "  note    could not confirm containerisation from inside"
fi
if [ -r /proc/self/status ]; then
    caps=$(awk '/^CapEff/ {print $2}' /proc/self/status)
    case "$caps" in
        0000000000000000) ok "no effective capabilities" ;;
        *) # SYS_ADMIN (bit 21) would allow remounting the asset tree rw.
           if [ "$(( 0x$caps & 0x200000 ))" -ne 0 ]; then
               unsafe "CAP_SYS_ADMIN held (CapEff=$caps) -- a ro mount can be remounted rw"
           else
               ok "no CAP_SYS_ADMIN (CapEff=$caps)"
           fi ;;
    esac
fi

if [ "$ROOT_MODE" -eq 0 ]; then
    echo "[safety] writable paths visible to the agent"
    # State the blast radius explicitly instead of asserting a guess about it.
    # Everything listed WRITABLE is destroyable by the agent, so each one must be
    # container-internal or trial-scoped -- never a durable host tree.
    for d in /workspace /logs /logs/artifacts /home/agent /tmp "$ASSETS" /opt/src /opt/private; do
        [ -d "$d" ] || continue
        if probe_writable "$d"; then
            echo "  WRITABLE  $d"
        else
            echo "  read-only $d"
        fi
    done
fi

if [ "${fail}" -eq 0 ]; then
    echo "[safety] PASS -- no host dataset is writable or deletable from this container"
else
    echo "[safety] FAIL" >&2
fi
exit "${fail}"
