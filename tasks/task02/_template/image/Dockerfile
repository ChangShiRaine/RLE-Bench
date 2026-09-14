# Agent container for task02. Derives from the shared RoboCasa simulator image
# (sim/robocasa); the asset dataset arrives as a read-only mount (see
# docker-compose.yaml), not as image content.
#
# The simulator commits are pinned in sim/robocasa/pins.env. Bumping them changes
# what task02 measures -- the stage functions are hand-written mirrors of RoboCasa source
# -- so re-run tests/test_toolsmith_stages.py after any bump.
#
# THE ISOLATION IS THE TASK, and it protects something different from task01's. Task01
# hides how much a run cost. Task02 hides WHICH TASK IT IS GRADED ON: an agent that read
# the held-out name would simply develop against it, and there would be no transfer left
# to measure. RoboCasa is an importable library, so a readable install would also let the
# agent build its own env and interact off-meter. Hence:
#
#   /opt/src            robosuite + robocasa      root:root 0700  (agent CANNOT read)
#   /opt/private        daemon, config, stages    root:root 0700  (agent CANNOT read)
#   /opt/rlebench      client + protocol only    world-readable
#   /var/lib/rlebench  the cost ledger           root:root 0700
#   the Unix socket     mode 0666 by design -- reaching it only spends YOUR budget
#
# The asset mount may stay world-readable: assets are inert without the robocasa code,
# so locking the code is what enforces both metering and the split.
#
# NO GROUND TRUTH here (invariant #2): no evaluation task name, no stage functions, no
# scorer. tasks/task02/build_assets.py stages the two payload trees and asserts that the
# agent-visible one contains no scoring module, no import of one, and no evaluation task
# name anywhere in its text.
#
# THE SAME IMAGE SERVES EVERY GROUP, and every agent in a run -- one development agent
# and one per graded trial, each a fresh Harbor step in this one container. /workspace is what they share, and
# the only thing they share: Harbor empties /logs/agent between non-resumed steps.
#
# Build the simulator image first:  sim/robocasa/robocasa.sh image
# Then this one:                    make task02
ARG SIM_IMAGE=rlebench-robocasa-sim:1.0.1
FROM ${SIM_IMAGE}

# 1000 is the default, and the right one for anything you distribute: the image stays
# portable and reproducible.
#
# `RLEBENCH_DEBUG=1 make task02` overrides it to the uid of whoever will read the
# results, for ONE reason -- /logs/agent is a bind mount and agent CLIs write their
# session trace mode 0600, so ownership is the only thing that makes it readable while
# the run is still going.
ARG AGENT_UID=1000
ARG AGENT_USER=agent

# --non-unique: an arbitrary host uid may already belong to a base-image account, and
# colliding is harmless here (permissions are enforced by uid, not by name) whereas
# failing the build would not be.
# The uid-0 guard is not paranoia: every bit of the isolation above is uid-based.
RUN test "${AGENT_UID}" -ne 0 \
    || (echo "AGENT_UID must not be 0 -- the agent must be unprivileged" >&2; exit 1) \
    && useradd --create-home --non-unique --uid ${AGENT_UID} --shell /bin/bash ${AGENT_USER}

# --- generic agent-harness dependencies ----------------------------------------
# Harbor's "installed" agents bootstrap themselves inside THIS container at setup.
# claude-code, for example, runs as root:
#   apt-get update && apt-get install -y curl procps
# and then fetches its own release with curl. Without these baked in, setup dies with
# "exit 100 ... Unable to locate package curl".
#
# Only AGENT-NEUTRAL packages belong here. The image must not know or care which agent is
# run against it, so the agent's own binary is deliberately NOT installed -- that is the
# runner's business, allowed per-run with --allow-agent-host.
#
# The apt lists are deliberately RETAINED (contrast the usual `rm -rf
# /var/lib/apt/lists/*`): python:3.11-slim ships without them, and apt cannot resolve
# even an already-installed package with no lists present -- which is exactly the
# "Unable to locate package curl" above. With the lists cached, `apt-get update` fails to
# reach the mirrors but exits 0 with warnings, and the install step then finds everything
# already at its newest version and succeeds offline. Costs ~40 MB.
#
# Placed directly after useradd so this layer is not invalidated by payload changes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates procps

# Agent-visible: just enough to talk to the daemon and write a controller.
COPY payload_agent/harness /opt/rlebench/harness
# Trusted side: daemon, metered session, evaluation split, stage functions, ledger.
COPY payload_private/harness /opt/private/harness
# rlebench.core.media, which the daemon records evaluation-trial video with.
COPY payload_private/rlebench /opt/private/rlebench

# Lock the simulator and the scoring side away from the agent's uid. 0700 on a DIRECTORY
# also blocks traversal, so nothing underneath is reachable either. robosuite/robocasa
# are editable installs, so their code lives in /opt/src and a .pth in site-packages
# points at it -- locking /opt/src is what makes `import robocasa` fail for the agent.
#
# /opt/private/debug is the mount point for the live debug tree (docker-compose.yaml). It
# lives here, under a 0700 root directory, precisely so the AGENT CANNOT TRAVERSE TO IT:
# the debug output carries object poses and the per-trial stage vectors, and ownership
# cannot protect it because AGENT_UID may be the host user's own uid. Directory
# permissions on the parent are what separate them.
RUN mkdir -p /opt/private/debug
RUN chmod 700 /opt/src /opt/private \
    && chown -R root:root /opt/src /opt/private \
    && mkdir -p /var/lib/rlebench /run/rlebench \
    && chmod 700 /var/lib/rlebench \
    && chmod 755 /run/rlebench \
    && chmod -R a+rX /opt/rlebench

# NOT `harness`, and the name is load-bearing. WORKDIR is /workspace, and Harbor execs
# collect hooks and the verifier from it; `python -m` puts the CWD ahead of PYTHONPATH, so
# an agent-authored /workspace/harness/__init__.py would shadow the root-only
# /opt/private/harness -- the control plane AND the scorer -- and every hook would die
# while Harbor recorded a completed zero.
#
# The agent's interpreter sees only the client package, plus its OWN harness directory --
# so `import controllers.reach` works from any working directory, in any step, without
# every evaluation agent having to rediscover the path. /workspace/agent_harness is created
# below and owned by the agent; it is the entire handoff channel between the development
# agent and the evaluation agents that follow.
ENV PYTHONPATH=/opt/rlebench:/workspace/agent_harness
ENV RLEBENCH_TOOLSMITH_SOCKET=/run/rlebench/toolsmith.sock

COPY entrypoint.sh check_isolation.sh seal_ledger.sh /opt/
RUN chmod 755 /opt/entrypoint.sh /opt/check_isolation.sh \
    && chmod 700 /opt/seal_ledger.sh

WORKDIR /workspace
RUN mkdir -p /workspace/agent_harness/controllers /logs/artifacts \
    && chown -R ${AGENT_USER}:${AGENT_USER} /workspace /logs/artifacts

# Harbor runs each step's agent phase as [agent] user = "agent"; the entrypoint starts
# the metering daemon as root first, and that daemon outlives every one of them.
ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["sleep", "infinity"]
