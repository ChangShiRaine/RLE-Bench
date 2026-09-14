# Agent container for task01. Derives from the shared RoboCasa simulator image
# (sim/robocasa); the asset dataset arrives as a read-only mount (see
# docker-compose.yaml), not as image content.
#
# THE ISOLATION IS THE TASK. Task01 scores how many env.step calls and evaluation
# attempts the agent spent, so those counts have to be unforgeable. RoboCasa is an
# importable library, so a readable install would let the agent build its own env and
# interact off-meter, making the score meaningless. Hence:
#
#   /opt/src            robosuite + robocasa      root:root 0700  (agent CANNOT read)
#   /opt/private        daemon, session, ledger,  root:root 0700  (agent CANNOT read)
#                       scorer, privileged.py
#   /opt/rlebench      client + protocol, and    world-readable
#                       the skill library at L2 and L3
#   /var/lib/rlebench  the cost ledger           root:root 0700
#   the Unix socket     mode 0666 by design -- reaching it only spends YOUR budget
#
# The asset mount may stay world-readable: assets are inert without the robocasa
# code, so locking the code is what enforces metering.
#
# NO GROUND TRUTH here (invariant #2): no thresholds, no evaluation seeds, no scorer.
# tasks/task01/build_assets.py stages the two payload trees and asserts that the
# agent-visible one contains no scoring module.
#
# ONE IMAGE PER HARNESS LEVEL. The RoboCasa task reaches the daemon as a runtime string,
# so the subtask directories at a level share this image and differ only in
# RLEBENCH_TASK. The level cannot be a runtime knob: it decides which modules land in
# the agent-readable tree, which is a fact about the file system, settled here.
#
# Build the simulator image first:  sim/robocasa/robocasa.sh image
# Then these:                       make task01            (all three levels)
#                                   make task01-L2         (one)
# THE PERCEPTION MODELS ARE A STAGE, NOT A SECOND IMAGE. `MODELS` selects it: L1 takes
# `models-off` and never installs torch or copies a weight, L2 and L3 take `models-on`.
# BuildKit does not build an unselected stage, so the control condition costs nothing for
# a harness it does not have -- and an L1-only workflow needs neither the ~6 GB of models
# nor the HuggingFace token they are gated behind.
#
# An ARG used in a FROM must be declared before the first one, hence the position.
ARG MODELS=models-off
ARG SIM_IMAGE=rlebench-robocasa-sim:1.0.1
FROM ${SIM_IMAGE} AS agent-base

# Which harness this image ships. tasks/task01/build_levels.py has already staged the
# payloads accordingly; this records the decision so the image can be asked, and so
# check_isolation.sh can verify the tree matches the claim.
ARG LEVEL=L1
RUN case "${LEVEL}" in L1|L2|L3) ;; \
    *) echo "LEVEL must be L1, L2 or L3, got '${LEVEL}'" >&2; exit 1 ;; esac
# Baked in under its OWN name, never as RLEBENCH_LEVEL. task.toml sets that one from the
# directory the run was launched from, and entrypoint.sh refuses to start if the two
# disagree -- a run graded under a harness it did not have is invisible at every other
# layer.
ENV RLEBENCH_IMAGE_LEVEL=${LEVEL}

# 1000 keeps the image portable and reproducible. `RLEBENCH_DEBUG=1 make task01`
# overrides it to the uid of whoever will read the results, for one reason: /logs/agent
# is a bind mount and agent CLIs write their session trace mode 0600, so ownership is
# what makes it readable while the run is still going. Artifact archiving does not depend
# on it -- the task-level collect hook chmods a+rwX.
ARG AGENT_UID=1000
ARG AGENT_USER=agent

# --non-unique: an arbitrary host uid may already belong to a base-image account, and
# colliding is harmless (permissions are enforced by uid, not by name). The uid-0 guard
# is load-bearing: the whole metering guarantee is uid-based.
RUN test "${AGENT_UID}" -ne 0 \
    || (echo "AGENT_UID must not be 0 -- the agent must be unprivileged" >&2; exit 1) \
    && useradd --create-home --non-unique --uid ${AGENT_UID} --shell /bin/bash ${AGENT_USER}

# --- generic agent-harness dependencies ----------------------------------------
# Harbor's "installed" agents bootstrap themselves inside THIS container at setup.
# claude-code, for example, runs as root:
#   apt-get update && apt-get install -y curl procps
# and then fetches its own release with curl. The task is `no-network`, so without
# these baked in, setup dies with "exit 100 ... Unable to locate package curl".
#
# Only AGENT-NEUTRAL packages belong here. The image must not know or care which agent
# is run against it, so the agent's own binary is deliberately NOT installed -- that is
# the runner's business, allowed per-run with --allow-agent-host.
#
# The apt lists are deliberately RETAINED (contrast the usual `rm -rf
# /var/lib/apt/lists/*`): apt cannot resolve even an already-installed package with no
# lists present, which is exactly the failure above. With them cached, `apt-get update`
# fails to reach the mirrors but exits 0, and the install then succeeds offline. ~40 MB.
#
# Placed directly after useradd so this layer is not invalidated by payload changes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates procps


# ---------------------------------------------------------------------------
# THE MODEL STAGES. Only one is built.
# ---------------------------------------------------------------------------

# L1: the control condition. Nothing is added, so the image carries no torch, no weights
# and no /opt/perception for the entrypoint to find.
FROM agent-base AS models-off

# L2 and L3: SAM3 (text-prompted segmentation) and Contact-GraspNet (grasp proposals),
# at the commits sim/perception/pins.env pins.
#
# The vendored tree arrives as a NAMED BUILD CONTEXT rather than as image content copied
# into this directory: it is several GB, Docker cannot COPY from outside a build context,
# and staging it into two per-level contexts would duplicate it twice for nothing. The
# Makefile points `perception` at third_party/perception, which sim/perception/perception.sh
# populates.
FROM agent-base AS models-on
ARG TORCH_VERSION
ARG SAM3_SHA
ARG CGN_SHA
RUN test -n "$TORCH_VERSION" -a -n "$SAM3_SHA" -a -n "$CGN_SHA" \
    || (echo "model build args missing -- build with 'make task01', which reads sim/perception/pins.env" >&2; exit 1)

# Torch is deliberately absent from the simulator image (see its note on robocasa's
# --no-deps install, which prunes ~4 GB of policy-learning stack). It comes back here,
# in this stage only.
# The lock includes transitive and build dependencies. --no-deps preserves the
# compatible PyOpenGL pin instead of pyrender's obsolete 3.1.0 requirement.
COPY requirements.lock /opt/perception/requirements.lock
RUN pip install --no-cache-dir --no-deps --no-build-isolation \
        -r /opt/perception/requirements.lock

# pyrender imports pyglet's GL stack at module import; without a display that dies in
# xlib. The image is headless EGL, and this makes pyglet pick that backend.
ENV PYGLET_HEADLESS=1

COPY --from=perception sam3 /opt/perception/sam3
COPY --from=perception contact_graspnet_pytorch /opt/perception/contact_graspnet_pytorch
COPY --from=perception hf-cache /opt/perception/hf-cache
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/perception/sam3 \
    && pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/perception/contact_graspnet_pytorch

# Recorded so the image can be asked which models it carries rather than inferred.
ENV RLEBENCH_SAM3_SHA=${SAM3_SHA}
ENV RLEBENCH_CGN_SHA=${CGN_SHA}

# OFFLINE, and that is not a convenience: a weight that resolved differently between two
# runs would make their scores incomparable, and the cache copied above is the only place
# the models may load from.
ENV HF_HOME=/opt/perception/hf-cache
ENV HF_HUB_OFFLINE=1
ENV RLEBENCH_PERCEPTION_DIR=/opt/perception

# ---------------------------------------------------------------------------
# THE IMAGE. `MODELS` decided which of the two above it continues from.
# ---------------------------------------------------------------------------
FROM ${MODELS} AS final

# Agent-visible: just enough to talk to the daemon -- plus, at L2 and L3, the harness
# library. Which one this is was decided by build_assets.py when the payload was staged,
# so there is nothing conditional to express here.
COPY payload_agent/harness /opt/rlebench/harness
# Trusted side: daemon, metered session, ledger writer.
COPY payload_private/harness /opt/private/harness
# rlebench.core.media, which the daemon records evaluation-trial video with.
COPY payload_private/rlebench /opt/private/rlebench

# Lock the simulator and the scoring side away from the agent's uid. 0700 on a
# DIRECTORY also blocks traversal, so nothing underneath is reachable either.
# robosuite/robocasa are editable installs, so their code lives in /opt/src and a
# .pth in site-packages points at it -- locking /opt/src is what makes `import
# robocasa` fail for the agent.
# /opt/private/debug is the mount point for the live debug tree (docker-compose.yaml).
# It lives under a 0700 root directory precisely so the AGENT CANNOT TRAVERSE TO IT: the
# debug output carries object poses and the reward breakdown, and ownership cannot
# protect it when AGENT_UID is the host user's own uid.
RUN mkdir -p /opt/private/debug
# /opt/perception exists only at L2 and L3. Locked to root for the same reason /opt/src
# is: the agent gets a SERVICE with a contract on a socket, not a tree of modules it can
# import and monkeypatch. The `|| true` is what lets L1 build without it.
RUN chmod 700 /opt/perception 2>/dev/null && chown -R root:root /opt/perception || true
RUN chmod 700 /opt/src /opt/private \
    && chown -R root:root /opt/src /opt/private \
    && mkdir -p /var/lib/rlebench /run/rlebench \
    && chmod 700 /var/lib/rlebench \
    && chmod 755 /run/rlebench \
    && chmod -R a+rX /opt/rlebench

# The agent's interpreter sees only the client package. The daemon is started with
# PYTHONPATH=/opt/private/... by the entrypoint, as root.
ENV PYTHONPATH=/opt/rlebench
ENV RLEBENCH_SPEEDRUN_SOCKET=/run/rlebench/speedrun.sock
ENV RLEBENCH_PERCEPTION_SOCKET=/run/rlebench/perception.sock

# The skill manual, where the L2 and L3 instructions say it is. A copy rather than a path
# into the package, because the agent is told to READ it. Absent at L1, which ships no
# skills -- the `true` is what allows that rather than faking a manual.
RUN cp /opt/rlebench/harness/skills/MANUAL.md /opt/HARNESS_MANUAL.md \
        2>/dev/null && chmod 644 /opt/HARNESS_MANUAL.md || true

COPY entrypoint.sh check_isolation.sh check_safety.sh seal_ledger.sh /opt/
RUN chmod 755 /opt/entrypoint.sh /opt/check_isolation.sh /opt/check_safety.sh \
    && chmod 700 /opt/seal_ledger.sh

WORKDIR /workspace
RUN mkdir -p /workspace /logs/artifacts \
    && chown ${AGENT_USER}:${AGENT_USER} /workspace /logs/artifacts

# Harbor runs the agent phase as [agent] user = "agent"; the entrypoint starts the
# metering daemon as root first.
ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["sleep", "infinity"]
