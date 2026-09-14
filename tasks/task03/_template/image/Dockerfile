# Private simulator and public RGB/proprioception client for task03.
FROM python:3.11-slim AS sim-base

# build-essential + linux-libc-dev: robosuite depends on pynput -> evdev
# (compiles against linux/input.h) and qpsolvers[quadprog]. libegl1 is the GPU
# rendering path. curl/procps: Harbor's "installed" agents bootstrap themselves
# in this container. The apt lists are deliberately RETAINED: with them cached,
# an agent's offline `apt-get install` of an already-fetched package resolves.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl procps \
        build-essential linux-libc-dev \
        libgl1 libegl1 libosmesa6 libglib2.0-0

# Pin from sim/robocasa/pins.env; build through the Makefile.
ARG ROBOSUITE_SHA
RUN test -n "$ROBOSUITE_SHA" \
    || (echo "ROBOSUITE_SHA missing -- build with 'make task03'" >&2; exit 1)

RUN mkdir -p /opt/src \
    && git clone https://github.com/ARISE-Initiative/robosuite.git /opt/src/robosuite \
    && git -C /opt/src/robosuite checkout --quiet "$ROBOSUITE_SHA" \
    && rm -rf /opt/src/robosuite/.git
WORKDIR /

# The scientific core FIRST, pinned to the RoboCasa layer's exact versions
# (sim/robocasa/base/Dockerfile), so physics is bit-identical across the
# two families.
RUN pip install --no-cache-dir \
        "numpy==2.2.5" \
        "scipy==1.15.3" \
        "mujoco==3.3.1" \
        "numba==0.61.2" \
        "gymnasium==0.29.1"

RUN pip install --no-cache-dir -e /opt/src/robosuite \
    && pip install --no-cache-dir \
        "Pillow==12.3.0" \
        "imageio==2.37.4" \
        "imageio-ffmpeg==0.6.0" \
        "h5py==3.16.0" \
        "lxml==6.1.1" \
        "pyyaml==6.0.3" \
        "termcolor==3.3.0" \
        "tqdm==4.70.0"

RUN python -m robosuite.scripts.setup_macros || true

# GPU EGL rendering; see sim/robocasa/base/Dockerfile for why both the
# driver capability and the vendor ICD are required (missing either falls back
# to CPU llvmpipe SILENTLY).
ENV MUJOCO_GL=egl
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
       > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV PYTHONDONTWRITEBYTECODE=1

ARG AGENT_UID=1000
ARG AGENT_USER=agent
RUN test "${AGENT_UID}" -ne 0 \
    || (echo "AGENT_UID must not be 0 -- the agent must be unprivileged" >&2; exit 1) \
    && useradd --create-home --non-unique --uid ${AGENT_UID} --shell /bin/bash ${AGENT_USER}

COPY payload_agent/harness /opt/rlebench/harness
COPY payload_private/harness /opt/private/harness
# rlebench.core.media, which the daemon records evaluation-trial video with.
COPY payload_private/rlebench /opt/private/rlebench

RUN mkdir -p /opt/private/debug
RUN chmod 700 /opt/src /opt/private \
    && chown -R root:root /opt/src /opt/private \
    && mkdir -p /var/lib/rlebench /run/rlebench \
    && chmod 700 /var/lib/rlebench \
    && chmod 755 /run/rlebench \
    && chmod -R a+rX /opt/rlebench

ENV PYTHONPATH=/opt/rlebench
ENV RLEBENCH_SPEEDRUN_SOCKET=/run/rlebench/speedrun.sock

COPY entrypoint.sh check_isolation.sh check_safety.sh seal_ledger.sh /opt/
RUN chmod 755 /opt/entrypoint.sh /opt/check_isolation.sh /opt/check_safety.sh \
    && chmod 700 /opt/seal_ledger.sh

WORKDIR /workspace
RUN mkdir -p /workspace /logs/artifacts \
    && chown ${AGENT_USER}:${AGENT_USER} /workspace /logs/artifacts

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["sleep", "infinity"]
