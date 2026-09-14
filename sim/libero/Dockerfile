ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04@sha256:b3e7fba84d169f46939f00c25be7d016f712a8d651f4756d6a55e693d84d94f2
ARG HF_IMAGE=rlebench-task05-hf-base:dev

# The subtask's encoder bundle (sim/libero/Dockerfile.hf), copied into the
# agent image below and into each per-subtask verifier image.
FROM ${HF_IMAGE} AS hf

# ---------------------------------------------------------------- agent
# Development container: CUDA + torch + the HF loaders + the subtask's bundle.
# No simulator, no evaluator source; rollouts go through the evaluation
# service beside it. Built once per bundle (agent-base, agent-dinov2, agent-open).
FROM ${BASE_IMAGE} AS agent
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git procps python3 python3-dev python3-pip python3-venv util-linux \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/nanovla-venv
ENV PATH=/opt/nanovla-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY sim/libero/requirements-agent.txt /tmp/nanovla-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/nanovla-requirements.txt
COPY --from=hf /assets/hf /assets/hf
COPY tasks/task05/harness/runtime/ /workspace/nanovla/
COPY tasks/task05/harness/stage_submission.py /usr/local/bin/nanovla-stage
COPY tasks/task05/harness/finalize_submission.py /usr/local/sbin/nanovla-finalize
ENV HF_HOME=/assets/hf \
    HF_HUB_OFFLINE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/nanovla \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_OFFLINE=1
RUN useradd --create-home --uid 1000 --shell /bin/bash agent \
    && mkdir -p /assets/nanovla /run/nanovla-eval /logs/artifacts/submission \
        /logs/nanovla-handoff /workspace/runs \
    && chmod 0700 /logs/nanovla-handoff \
    && chmod 0555 /usr/local/bin/nanovla-stage \
    && chmod 0500 /usr/local/sbin/nanovla-finalize \
    && chmod -R a+rX /workspace/nanovla \
    && chown -R agent:agent /workspace /logs/artifacts
USER 1000:1000
WORKDIR /workspace/nanovla
CMD ["sleep", "infinity"]

# ------------------------------------------------------------- verifier
# Root-owned base for the per-subtask verifier images and for the evaluation
# service: the LIBERO and LIBERO-plus forks and the rollout evaluator, all
# root-only. The per-subtask verifier image adds the subtask's bundle and /tests.
FROM ${BASE_IMAGE} AS verifier
ARG LIBERO_SHA
ARG LIBERO_PLUS_SHA
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git imagemagick libegl1 libgl1 libglib2.0-0 \
        libglvnd0 libosmesa6 libsm6 libxext6 libxrandr2 libxrender1 \
        procps python3 python3-dev python3-pip python3-venv util-linux \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/nanovla-venv
ENV PATH=/opt/nanovla-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY sim/libero/requirements.txt /tmp/nanovla-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/nanovla-requirements.txt \
    && python3 -m pip install --no-cache-dir --no-deps robosuite==1.4.1
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO \
    && git -C /opt/LIBERO checkout --detach "${LIBERO_SHA}" \
    && git clone https://github.com/sylvestf/LIBERO-plus.git /opt/LIBERO-plus \
    && git -C /opt/LIBERO-plus checkout --detach "${LIBERO_PLUS_SHA}" \
    && python3 -m pip install --no-deps -e /opt/LIBERO \
    && rm -rf /opt/LIBERO/.git /opt/LIBERO-plus/.git
COPY tasks/task05/harness/runtime/ /workspace/nanovla/
COPY tasks/task05/harness/verifier/ /opt/nanovla-verifier/
COPY rlebench/__init__.py /opt/nanovla-verifier/rlebench/__init__.py
COPY rlebench/core/__init__.py rlebench/core/media.py /opt/nanovla-verifier/rlebench/core/
COPY tasks/task05/harness/config/libero.yaml /opt/nanovla-config/libero/config.yaml
COPY tasks/task05/harness/config/libero_plus.yaml /opt/nanovla-config/libero_plus/config.yaml
RUN LIBERO_CONFIG_PATH=/opt/nanovla-config/libero PYTHONPATH=/opt/LIBERO \
        python3 -c "from libero.libero import get_libero_path; assert get_libero_path('benchmark_root') == '/opt/LIBERO/libero/libero'" \
    && LIBERO_CONFIG_PATH=/opt/nanovla-config/libero_plus PYTHONPATH=/opt/LIBERO-plus \
        python3 -c "from libero.libero import get_libero_path; assert get_libero_path('benchmark_root') == '/opt/LIBERO-plus/libero/libero'"
ENV HF_HOME=/assets/hf \
    HF_HUB_OFFLINE=1 \
    LIBERO_CONFIG_PATH=/opt/nanovla-config/libero \
    MUJOCO_GL=egl \
    NANOVLA_LIBERO_PLUS_ROOT=/opt/LIBERO-plus/libero/libero \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    PYOPENGL_PLATFORM=egl \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/nanovla-verifier:/workspace/nanovla:/opt/LIBERO \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_OFFLINE=1
# /assets/nanovla fences the shard mount: group 65534 (the training UID) may
# enter, the serving UID may not. /assets/hf stays world-readable.
RUN mkdir -p /assets/nanovla /logs/eval-service /run/nanovla-eval \
    && chown root:65534 /assets/nanovla && chmod 0750 /assets/nanovla \
    && chmod -R go-rwx /opt/nanovla-verifier /opt/LIBERO /opt/LIBERO-plus \
    && chmod -R a+rX /workspace/nanovla
USER 0:0
WORKDIR /workspace/nanovla
CMD ["sleep", "infinity"]
