ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04@sha256:b3e7fba84d169f46939f00c25be7d016f712a8d651f4756d6a55e693d84d94f2

# ------------------------------------------------------------------- sim
# RoboTwin 2.0 stack: SAPIEN 3 (Vulkan ray tracing), mplib + cuRobo planners
# for the expert scripts, the task suite, embodiments and 15 GB of assets.
FROM ${BASE_IMAGE} AS sim
ENV DEBIAN_FRONTEND=noninteractive
# libegl1/libglvnd0/libgles2 are not optional: NVIDIA's Vulkan ICD dlopens
# libEGL.so.1 while initialising, and without it vk_icdNegotiateLoaderICDInterfaceVersion
# returns VK_ERROR_INITIALIZATION_FAILED and SAPIEN reports "failed to find a rendering device".
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git libegl1 libgl1 libgles2 libglib2.0-0 libglvnd0 libgomp1 \
        libvulkan1 libx11-6 libxext6 libxrender1 ninja-build procps python3 \
        python3-dev python3-pip python3-venv util-linux \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/robotwin-venv
ENV PATH=/opt/robotwin-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY sim/robotwin/requirements.txt /tmp/robotwin-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/robotwin-requirements.txt
# cuRobo (the expert planner) compiled for H100; the vendored tree carries the
# helper_math lerp rename that makes it build against CUDA 12.6 / gcc 11. It ships
# without .git, so setuptools-scm is given the version.
COPY third_party/robotwin/curobo /opt/curobo
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=16 \
    python3 -m pip install --no-cache-dir --no-build-isolation -e /opt/curobo
# RoboTwin itself: code, task configs, instruction templates, embodiments, objects, textures.
COPY third_party/robotwin/envs /opt/robotwin/envs
COPY third_party/robotwin/description /opt/robotwin/description
COPY third_party/robotwin/script /opt/robotwin/script
COPY third_party/robotwin/task_config /opt/robotwin/task_config
COPY third_party/robotwin/code_gen /opt/robotwin/code_gen
COPY third_party/robotwin/assets /opt/robotwin/assets
# upstream install fixes (script/_install.sh)
RUN SAPIEN=$(python3 -c "import sapien, os; print(os.path.dirname(sapien.__file__))") \
    && sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$SAPIEN/wrapper/urdf_loader.py" \
    && MPLIB=$(python3 -c "import mplib, os; print(os.path.dirname(mplib.__file__))") \
    && sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$MPLIB/planner.py" \
    && mkdir -p /usr/share/vulkan/icd.d \
    && printf '{"file_format_version":"1.0.1","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}\n' > /usr/share/vulkan/icd.d/nvidia_icd.json \
    && cd /opt/robotwin && python3 script/update_embodiment_config_path.py \
    && grep -q "/opt/robotwin" assets/embodiments/aloha-agilex/curobo_left.yml \
    && grep -q "/opt/robotwin" assets/embodiments/aloha-agilex/curobo_right.yml
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/robotwin \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1
WORKDIR /opt/robotwin
CMD ["sleep", "infinity"]

# -------------------------------------------------------------- verifier
# Root-owned base for the RoboTwin subtasks' verifier images and their
# evaluation sidecar: the simulator stack above plus the task05 verifier
# harness. Everything private is root-only; the shard mount is fenced.
FROM sim AS verifier
COPY sim/robotwin/requirements-verifier.txt /tmp/robotwin-verifier-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/robotwin-verifier-requirements.txt
COPY tasks/task05/harness/runtime/ /workspace/nanovla/
COPY tasks/task05/harness/verifier/ /opt/nanovla-verifier/
COPY rlebench/__init__.py /opt/nanovla-verifier/rlebench/__init__.py
COPY rlebench/core/__init__.py rlebench/core/media.py /opt/nanovla-verifier/rlebench/core/
ENV HF_HOME=/assets/hf \
    PYTHONPATH=/opt/nanovla-verifier:/workspace/nanovla:/opt/robotwin \
    NANOVLA_SIMULATOR=robotwin
RUN mkdir -p /assets/nanovla /logs/eval-service /run/nanovla-eval /workspace/runs \
    && chown root:65534 /assets/nanovla && chmod 0750 /assets/nanovla \
    && chmod -R go-rwx /opt/nanovla-verifier /opt/robotwin \
    && chmod -R a+rX /workspace/nanovla
USER 0:0
# the RoboTwin action space is 14-dim; remote_protocol reads this on both sides
ENV NANOVLA_ACTION_DIM=14
WORKDIR /workspace/nanovla
CMD ["sleep", "infinity"]
