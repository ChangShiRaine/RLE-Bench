# Verifier image (SEPARATE environment): the whole harness including the scorer,
# the calibrated thresholds and the hidden evaluation seeds.
#
# CPU only. Training is MuJoCo-Warp on the GPU; scoring is MuJoCo-C, so this
# image carries neither torch nor warp and scores a battery in seconds.
#
# Built from the REPO ROOT as context, same as the agent image.
FROM python:3.12-slim

ARG MUJOCO_VER
ARG ONNX_VER
ARG ONNXRUNTIME_VER
ARG NUMPY_VER

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libosmesa6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "mujoco==${MUJOCO_VER}" \
        "onnx==${ONNX_VER}" \
        "onnxruntime==${ONNXRUNTIME_VER}" \
        "numpy==${NUMPY_VER}"

COPY tasks/task04/harness /tests/harness
COPY tasks/task04/_template/tests/test.sh tasks/task04/_template/tests/score_task.py /tests/
COPY rlebench/__init__.py /tests/rlebench/__init__.py
COPY rlebench/core/__init__.py rlebench/core/media.py /tests/rlebench/core/
COPY third_party/motiontrack/unitree_g1 /tests/assets/unitree_g1
COPY third_party/motiontrack/motions /tests/assets/motions

ENV MUJOCO_GL=osmesa
ENV PYTHONPATH=/tests
ENV MOTIONTRACK_ROBOT_DIR=/tests/assets/unitree_g1

RUN chmod +x /tests/test.sh
