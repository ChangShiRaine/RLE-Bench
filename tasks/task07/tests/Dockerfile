# Verifier container (separate environment): full harness including the
# scorer, thresholds and evaluation seeds. test.sh locks /tests before any
# submitted code runs. Built by tasks/task07/build_assets.py.
FROM python:3.12-slim

# Submissions are re-imported here, so the runtime must match the agent
# image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libosmesa6 libegl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        mujoco==3.5.0 \
        numpy==2.5.1 \
        scipy==1.18.0 \
        matplotlib==3.11.0 \
        pytest==9.1.1 \
        opencv-python-headless==4.13.0.92 \
        torch==2.7.1+cpu

ENV MUJOCO_GL=osmesa
ENV RLEBENCH_ASSETS=/tests/assets

COPY harness/ /tests/harness/
COPY rlebench/ /tests/rlebench/
COPY assets/ /tests/assets/
COPY score_task.py test.sh /tests/

RUN chmod +x /tests/test.sh
