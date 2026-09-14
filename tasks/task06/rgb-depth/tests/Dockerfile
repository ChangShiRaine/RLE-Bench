# Verifier container (separate environment): the full harness and the
# evaluation seeds. The agent container never sees this image; at runtime
# test.sh locks /tests before any submitted code runs.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libosmesa6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mujoco==3.5.0 \
        numpy==2.5.1 \
        scipy==1.18.0 \
        matplotlib==3.11.0 \
        opencv-python-headless==4.13.0.92

ENV MUJOCO_GL=osmesa
ENV RLEBENCH_ASSETS=/tests/harness/assets

COPY tasks/task06/rgb-depth/tests/harness/ /tests/harness/
COPY rlebench/__init__.py /tests/rlebench/__init__.py
COPY rlebench/core/__init__.py rlebench/core/model.py rlebench/core/scoring.py rlebench/core/media.py /tests/rlebench/core/
COPY tasks/task06/rgb-depth/tests/score_task.py tasks/task06/rgb-depth/tests/test.sh /tests/

RUN chmod +x /tests/test.sh
