# Verifier container: the scoring harness plus the reference arms. Runs as a
# separate environment (task.toml [verifier]); the agent container never sees
# this image. Must contain /tests/test.sh.
# Pins must match requirements.txt and environment/Dockerfile.
FROM python:3.12-slim

# software GL for the post-eval device renders (physics never touches GL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        mujoco==3.5.0 \
        numpy==2.5.1 \
        scipy==1.18.0 \
        matplotlib==3.11.0 \
        python-fcl==0.7.0.11

# Build context is the repo root (see Makefile); tests/harness and
# tests/models are staged by tasks/task09/build_assets.py.
COPY tasks/task09/tests/harness/ /tests/harness/
COPY rlebench/__init__.py /tests/rlebench/__init__.py
COPY rlebench/core/__init__.py rlebench/core/model.py rlebench/core/scoring.py rlebench/core/media.py /tests/rlebench/core/
COPY tasks/task09/tests/models/ /tests/models/
COPY tasks/task09/tests/score_task.py tasks/task09/tests/test.sh /tests/

RUN chmod +x /tests/test.sh
