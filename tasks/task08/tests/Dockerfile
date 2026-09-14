# Verifier container: the task08 harness (metrics, sim, scoring) plus the
# canonical mesh/component assets a submission resolves against. Runs as a
# separate environment (task.toml [verifier]); the agent container never sees
# this image. Pins must match environment/Dockerfile.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        mujoco==3.5.0 \
        numpy==2.5.1 \
        scipy==1.18.0 \
        matplotlib==3.11.0

# Build context is the repo root: the staged trees come from build_assets.py
# and rlebench.core from its one source.
COPY tasks/task08/tests/harness/ /tests/harness/
COPY rlebench/__init__.py /tests/rlebench/__init__.py
COPY rlebench/core/__init__.py rlebench/core/model.py rlebench/core/scoring.py rlebench/core/media.py /tests/rlebench/core/
COPY tasks/task08/tests/models/ /tests/models/
COPY tasks/task08/tests/score_task.py tasks/task08/tests/test.sh /tests/

RUN chmod +x /tests/test.sh && chmod 700 /tests
