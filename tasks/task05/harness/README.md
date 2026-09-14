# Task05 harness

Source for the four nanoVLA subtasks. The agent images copy `runtime/` into the
agent-visible workspace; `verifier/` and the handoff helpers are root-owned
verifier infrastructure.

- `runtime/`: encoder loaders (`towers.py`), the policy socket protocol
  (`remote_protocol.py`) and the evaluation-service client (`nanovla_eval.py`)
- `verifier/`: replay, rollout, the evaluation service and scoring
- `config/`: pinned LIBERO and LIBERO-plus in-container roots
- `validate_assets.py`: contracts for the external read-only assets and encoder bundles
- `stage_submission.py` and `finalize_submission.py`: the one-file handoff boundary

`sim/libero/` and `sim/robotwin/` own only the pinned dependency layers and
Dockerfiles. Task-specific code belongs here.
