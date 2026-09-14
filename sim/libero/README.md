# LIBERO layer

Simulator stack for the task05 LIBERO subtasks, plus the family's encoder bundles
and agent images. Task-owned source stays in `tasks/task05/harness/`. The build
creates:

- `rlebench-task05-hf-{base,dinov2,open}:dev` (`Dockerfile.hf`): one validated
  offline `HF_HOME` each, fetched at the commits `validate_assets.py` pins by
  `libero.sh bundles`; only ever a `COPY --from` source.
- `rlebench-task05-agent-{base,dinov2,open}:dev`: the development container
  with its bundle at `/assets/hf`. CUDA, torch, the Hugging Face loaders, the
  policy socket protocol and the evaluation-service client. No simulator, no
  evaluator source.
- `rlebench-task05-libero-verifier-base:dev`: root-owned LIBERO and LIBERO-plus
  stack with the rollout evaluator. Each LIBERO subtask's verifier image adds its
  bundle and `/tests` (`make task05`); the same base runs the evaluation
  service beside every LIBERO agent container. The RoboTwin subtasks derive from
  `sim/robotwin` instead.

```bash
sim/libero/libero.sh bundles     # fetch + validate the bundles, build the bundle images
sim/libero/libero.sh base        # the agent images + the verifier base (make sim-libero)
sim/libero/libero.sh verifiers   # the four per-subtask verifier images (make task05)
sim/libero/libero.sh check <NN-slug>   # validate the directories a subtask mounts
```

Shards and the LIBERO-plus assets stay outside Git and are mounted read-only by
exact path; see `tasks/task05/README.md` for the variables.
