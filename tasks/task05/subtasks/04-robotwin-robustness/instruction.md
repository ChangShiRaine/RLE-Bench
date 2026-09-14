# Robust nanoVLA on RoboTwin 2.0

Maximise success on perturbed versions of 15 bimanual RoboTwin 2.0 tasks. You
train on clean demonstrations and can only evaluate on clean episodes; you get
demonstrations, a bundle of encoders and a socket protocol, and you design the
model, the training recipe and the serving code.

## Time model

- **Exploration:** 14,400 seconds on one H100 (48 CPUs, 64 GiB). No network
  beyond your model API: nothing can be downloaded. Unlimited local
  trainings and development evaluations; every attempt consumes session
  time and none of it is an official submission.
- **Replay:** the offline verifier runs your final `solution.py` in train
  mode once, with 1,800 seconds plus a fixed 120-second infrastructure
  allowance for the whole process: interpreter start, data loading, feature
  extraction, compilation, training and the checkpoint write.
- **Evaluation:** afterwards, outside the replay clock, the verifier runs the
  same file in serve mode and rolls out episodes against it. The rollout is not
  untimed: a worker that finishes no episode for 3,600 seconds is killed and its
  episodes are retried from scratch, so a policy too slow to complete one episode
  inside that window scores nothing however good it is.

## What is available

- `/workspace/nanovla`: `remote_protocol.py` (the policy socket protocol and
  a `serve` helper), `towers.py` (loaders for the bundle models),
  `nanovla_eval.py` (client for the evaluation service). There is no model,
  trainer, simulator or evaluator source: architecture, training and
  inference are yours.
- `/assets/nanovla/shards` (read-only): 700 clean-scene demonstrations of
  15 RoboTwin 2.0 tasks on the aloha-agilex dual-arm robot, 135,519 frames.
  `agent.u8` (head camera) and `wrist.u8` (front camera) are `(N, 128, 128, 3)`
  uint8 memmaps; `state.f32` `(N, 16)` is the raw end-effector state (left
  pose 7, left gripper, right pose 7, right gripper); `actions.f32` `(N, 14)`
  are the raw joint targets (left arm 6, left gripper, right arm 6, right
  gripper); `ep_end.i32` gives each frame's exclusive episode end;
  `task_idx.u8` indexes `task_names.json` (15 tasks); `episode_id.i32` indexes
  `instructions.json`, which lists for every demonstration its 100
  instruction paraphrases (different wording, same goal);
  `norm_stats.json` holds per-dimension percentiles and extremes; `meta.json`
  describes the layout and holds `n_frames`, `n_episodes`, `img_size`,
  `state_dim`, `act_dim`, `n_tasks` and `episodes_per_task`.
- `/assets/hf` (read-only, `HF_HOME`): the open encoder registry, offline.
  Vision: `facebook/dinov2-base`, `facebook/dinov2-large`,
  `facebook/dinov2-giant`, `google/siglip-base-patch16-224`,
  `google/siglip2-base-patch16-224`, `google/siglip2-so400m-patch14-384`,
  `openai/clip-vit-base-patch16`. Text: the SigLIP and CLIP text towers,
  `sentence-transformers/all-MiniLM-L6-v2`, `google/flan-t5-small`,
  `google/flan-t5-base`. Use them frozen or fine-tune them; training from
  scratch is equally allowed. Only weights from this bundle may initialise
  the policy.

## Test episodes

Every scored episode is one of the 15 tasks under perturbations that neither
the demonstrations nor the development service contain, on seeds neither ever
uses: the language instruction, the lighting, the background, etc. The robot,
the cameras, the action space and the success criteria are unchanged, and the
clean demonstrations are the only data.

## Policy interface

`solution.py` is one file (<= 1 MiB, UTF-8) that the verifier runs twice, from
`/workspace/nanovla` with `PYTHONPATH` set, selected by `NANOVLA_MODE`:

- `train`: read the shards, train, write `$NANOVLA_OUTPUT_DIR/ckpt.pt` and
  exit 0. `ckpt.pt` is a dict written with `torch.save` and loaded with
  `weights_only=True`; its tensors may sit at the top level or nested in
  dicts, lists and tuples. An optional
  `meta.json` (<= 1 MiB, a JSON object) may sit beside it. Nothing else may be
  in that directory.
- `serve`: load from `$NANOVLA_OUTPUT_DIR` and serve on `$NANOVLA_SOCKET`.
  `remote_protocol.serve(socket_path, infer)` implements the protocol: `infer`
  receives a batch of requests, each with `language` (str), `cameras`
  (uint8 `(2, 128, 128, 3)`: head camera then front camera, preprocessed
  exactly like `agent.u8` / `wrist.u8`) and `state` (float32 `(16,)`, raw, like
  `state.f32`), and returns one `(K, 14)` float32 chunk of joint targets per
  request in the demonstrations' units (`actions.f32`: left arm 6, left
  gripper, right arm 6, right gripper), `1 <= K <= 64`. The protocol module
  reads the width from `NANOVLA_ACTION_DIM` (14 here), so
  `remote_protocol.ACTION_DIM` is already correct in both phases. The evaluator executes
  every returned action, then asks again. The socket must exist within 300
  seconds of start. The serve process can read only the output directory and
  the model bundle; the shards are not readable.

Both modes run offline. Keep every stochastic operation seeded.

## Development evaluation

A separate evaluation service holds the simulator. Start your policy server on
a socket created directly under `/run/nanovla-eval/`, then:

```bash
python nanovla_eval.py --policy /run/nanovla-eval/policy.sock --episodes 10
python nanovla_eval.py --policy /run/nanovla-eval/policy.sock --tasks 0,3 --episodes 5
```

Task indices follow `task_names.json`. Development episodes are a fixed list
of solvable clean seeds per task: episode i of a task is the same scene and
sentence on every request (up to 20 per task). The perturbed episodes the
verifier scores are never available during development.
Requests run one at a time, are logged, and share a budget of 20,000 episodes
per session. RoboTwin episodes take 20-90 seconds each; plan evaluations
accordingly. Import `nanovla_eval.evaluate` for the same call from Python.

## Submission

```bash
nanovla-stage /workspace/candidate.py   # -> /logs/artifacts/submission/solution.py
```

Stage early and after every improvement; the last staged file is the
submission. At the deadline `/workspace`, downloads, installed packages,
checkpoints, logs and background processes are discarded.

## Rules

- Train only on the mounted shards and the mounted model bundle. No other
  data, labels or weights may reach the final policy.
- The final file must be reproducible offline from the shards and the bundle
  inside the replay clock. There is no network during exploration either:
  nothing can be downloaded, and only what is in the container can be used.
- Do not key behaviour on task, episode or seed identity, probe the
  evaluator, or tamper with the protocol. No remote compute.

## Scoring

The verifier replays train mode, serves the policy and runs 225 perturbed
episodes: 15 solvable seeds for each of the 15 tasks, under RoboTwin's per-task
step limits (at least 95% must complete, and missing episodes count as failures
in the fixed denominator).

```text
reward = perturbed_success_rate
```

There is no clean-episode gate. Only the verifier holds the episode list.
