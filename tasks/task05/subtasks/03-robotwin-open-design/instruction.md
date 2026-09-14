# Open-design nanoVLA on RoboTwin 2.0 with DINOv2

Maximise success on 15 bimanual RoboTwin 2.0 tasks with DINOv2-base as the
only pretrained component. You get demonstrations, the encoder and a socket
protocol; you decide everything else: whether to freeze, fine-tune or
partially reuse the encoder, how language is handled, the policy architecture,
the optimiser, the serving code.

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
  training-style instruction paraphrases (different wording, same goal);
  `norm_stats.json` holds per-dimension percentiles and extremes; `meta.json`
  describes the layout and holds `n_frames`, `n_episodes`, `img_size`,
  `state_dim`, `act_dim`, `n_tasks` and `episodes_per_task`.
- `/assets/hf` (read-only, `HF_HOME`): `facebook/dinov2-base` (ViT-B/14,
  86M parameters), offline. `towers.load_vision("dinov2")` returns it as a
  frozen feature extractor; you may also load it with `transformers`
  (`local_files_only=True`) and train, prune, distil or read any part of it.
  There is no text encoder: every instruction the policy will receive is a
  paraphrase in the style of `instructions.json`, and language handling is
  yours.

## What the test episodes look like

The same 15 tasks in the clean setting of the demonstrations (plain table,
fixed background and lighting), with object placements from seeds the
demonstrations and the development service never use, and an instruction
drawn from the same paraphrase templates as `instructions.json` for that
scene's objects.

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
of solvable clean-scene seeds per task with training-style instructions:
episode i of a task is the same scene and sentence on every request (up to 20
per task). The verifier scores different solvable seeds of the clean setting, which the
service never serves.
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
- DINOv2-base is the only pretrained initialisation allowed; anything you do
  with it is fine. If the served policy needs modified encoder weights, they
  must be in `ckpt.pt` (2 GiB cap).

## Scoring

The verifier replays train mode, serves the policy and runs 300 held-out
clean-scene episodes: 20 solvable seeds for each of the 15 tasks, each with
one training-style instruction, under RoboTwin's per-task step limits (at
least 95% must complete, and missing episodes count as failures in the fixed
denominator).

```text
reward = success_rate
```
