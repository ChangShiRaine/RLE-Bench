# Contributing

Contributions are accepted under the repository's MIT License (inbound =
outbound); vendored third-party material keeps its own terms, see
`THIRD_PARTY_NOTICES.md`.

Read `AGENTS.md` first — its hard invariants (score only against the
harness's own instrumentation; no ground truth in any agent environment;
determinism; golden-first) are non-negotiable and several are enforced by
tests.

## Dev setup

```bash
make install               # .venv via uv: pinned deps + harbor + pyflakes
make task-assets           # stage all families, no Docker
make sim-robocasa           # task01/02: image + sources + ~15 GB dataset + .venv-robocasa
make sim-motiontrack       # task04: assets + .venv-motiontrack
make sim-libero           # task05: encoder bundles + agent images + LIBERO verifier base
make sim-robotwin          # task05 03/04: the RoboTwin 2.0 simulator + its verifier base
```

The Makefile has one scheme: `sim-<layer>` builds a shared simulator layer,
`taskNN` / `taskNN-assets` / `taskNN-clean` build, stage and tear down a family,
and everything else is `test-*`, `check-*` or `calibrate-*` keyed by family id.

Host pythons routinely lack ensurepip; everything venv-related goes through
uv. The three venvs exist because their mujoco pins conflict — never merge
them.

`make lint` and `make lint-strict` enforce Pyflakes on tracked and non-ignored
new Python files under `rlebench`, `scripts`, `sim`, `tasks`, and `tests`.
Ignored generated trees are excluded; task08's controller fragment is checked
after assembly in memory, without staging assets; three private-only names
(`ArmSpec`, `ARM_STOW`, `arm_extended`) used by its `arm=None` constructor are
exempted, since the Oracle supplies an arm explicitly.

## The rules that keep evals comparable

- **Generator-owned files are never hand-edited.** task01/02/03/04 trees are
  emitted from `_template/`; task06's four task dirs and task07's staged
  tomls/scripts are committed *and* generator-owned — edit the generator,
  re-run it, commit the output together (`make check-taskgen` asserts
  stability). task07's staged franka meshes are gitignored and rebuilt by
  `make task-assets`, like task08/09's trees.
  Task09's generated models and geometry contract are also committed and
  checked for drift.
- **Eval setting changes require a `[task].version` bump** for the affected
  family.
- **Pins live in one place per layer**: `requirements.txt` for the
  mujoco-3.5.0 image family (parity-tested against every consuming Dockerfile
  and the task06 generator), `sim/*/pins.env` for the ARG-pinned images. Bump a pin and the parity tests point at every
  consumer that must move with it.
- **Two verifier models are deliberate**: separate-environment
  (task04/05/06/07/08/09) vs shared-container root verifier (task01/02, because
  Harbor chmods `/logs/artifacts` 0777). Do not unify them.
- **Agent-visible bytes are behavior.** Comment or prose edits inside
  `environment/`, `tests/`, `instruction.md` or anything staged into an
  image are eval changes, not cleanups.

## CI and release conventions

- CI (`.github/workflows/ci.yml`) runs lint, per-family host suites,
  generator self-checks and the Harbor pin check on every PR.
  Shared checks run once with `make test-host`; nightly CI runs only the
  deferred `heavy` tests in task08 and task06. Task02's shared wiring checks
  sample two groups; split coverage checks still inspect every active group.
  Task04's CPU suite runs with its own pins and motion assets on every PR;
  CUDA trainer checks run in GPU nightly.

`make check-taskgen` stages all nine families, runs the available generator
and public-boundary checks, and rejects tracked task06/07/09 output drift.
Task03 includes all five tasks and its three image contexts. These checks
need the installed host dependencies, but no Docker, GPU, or simulator dataset.
For one family, use `make taskNN-assets`.

## The manual release gate (what CI cannot run)

Before tagging, or after changing `tasks/<family>/harness/` or `tasks/<family>`:

1. Run the family's full host suite with `make test TASK=<family>`.
   Dedicated `make test-<family>-golden` suites exist for task04 and
   task06–09; run those where available. These are host regressions, not
   full-container Oracle evaluations.
2. Simulator suites where they exist (`.venv-robocasa` + `-m simulator`,
   `make test TASK=task04`).
3. Where an Oracle script is available, run it end-to-end:
   `rlebench run <task> -a oracle`. Check a full solution's expected score;
   distinguish protocol smoke checks from evidence of solvability. Oracle
   solutions are optional; report which validation was actually performed.

Known environment caveats: the task06 `rgb-depth-model-training` oracle
trains in-container and can exceed its 2 h agent budget on a loaded host —
run it on a quiet machine; task07's verifier is wall-budget sensitive, never
run two of them concurrently.
