# RoboTwin 2.0 layer

Simulator stack for the task05 RoboTwin subtasks: SAPIEN 3 with Vulkan ray
tracing, mplib and cuRobo (the expert planner the seed-feasibility check and
instruction generation need), the RoboTwin task suite, the aloha-agilex
embodiment, object and background-texture assets.

`third_party/robotwin/` (gitignored) is vendored from `pins.env`: RoboTwin at
`ROBOTWIN_SHA` and cuRobo at `CUROBO_SHA`, each with its `patches/` (the
verifier's `clean50`/`rand50` task configs; the `helper_math.h` lerp rename that
lets cuRobo compile against CUDA 12.6 / gcc 11), plus the three asset archives
of `TianxingChen/RoboTwin2.0` at `ROBOTWIN_ASSETS_REVISION`, checked by sha256.

```bash
sim/robotwin/robotwin.sh images    # vendor if stale, then rlebench-robotwin-sim:dev
                                   # and rlebench-task05-robotwin-verifier-base:dev
```

The verifier base carries the task05 verifier harness on top of the
simulator; the per-subtask verifier images add the subtask's encoder bundle and
`/tests` (`make task05`), and the same base runs the evaluation service beside
the agent container. Agents use the ordinary `rlebench-task05-agent-*` images:
nothing of the simulator reaches them.
