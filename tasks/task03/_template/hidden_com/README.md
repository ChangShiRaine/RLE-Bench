# 05 — Hidden center of mass

One agent phase: identify a sealed box's ballast quadrant by robot interaction,
across three boxes with different mass and inertia configurations. Submit once
per box with `HiddenCOMClient.submit("A"|"B"|"C"|"D")`. See
[instruction.md](instruction.md) for the goal and budget; API and scene
documentation lives in the public client module.

No reset, no development/evaluation split, no force/torque observations, and no
object-pose API. The box carries A/B/C/D markings that follow its motion and obey
camera occlusion. A centred 30 g bridge handle permits a top-down grasp and
transport; its mass and inertia are part of the rigid body.

The image derives from the tabletop image and needs 8 CPUs, 16 GiB RAM and EGL
rendering on one GPU. The agent timeout is one hour, with @@INTERACTION_STEPS@@
control steps per box at 20 Hz. The shared-container verifier runs as root and
reads a record under `/var/lib/rlebench` written only by the private simulator;
submission acknowledges receipt without correctness feedback and starts the next
box.

The simulator records workspace and top views at 10 fps and the verifier exports
them to `verifier/media/trial-NN/interaction.mp4`. Recording is best-effort and
never scored; set `RLEBENCH_MEDIA=0` in the container environment to disable it.

```
instruction.md   the agent prompt
environment/     Docker build context, private simulator and public client
solution/        RGB-only handle-lift Oracle
tests/test.sh    root verifier; writes /logs/verifier/reward.json
```

```bash
make task03
harbor run -p tasks/task03/05-hidden-center-of-mass -a oracle --override-gpus 0
```

Sources live in `tasks/task03/tabletop/hidden_com/` and
`tasks/task03/_template/hidden_com/`; `build_hidden_com.py` emits this
gitignored tree. `dev/hidden_com/` holds the renderer preview and the
in-container isolation check.
