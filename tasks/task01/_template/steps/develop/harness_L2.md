**There are no object or fixture poses.** The simulator knows where everything is; you
are not told. That is deliberate — a real robot would have to perceive it — so finding
what you need to manipulate is part of the task, and the cameras are how you do it.

### You have a harness library

You are not starting from an empty action loop. A library ships in this container, and
**its manual is at `/opt/HARNESS_MANUAL.md` — read it first.**

```python
from harness.skills import camera, geometry, perception, transforms
from harness.skills import reach, grasp, lift, settle, move_base

hits = perception.segment_by_text(rgb, "coffee mug")    # SAM3, free, may return []
pts = camera.cloud_in_base(obs, CAM, mask=hits[0]["mask"])   # pixels -> BASE frame
box = geometry.oriented_bbox(geometry.cluster_points(pts)[0])

reach(sim, box["center"], gripper=-1.0)                 # a target YOU worked out
settle(sim, gripper=-1.0); grasp(sim); lift(sim, 0.15)
```

Five tiers: `perception` (SAM3 text-prompted segmentation and Contact-GraspNet
proposals, running in this container and free — they touch no simulator), `transforms`
(frames and rotations, `world_to_base` above all), `camera` (intrinsics and z-depth
deprojection into the camera frame), `geometry` (drop the counter, cluster, box, choose a
grasp) and `primitives` (`reach`, `move_eef`, `grasp`, `release`, `lift`, `move_base`,
`settle`, taking base-frame targets).

**There is no `pick("mug")`.** Deciding what matters, finding it and choosing how to take
hold of it is the task; the library removes the servo loops and the pixel arithmetic
around it, not the problem itself.

Skills are ordinary code in **your** process over the same metered socket. Their steps
cost what your own would, and they see exactly the observation you see — a skill cannot
tell you anything the cameras did not. Call them, wrap them, rewrite them, or ignore them
entirely.
