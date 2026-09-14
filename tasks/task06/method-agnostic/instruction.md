# Task 06 / method-agnostic — Pose and shape estimation

A robot-mounted stick pushes one of three red, asymmetric blocks on a table.
From a fixed oblique camera, estimate the block's planar pose and identify its
shape. Predictions should remain accurate through rotation and occlusion,
with efficient CPU inference.

## Training data

Use the public clients in `/workspace/harness`. Meshes and the private scene
are not available; learn shape geometry and orientation conventions from
rendered examples.

~~~python
from harness import episodes

frame = episodes.single_frame(11)
print(frame.obs.keys())  # rgb, depth, K, T_cam_table, t
print(frame.gt)          # noisy (x, y, theta), no shape label
episode = episodes.push_episode(23)
~~~

Public design seeds are 11, 23, and 37. You may draw additional 32-bit seeds
from them. Calls are deterministic by seed, including observation and label
noise. Pose labels are noisy; shape labels are not provided. The shape stays
fixed within an episode but may differ between scenes.

Observations contain `rgb, depth, K, T_cam_table, t`.
RGB is uint8; depth, when available, is in meters with NaN for missing pixels.
`K` is the camera intrinsic matrix, `T_cam_table` its pose in the table frame,
and `t` is time in seconds. See `harness/spec.py` for camera constants.

Pose means `(x, y, theta)`: x and y in meters in the table frame, theta about
+z in radians wrapped to (-pi, pi]. Each block is 40 mm tall. Its orientation
convention is defined by the pose labels. Shape IDs are fixed: 0 = T,
1 = C, 2 = F. Use this mapping on every frame, including after resets.
Shape identity is never an input.

## Method and submission

Any method is allowed using numpy, scipy, OpenCV, torch, and scikit-learn. One GPU is available during development; stateful tracking is allowed.

Submit `/logs/artifacts/estimator.py` with `make_estimator()`, returning an
object with `reset()` and `update(*, rgb, K, T_cam_table, t, depth=None)`.
Each update returns exactly `(x, y, theta, shape_id)`, with an integer shape ID.
Reset clears tracking state, not the shape-ID convention. See
`harness/estimator_template.py`. Helper files may sit beside `estimator.py`;
the harness package is not importable during evaluation.

All variants must run on CPU during evaluation, even when a GPU is available
for training. Process observations causally, without future frames. Return
finite outputs and handle missing depth and temporary occlusion gracefully.

## Development

After staging a submission, `harness.evaluator.evaluate()` provides limited
design-set pose diagnostics. At most six calls are available per container.
Training and data generation must finish within the two-hour agent budget.
Only files under `/logs/artifacts/` are collected; reports and CSVs are optional.
