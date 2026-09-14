# Hidden center-of-mass development tools

The runnable single-stage task is emitted by `make task03-assets`.
See [the task README](../../_template/hidden_com/README.md) for its interface,
private simulator boundary, build commands and Oracle run.

Render the shared scene using the pinned simulator environment:

```bash
.venv-robocasa/bin/python tasks/task03/dev/hidden_com/preview.py
```

`--quadrant A|B|C|D` selects the preview fixture; `--output DIR` overrides
`results/task03/hidden_com`. Outputs include three views, a comparison
sheet, portable XML/assets, and mass/COM and repeatability checks. Load the
`preview` keyframe to inspect the initial pose. XML contains private inertia
and is only for maintainers.

`check_isolation.py` runs inside a running task image as the `agent` user.
It checks protected-file reads/writes, private imports, root-only operations,
and the public observation allowlist. The physical Oracle regression exercises
all four hidden quadrants for each of the three mass configurations in `tests/test_tabletop_hidden_com.py`.
