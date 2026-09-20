"""Compose the single-stage tabletop session with the shared socket daemon."""

from __future__ import annotations

from typing import Any


def _install_env_factory() -> None:
    from harness import env as _env

    from . import TASKS

    original = getattr(_env, "make_env", None)
    if original is None or not callable(original):
        raise RuntimeError(
            "harness.env.make_env is missing -- the tabletop family "
            "wraps it to build its own scenes; see this module's docstring")
    if getattr(original, "_tabletop_wrapped", False):
        return

    def make_env(*args: Any, **kwargs: Any):
        task = kwargs["task"] if "task" in kwargs else (args[0] if args else None)
        if task in TASKS:
            return _make_tabletop(task, kwargs)
        return original(*args, **kwargs)

    make_env._tabletop_wrapped = True          # type: ignore[attr-defined]
    _env.make_env = make_env


def _make_tabletop(task: str, kwargs: dict):
    """Construct a tabletop scene using the shared observation defaults."""
    from harness import env as _env

    from . import make as _make

    # Match the shared recorder's upright image convention.
    _env._use_upright_images()
    return _make(
        task,
        seed=kwargs.get("seed"),
        camera_names=list(kwargs.get("camera_names", _env.DEFAULT_CAMERAS)),
        camera_height=kwargs.get("camera_height", _env.RENDER_RESOLUTION),
        camera_width=kwargs.get("camera_width", _env.RENDER_RESOLUTION),
        camera_depths=kwargs.get("camera_depths", False),
    )


def install() -> None:
    """Compose task03's session and scene factory in this daemon process only."""
    from harness import service, daemon_main
    from .session import Session
    from .service import Service

    _install_env_factory()
    service.MeteredSession = Session
    daemon_main.Service = Service


def main(argv: list[str] | None = None) -> int:
    import os
    os.environ["RLEBENCH_LEVEL"] = "L1"
    install()
    from harness.daemon_main import main as shared_main
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    return shared_main([*args, "--eval-plan", "1x1", "--submissions", "1"])


if __name__ == "__main__":
    raise SystemExit(main())
