"""Shared plumbing for the task generators and asset stagers.

Only the pieces that were byte-duplicated live here: @@TOKEN@@ substitution
(task01, task04), the cleaned copytree and the base verifier-exclude globs
(task08, task09). Each generator's matrix, checks and CLI stay local -- their
--check commands prove per-task theorems (cross-level byte-equality, split
agreement, template fidelity) that a generic engine would only water down.
task02's single .replace(SLUG, ...) stays inline for the same reason.

Stdlib only: the Makefile invokes the generators with bare python3.
"""

from __future__ import annotations

import os
import shutil

def render(text: str, values: dict[str, str]) -> str:
    """@@TOKEN@@ substitution. Values must not themselves contain tokens."""
    for token, value in values.items():
        text = text.replace(token, value)
    return text


def unrendered(text: str, marker: str = "@@") -> bool:
    return marker in text


def clean_copytree(src: str | os.PathLike, dst: str | os.PathLike,
                   extra_ignore: tuple = (), **kw) -> None:
    """copytree into a freshly-removed dst, never shipping bytecode caches."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                  *extra_ignore), **kw)
