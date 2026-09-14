"""Safely copy untrusted Task 06 artifacts into verifier-owned storage."""
from __future__ import annotations

import os
import shutil
import stat


MAX_FILES = 4096
MAX_BYTES = 2 * 1024**3


def _copy_directory(source_fd: int, destination: str, budget: dict) -> None:
    os.mkdir(destination, 0o755)
    with os.scandir(source_fd) as entries:
        for entry in entries:
            info = os.stat(
                entry.name, dir_fd=source_fd, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"artifact links are not allowed: {entry.name}")
            target = os.path.join(destination, entry.name)
            if stat.S_ISDIR(info.st_mode):
                flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                         | getattr(os, "O_NOFOLLOW", 0))
                child_fd = os.open(entry.name, flags, dir_fd=source_fd)
                try:
                    opened = os.fstat(child_fd)
                    if (opened.st_dev, opened.st_ino) != \
                            (info.st_dev, info.st_ino):
                        raise ValueError(
                            f"artifact changed while staging: {entry.name}")
                    _copy_directory(child_fd, target, budget)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(
                    f"artifact must be a regular file: {entry.name}")
            budget["files"] += 1
            budget["bytes"] += info.st_size
            if budget["files"] > MAX_FILES or budget["bytes"] > MAX_BYTES:
                raise ValueError("artifact payload exceeds the staging limit")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(entry.name, flags, dir_fd=source_fd)
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode) \
                        or (opened.st_dev, opened.st_ino) != \
                        (info.st_dev, info.st_ino):
                    raise ValueError(
                        f"artifact changed while staging: {entry.name}")
                with os.fdopen(descriptor, "rb", closefd=False) as src, \
                        open(target, "xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            finally:
                os.close(descriptor)
            os.chmod(target, 0o555 if info.st_mode & 0o111 else 0o444)


def stage_submission(source: str, destination: str) -> None:
    """Create ``destination`` as a link-free snapshot of ``source``."""
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    try:
        info = os.lstat(source)
    except FileNotFoundError:
        os.mkdir(destination, 0o755)
        return
    if not stat.S_ISDIR(info.st_mode):
        os.mkdir(destination, 0o755)
        return
    flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
             | getattr(os, "O_NOFOLLOW", 0))
    source_fd = os.open(source, flags)
    try:
        opened = os.fstat(source_fd)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("artifact directory changed while staging")
        _copy_directory(
            source_fd, destination, {"files": 0, "bytes": 0})
    finally:
        os.close(source_fd)
