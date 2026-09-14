"""Task07 agent/verifier dependency parity."""
import re

from tasks.task07 import build_assets


def _pins(dockerfile: str) -> dict[str, str]:
    return dict(re.findall(r"([a-zA-Z0-9_.-]+)==([^\s\\]+)", dockerfile))


def _apt_packages(dockerfile: str) -> set[str]:
    block = dockerfile.split("apt-get install -y --no-install-recommends", 1)[1]
    block = block.split("&& rm -rf", 1)[0]
    return set(re.findall(r"lib[a-zA-Z0-9+.-]+|ffmpeg", block))


def test_agent_and_verifier_dependency_sets_match():
    agent = build_assets.AGENT_DOCKERFILE
    verifier = build_assets.VERIFIER_DOCKERFILE
    assert _pins(verifier) == _pins(agent)
    assert _apt_packages(verifier) == _apt_packages(agent)
    assert {"opencv-python-headless", "torch"} <= _pins(verifier).keys()
