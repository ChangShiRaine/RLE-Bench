"""tasks/task05/fetch_data.py: archive extraction and the pinned fast path, offline."""

import importlib.util
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tasks" / "task05" / "fetch_data.py"
spec = importlib.util.spec_from_file_location("task05_fetch_data", SCRIPT)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)

PREFIX = "build/host/assets/"


def archive(tmp_path: Path, members: dict[str, bytes | None]) -> Path:
    path = tmp_path / "assets.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data or b"")
    return path


def test_extract_drops_the_prefix_its_parents_and_resource_forks(tmp_path):
    source = archive(tmp_path, {"build/": None, "build/host/": None, PREFIX: None,
                                PREFIX + "scenes/": None, PREFIX + "scenes/light.xml": b"<x/>",
                                "__MACOSX/build/._light.xml": b"fork"})
    fetch.extract(source, tmp_path / "out", PREFIX)
    files = {p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*") if p.is_file()}
    assert files == {"scenes/light.xml"}


@pytest.mark.parametrize("member", ["elsewhere/light.xml", PREFIX + "../../escape.xml"])
def test_extract_refuses_members_outside_the_prefix(tmp_path, member):
    with pytest.raises(ValueError):
        fetch.extract(archive(tmp_path, {member: b"x"}), tmp_path / "out", PREFIX)


def test_check_refuses_a_wrong_sha256(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"data")
    with pytest.raises(ValueError):
        fetch.check(path, "0" * 64)


def test_current_pins_only_validate(tmp_path, monkeypatch):
    (tmp_path / ".pins").write_text(fetch.PINS)
    validated = []
    monkeypatch.setattr(fetch, "validate", validated.append)
    monkeypatch.setattr(fetch, "fetch_shards", lambda *_: pytest.fail("downloaded despite current pins"))
    fetch.main(["--dest", str(tmp_path)])
    assert validated == [tmp_path.resolve()]
