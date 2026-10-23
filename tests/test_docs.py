import importlib.util
import re
import sys
from pathlib import Path

import pytest

# The guides' examples run as written: each ```python block of a guide, in
# order, as one module (py2ver reads the generators' source, so a file, not
# exec). ```py blocks (Quartus, the board) are shown, not run.

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def _blocks(path: Path) -> str:
    return "\n\n".join(re.findall(r"^```python\n(.*?)^```", path.read_text(encoding="utf-8"),
                                  flags=re.S | re.M))


@pytest.mark.sim
def test_the_processes_guide_runs_as_written(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "guide.py"
    source.write_text(_blocks(DOCS / "processes.md"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("guide", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "guide", module)
    spec.loader.exec_module(module)
    assert module.run.value == b"hi"


@pytest.mark.sim
def test_the_readme_runs_as_written(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "readme.py"
    source.write_text(_blocks(ROOT / "README.md"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("readme", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "readme", module)
    spec.loader.exec_module(module)
    assert module.outputs == [(4, 2, 3, 3), (9, 5, 14, 3), (510, 0, 65025, 1), (7, -3, 10, 0)]
    assert module.tx.process.states == 5
