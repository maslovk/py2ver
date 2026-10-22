import importlib.util
import re
import sys
from pathlib import Path

import pytest

# The guides' examples run as written: each ```python block of a guide, in
# order, as one module (py2ver reads the generators' source, so a file, not
# exec). ```py blocks (Quartus, the board) are shown, not run.

DOCS = Path(__file__).resolve().parent.parent / "docs"


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
