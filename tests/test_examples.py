import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
# Example scripts are the ones using the shared runner (ignores generated tb.py etc.)
SCRIPTS = sorted(
    p for p in EXAMPLES.glob("*.py")
    if not p.name.startswith("_") and "from _run import run" in p.read_text(encoding="utf-8")
)


@pytest.mark.sim
@pytest.mark.parametrize("script", SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_example_matches_python_in_simulation(script, tmp_path):
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    pytest.importorskip("cocotb")

    # Generated files go to the working directory, so run each in its own
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=600
    )

    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "simulation     OK" in result.stdout
