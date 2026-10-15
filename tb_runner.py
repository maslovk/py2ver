from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from cocotb_tools.runner import get_runner

LANGUAGE = os.getenv("HDL_TOPLEVEL_LANG", "verilog").lower().strip()


def test_runner(top_name: str, proj_path: Path | None = None):
    hdl_toplevel_lang = os.getenv("HDL_TOPLEVEL_LANG", "verilog").lower().strip()
    sim = os.getenv("SIM", "icarus")

    # Output directory: where Py2ver wrote hdl/main.v and tb.py
    proj_path = Path.cwd() if proj_path is None else Path(proj_path)

    # Explicit build directory inside the project
    build_dir = proj_path / "sim_build"

    # Always start from a clean build so tb.py cannot be stale
    if build_dir.exists():
        shutil.rmtree(build_dir)

    sources = []

    if hdl_toplevel_lang == "verilog":
        sources = [proj_path / "hdl" / "main.v"]
    else:
        sources = [proj_path / "hdl" / "main.vhdl"]

    build_test_args = []
    if hdl_toplevel_lang == "vhdl" and sim == "xcelium":
        build_test_args = ["-v93"]

    runner = get_runner(sim)

    # Build DUT with a clean build_dir
    runner.build(
        sources=sources,
        hdl_toplevel=top_name,
        build_dir=str(build_dir),
        always=True,
        build_args=build_test_args,
    )

    # Run cocotb tests in the same build_dir, using tb.py as test_module.
    # The runner derives the simulator's PYTHONPATH from sys.path, so put
    # proj_path first to import this tb.py rather than any other on the path.
    sys.path.insert(0, str(proj_path))
    try:
        runner.test(
            hdl_toplevel=top_name,
            test_module="tb",
            build_dir=str(build_dir),
            test_args=build_test_args,
        )
    finally:
        sys.path.remove(str(proj_path))


def run_cocotb(top_name: str, sources, test_module: str, proj_path: Path,
               build_name: str = "sim_build", timescale=None):
    """Build `sources` with `top_name` as toplevel and run cocotb `test_module`."""
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(proj_path)
    build_dir = proj_path / build_name
    if build_dir.exists():
        shutil.rmtree(build_dir)

    runner = get_runner(sim)
    build_kwargs = {}
    if timescale is not None:
        build_kwargs["timescale"] = timescale
    runner.build(
        sources=[Path(s) for s in sources],
        hdl_toplevel=top_name,
        build_dir=str(build_dir),
        always=True,
        **build_kwargs,
    )

    sys.path.insert(0, str(proj_path))
    try:
        runner.test(
            hdl_toplevel=top_name,
            test_module=test_module,
            build_dir=str(build_dir),
        )
    finally:
        sys.path.remove(str(proj_path))
