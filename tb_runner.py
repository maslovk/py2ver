import os
import shutil
from pathlib import Path
from cocotb_tools.runner import get_runner

LANGUAGE = os.getenv("HDL_TOPLEVEL_LANG", "verilog").lower().strip()


def test_runner(top_name: str):
    hdl_toplevel_lang = os.getenv("HDL_TOPLEVEL_LANG", "verilog").lower().strip()
    sim = os.getenv("SIM", "icarus")

    # Project root: where hdl/main.v and tb.py live
    proj_path = Path(__file__).resolve().parent

    # Explicit build directory inside the project
    build_dir = proj_path / "sim_build"

    # Always start from a clean build so tb.py cannot be stale
    if build_dir.exists():
        shutil.rmtree(build_dir)

    verilog_sources = []
    vhdl_sources = []

    if hdl_toplevel_lang == "verilog":
        verilog_sources = [proj_path / "hdl" / "main.v"]
    else:
        vhdl_sources = [proj_path / "hdl" / "main.vhdl"]

    build_test_args = []
    if hdl_toplevel_lang == "vhdl" and sim == "xcelium":
        build_test_args = ["-v93"]

    runner = get_runner(sim)

    # Build DUT with a clean build_dir
    runner.build(
        verilog_sources=verilog_sources,
        vhdl_sources=vhdl_sources,
        hdl_toplevel=top_name,
        build_dir=str(build_dir),
        always=True,
        build_args=build_test_args,
    )

    # Run cocotb tests in the same build_dir, using tb.py as test_module
    runner.test(
        hdl_toplevel=top_name,
        test_module="tb",
        build_dir=str(build_dir),
        test_args=build_test_args,
    )
