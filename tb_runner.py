import os
from pathlib import Path
from cocotb.runner import get_runner


LANGUAGE = os.getenv("HDL_TOPLEVEL_LANG", "verilog").lower().strip()



def test_runner(top_name):
    hdl_toplevel_lang = os.getenv("HDL_TOPLEVEL_LANG", "verilog")
    sim = os.getenv("SIM", "icarus")

    proj_path = Path(__file__).resolve().parent

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
    runner.build(
        verilog_sources=verilog_sources,
        vhdl_sources=vhdl_sources,
        hdl_toplevel=top_name,
        always=True,
        build_args=build_test_args,
    )
    runner.test(
        hdl_toplevel=top_name, test_module="tb", test_args=build_test_args
    )

