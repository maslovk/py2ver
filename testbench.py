import queue
from cocotb.clock import Clock
import cocotb
from cocotb.triggers import Timer
import os
from pathlib import Path
from cocotb.runner import get_runner
from cocotb.triggers import Timer, RisingEdge, ReadOnly
import pickle


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
        hdl_toplevel=top_name, test_module="testbench", test_args=build_test_args
    )


async def save_results(ret_val):
    proj_path = Path(__file__).resolve().parent
    if not os.path.isdir(proj_path.joinpath('results')):
        os.mkdir(proj_path.joinpath('results'))

    with open(proj_path.joinpath('results').joinpath('results.pickle'), 'wb') as handle:
        pickle.dump(ret_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Start a testcase
@cocotb.test()
def tb_test(dut):
    dut.arg1.value = 2
    dut.arg2.value = 2
    yield Timer(1)
    return_val = int(dut.a.value)
    print("FIFO Dout Value=", return_val)
    yield save_results(str(return_val))

