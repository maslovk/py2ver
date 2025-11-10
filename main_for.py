from py2ver import Py2ver
import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ==========================================================
# Example HDL-like Python function with for-loop unrolling
# ==========================================================
def sum_loop(x):
    acc = 0
    for i in range(0, 10):       # unrolled by FunctionVisitor
        acc = acc + (x + i)
    return acc


# ----------------------------------------------------------
# Attribute definitions (signal widths, signedness)
# ----------------------------------------------------------
attr = {
    'x':   {'signed': 1, 'width': 8},   # input
    'acc': {'signed': 1, 'width': 16},  # accumulator output
}

# Optional synthesis / Quartus environment attributes
syn_attr = {
    'QUARTUS_DIR': '/opt/intelFPGA/22.1std/quartus/bin'
}

# ----------------------------------------------------------
# Create Py2ver wrapper and generate testbench / hardware
# ----------------------------------------------------------
p = Py2ver(sum_loop, attr)

sum_tb = p.TB()     # software simulation (executes Python)
sum_hw = p.HW(syn_attr)  # hardware IR / Verilog synthesis


# ----------------------------------------------------------
# Run both TB and HW representations
# ----------------------------------------------------------
tb_result = sum_tb(1)
hw_result = sum_hw(1)

print("TB:", tb_result)
print("HW:", hw_result)
