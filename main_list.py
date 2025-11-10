from py2ver import Py2ver
import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Test function using: data = [] ; data.append(...)
# ---------------------------------------------------------------------------

def foo_vec(arg1, arg2):
    data = []
    data.append(arg1 + arg2)
    data.append(arg1 - arg2)
    data.append(arg1 * arg2)
    data.append(arg1 ^ arg2)   # just to have 4 outputs
    return data                # -> data_0, data_1, data_2, data_3


# ---------------------------------------------------------------------------
# Attribute dictionaries
# ---------------------------------------------------------------------------

attr = {
    'arg1': {'signed': 0, 'width': 8},
    'arg2': {'signed': 0, 'width': 8},

    # Base "data" attr – elements data_0..3 will clone this
    'data': {'signed': 0, 'width': 16, 'type': 'wire'},
}

syn_attr = {
    'QUARTUS_DIR': '/opt/intelFPGA/22.1std/quartus/bin'
}

# ---------------------------------------------------------------------------
# py2ver wiring
# ---------------------------------------------------------------------------

p = Py2ver(foo_vec, attr)

foo_vec_tb = p.TB()
foo_vec_hw = p.HW(syn_attr)

g = foo_vec_tb(4, 4)   # simulation result
h = foo_vec_hw(4, 4)   # "hardware" result

print("TB:", g)
print("HW:", h)
