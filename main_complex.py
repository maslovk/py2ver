from py2ver import Py2ver
import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ---------------------------------------------------------------------
# Complex arithmetic demo function
# ---------------------------------------------------------------------
def foo_complex(ar, ai, br, bi):
    """Example function with complex arithmetic for Py2ver."""
    x = complex(ar, ai)
    y = complex(br, bi)

    s = x + y      # sum
    d = x - y      # difference
    p = x * y      # product

    # The FunctionVisitor will expand this into s_re,s_im,d_re,d_im,p_re,p_im
    return s, d, p


# ---------------------------------------------------------------------
# Attribute map for Py2ver HDL synthesis
# ---------------------------------------------------------------------
attr = {
    # Inputs
    'ar': {'signed': 1, 'width': 8},
    'ai': {'signed': 1, 'width': 8},
    'br': {'signed': 1, 'width': 8},
    'bi': {'signed': 1, 'width': 8},

    # Outputs for s = x + y
    's_re': {'signed': 1, 'width': 9},
    's_im': {'signed': 1, 'width': 9},

    # Outputs for d = x - y
    'd_re': {'signed': 1, 'width': 9},
    'd_im': {'signed': 1, 'width': 9},

    # Outputs for p = x * y
    'p_re': {'signed': 1, 'width': 16},
    'p_im': {'signed': 1, 'width': 16},
}

syn_attr = {'QUARTUS_DIR': '/opt/intelFPGA/22.1std/quartus/bin'}


# ---------------------------------------------------------------------
# Instantiate Py2ver objects
# ---------------------------------------------------------------------
p = Py2ver(foo_complex, attr)
foo_tb = p.TB()
foo_hw = p.HW(syn_attr)


# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------
def unpack_complex_outputs(vals):
    """Convert (s_re,s_im,d_re,d_im,p_re,p_im) → (s,d,p) as Python complex."""
    s_re, s_im, d_re, d_im, p_re, p_im = vals
    s = complex(s_re, s_im)
    d = complex(d_re, d_im)
    p = complex(p_re, p_im)
    return s, d, p


def run_case(ar, ai, br, bi):
    """Run one test case and print TB, HW, and pure Python results."""
    x = complex(ar, ai)
    y = complex(br, bi)

    tb_raw = foo_tb(ar, ai, br, bi)
    hw_raw = foo_hw(ar, ai, br, bi)
    tb_c = unpack_complex_outputs(tb_raw)
    hw_c = unpack_complex_outputs(hw_raw)

    # Native Python result (ground truth)
    py_s = x + y
    py_d = x - y
    py_p = x * y

    print(f"\nInputs: x = {x}, y = {y}")
    print(f"Raw Python:  ({py_s}, {py_d}, {py_p})")
    print(f"TB raw:      {tb_raw}")
    print(f"TB complex:  {tb_c}")
    print(f"HW raw:      {hw_raw}")
    print(f"HW complex:  {hw_c}")


# ---------------------------------------------------------------------
# Example run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_case(1, 2, 3, 4)
