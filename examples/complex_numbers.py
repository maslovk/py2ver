# Complex arithmetic: sum, difference and product of two complex numbers
# built from integer parts. Each complex result is two outputs (s_re, s_im,
# ...); declaring 'p' would set the width of both p_re and p_im.
from py2ver import Py2ver

from _run import run


def foo_complex(ar, ai, br, bi):
    """Example function with complex arithmetic for Py2ver."""
    x = complex(ar, ai)
    y = complex(br, bi)
    s = x + y      # sum
    d = x - y      # difference
    p = x * y      # product (four multipliers)
    return s, d, p


attr = {
    'ar': {'signed': 1, 'width': 8},
    'ai': {'signed': 1, 'width': 8},
    'br': {'signed': 1, 'width': 8},
    'bi': {'signed': 1, 'width': 8},
    's_re': {'signed': 1, 'width': 9},
    's_im': {'signed': 1, 'width': 9},
    'd_re': {'signed': 1, 'width': 9},
    'd_im': {'signed': 1, 'width': 9},
    'p_re': {'signed': 1, 'width': 16},
    'p_im': {'signed': 1, 'width': 16},
}


if __name__ == "__main__":
    run(Py2ver(foo_complex, attr),
        [(1, 2, 3, 4), (-128, 127, -1, -128), (5, -7, 11, 13), (127, 127, 127, 127)])
