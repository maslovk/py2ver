"""A list read at a variable index (the translator's selection tree) as a case
table: a function for numbers, a reg and its always block for signals."""
import random
import re

import pytest

from py2ver import DEFAULT_TEMPLATE_DIR
from py2ver.ast_ir import Assignment, ModuleIR, Port
from py2ver.renderer import Renderer

from icarus import assert_same_bits

_INPUTS = {"i8s": (8, True), "i5u": (5, False), "i3u": (3, False),
           "a": (8, True), "b": (8, False), "c": (16, True), "d": (3, False)}


def _tree(elems, idx, signed_idx):
    """elems[idx] as the translator writes it (visitor._select)."""
    n = len(elems)
    keys = ([(k - n, k) for k in range(n)] if signed_idx else []) + [(k, k) for k in range(n)]

    def tree(part):
        if len(part) == 1:
            return elems[part[0][1]]
        mid = len(part) // 2
        return f"(({idx}) < ({part[mid][0]})) ? ({tree(part[:mid])}) : ({tree(part[mid:])})"
    return f"({tree(keys)})"


def _render(assigns, outputs, inputs=_INPUTS):
    ir = ModuleIR(name="lk", inputs=[Port(n, w, s) for n, (w, s) in inputs.items()],
                  outputs=[Port(n, w, s, "output") for n, (w, s) in outputs.items()],
                  assigns=[Assignment(n, r) for n, r in assigns])
    return Renderer(str(DEFAULT_TEMPLATE_DIR)).render_module(ir)


def test_a_table_read_twice_is_one_function():
    taps = ["38535", "75758", "110401", "141284", "167356"]
    v = _render([("o0", f"(a) + {_tree(taps, 'i3u', False)}"),
                 ("o1", f"(c) * {_tree(taps, 'i5u', False)}"),
                 ("o2", _tree(taps, 'i8s', True))],       # (and Python's negative indices)
                {"o0": (32, True), "o1": (40, True), "o2": (32, True)})
    assert v.count("endfunction") == 2
    assert "function signed [31:0] table1(input [4:0] i);" in v
    assert "assign o0 = a + table1(i3u);" in v and "assign o1 = c * table1(i5u);" in v
    assert "    0: table1 = 38535;" in v and "    default: table1 = 167356;" in v
    assert "function signed [31:0] table2(input signed [7:0] i);" in v
    assert "    -5: table2 = 38535;" in v and "    4:  table2 = 167356;" in v


def test_a_list_of_signals_is_a_case_block():
    xs = {f"xs_{k}": (16, True) for k in range(4)}
    v = _render([("o0", _tree(list(xs), "i5u", False))], {"o0": (16, True)},
                {**_INPUTS, **xs})
    assert re.search(r"reg +signed \[15:0\] xs__at1;", v)
    assert "always @*\n  case (i5u)\n    0: xs__at1 = xs_0;" in v
    assert "    default: xs__at1 = xs_3;" in v
    assert "assign o0 = xs__at1;" in v


def test_short_lists_and_signed_narrow_elements_stay_selections():
    v = _render([("o0", _tree(["a", "b", "d"], "i3u", False)),
                 ("o1", _tree(["a", "c", "a", "c"], "i3u", False))],   # (a: signed, 8 bits of 16)
                {"o0": (8, False), "o1": (16, True)})
    assert "case" not in v


def _element(rng):
    return rng.choice(["a", "b", "c", "d", "0", "5", "70000",
                       "3'd5", "4'sd3", "8'shF0", "16'sd300"])


@pytest.mark.sim
def test_case_tables_compute_the_same_in_icarus(tmp_path):
    """Icarus computes each rendered module output as the selection tree it
    came from, in contexts whose width and sign extend the table's value."""
    rng = random.Random(3)
    contexts = ["{T}", "{T} + (c)", "{T} * (b)", "(a) - {T}", "{T} >>> 2", "{T} / 3",
                "{T} < (a)", "$signed({T}) + (d)", "{T} ^ (c)", "(b) ? {T} : (a)"]
    assigns, outputs = [], {}
    for k in range(120):
        idx = rng.choice(["i8s", "i5u", "i3u"])
        n = rng.randrange(2, 12)
        if rng.random() < 0.4:
            elems = [str(rng.randrange(0, 1 << 20)) for _ in range(n)]
        else:
            elems = [_element(rng) for _ in range(n)]
        tree = _tree(elems, idx, _INPUTS[idx][1])
        assigns.append((f"o{k}", rng.choice(contexts).format(T=tree)))
        outputs[f"o{k}"] = (rng.choice([1, 8, 24, 40]), rng.random() < 0.5)
    v = _render(assigns, outputs)
    assert "function" in v and "always @*" in v
    # (each output of the rendered module against its tree, computed in the testbench)
    body = [f"wire {'signed ' if s else ''}[{w - 1}:0] {n};" for n, (w, s) in outputs.items()]
    body.append("lk dut(" + ", ".join(f".{n}({n})" for n in [*_INPUTS, *outputs]) + ");")
    pairs = [(w, s, r, n) for (n, r), (w, s) in zip(assigns, outputs.values())]
    assert_same_bits(tmp_path, _INPUTS, pairs, sources=v, body=body, samples=3000)
