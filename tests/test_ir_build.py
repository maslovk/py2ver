import ast
from visitor import FunctionVisitor

def unit_add_xor(a, b):
    x = (a & 0xF) + (b & 0xF)
    y = (a ^ b) << 1
    return x, y


def test_ir_from_simple_function(toy_src_add_xor, attr_simple):
    tree = ast.parse(toy_src_add_xor)
    ir = FunctionVisitor(attr_simple).visit(tree)

    assert ir.name == "foo"

    in_names = [p.name for p in ir.inputs if p.kind != "clk"]
    in_widths = [p.width for p in ir.inputs if p.kind != "clk"]
    assert in_names == ["a", "b"]
    assert in_widths == [8, 8]

    out_names = [p.name for p in ir.outputs]
    out_widths = [p.width for p in ir.outputs]
    assert out_names == ["x", "y"]
    assert out_widths == [8, 9]

    assigns = {(a.left, a.right, a.is_reg) for a in ir.assigns}
    assert ("x", "((a) & (15)) + ((b) & (15))", False) in assigns
    assert ("y", "((a) ^ (b)) << (1)", False) in assigns

    assert ir.has_clk is False


def test_ir_from_simple_function_py_vs_sim(attr_simple, assert_python_matches_sim):
    assert_python_matches_sim(unit_add_xor, attr_simple, [(1, 2), (15, 7), (8, 3)])
