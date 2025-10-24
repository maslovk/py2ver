import ast
from visitor import FunctionVisitor
from renderer import Renderer
from py2ver import DEFAULT_TEMPLATE_DIR

def test_renderer_smoke(tmp_path, toy_src_add_xor, attr_simple):
    ir = FunctionVisitor(attr_simple).visit(ast.parse(toy_src_add_xor))

    r = Renderer(DEFAULT_TEMPLATE_DIR)
    verilog = r.render_module(ir)

    assert "module foo" in verilog
    assert "a" in verilog and "b" in verilog and "x" in verilog and "y" in verilog
    assert "output" in verilog
    assert ("assign x =" in verilog) or ("x <=" in verilog)
    assert ("assign y =" in verilog) or ("y <=" in verilog)

    (tmp_path / "foo.v").write_text(verilog)
