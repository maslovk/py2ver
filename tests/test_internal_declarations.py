from ast_ir import Assignment, ModuleIR, Port
from visitor import emit_internal_declarations


def test_emit_internal_declarations_skips_ports_and_formats_width_sign_type():
    ir = ModuleIR(
        name="foo",
        inputs=[Port(name="a", width=8, signed=False, kind="input")],
        outputs=[Port(name="y", width=9, signed=True, kind="output")],
        assigns=[
            Assignment(left="tmp_w", right="a + 1", is_reg=False),
            Assignment(left="tmp_r", right="tmp_w", is_reg=True),
            Assignment(left="y", right="tmp_r", is_reg=False),
        ],
    )

    attr = {
        "tmp_w": {"width": 8, "signed": 0, "type": "wire"},
        "tmp_r": {"width": 12, "signed": 1, "type": "reg"},
        # ports should not be emitted even if present
        "a": {"width": 8, "signed": 0, "type": "wire"},
        "y": {"width": 9, "signed": 1, "type": "wire"},
    }

    decls = emit_internal_declarations(ir, attr)

    assert "tmp_w" in decls
    assert "wire [7:0] tmp_w;" in decls

    assert "tmp_r" in decls
    assert "reg signed [11:0] tmp_r;" in decls

    assert " a;" not in decls
    assert " y;" not in decls
