import ast
from visitor import FunctionVisitor

def test_complex_number_support():
    """Test complex number operations."""
    src = """
def foo(a, b):
    z = complex(a, b)
    return z
"""
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'wire'},
        'b': {'width': 8, 'signed': 0, 'type': 'wire'},
        'z': {'width': 8, 'signed': 0, 'type': 'wire'},
    }
    
    tree = ast.parse(src)
    ir = FunctionVisitor(attr).visit(tree)
    
    assert ir.name == "foo"
    # Should have complex variable with real and imaginary components
    assert len(ir.assigns) >= 2  # real and imaginary assignments

def test_abs_complex():
    """Test abs() function with complex numbers."""
    src = """
def foo(a, b):
    z = complex(a, b)
    mag = abs(z)
    return mag
"""
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'wire'},
        'b': {'width': 8, 'signed': 0, 'type': 'wire'},
        'z': {'width': 8, 'signed': 0, 'type': 'wire'},
        'mag': {'width': 16, 'signed': 0, 'type': 'wire'},
    }
    
    tree = ast.parse(src)
    ir = FunctionVisitor(attr).visit(tree)
    
    assert ir.name == "foo"
    # Should have complex abs expression
    assert any('$sqrt' in assign.right for assign in ir.assigns)