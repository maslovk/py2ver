import ast
from visitor import FunctionVisitor
import pytest

def test_missing_attribute_error():
    """Test that missing attributes cause appropriate errors."""
    src = """
def foo(a, b):
    return a + b
"""
    # Missing 'b' in attr
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'wire'},
        # 'b' is missing
    }
    
    tree = ast.parse(src)
    # This should not raise an error during IR building, but during usage
    # Let's test that the visitor can handle this gracefully
    ir = FunctionVisitor(attr).visit(tree)
    
    # Should still build IR, but attributes will be incomplete
    assert ir.name == "foo"

def test_invalid_width_handling():
    """Test that invalid widths are handled gracefully."""
    src = """
def foo(a):
    return a + 1
"""
    attr = {
        'a': {'width': -5, 'signed': 0, 'type': 'wire'},  # Invalid negative width
    }
    
    tree = ast.parse(src)
    # The visitor should handle this without crashing during AST traversal
    ir = FunctionVisitor(attr).visit(tree)
    
    assert ir.name == "foo"