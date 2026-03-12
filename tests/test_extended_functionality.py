import ast
from visitor import FunctionVisitor
import pytest

def unit_nested_control(a, b):
    result = 0
    for i in range(2):
        if a > b:
            result = i + 1
        else:
            result = i
    return result


def unit_boolean_ops(a, b, c):
    if a and b:
        result = c
    elif a or b:
        result = 0
    else:
        result = -1
    return result


def unit_var_reassign(a):
    result = a
    result = a + 1
    result = a + 2
    return result


def unit_complex_expr(a, b, c):
    result = (a + b) * c
    return result


def test_nested_control_flow():
    """Test nested control flow constructs."""
    src = """
def foo(a, b):
    result = 0
    for i in range(2):
        if a > b:
            result = result + i
        else:
            result = result - i
    return result
"""
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'wire'},
        'b': {'width': 8, 'signed': 0, 'type': 'wire'},
        'result': {'width': 10, 'signed': 0, 'type': 'reg'},
        'i': {'width': 8, 'signed': 0, 'type': 'reg'},
    }

    tree = ast.parse(src)
    ir = FunctionVisitor(attr).visit(tree)

    assert ir.name == "foo"
    # Should have nested control flow with conditional assignments
    # The actual count may vary based on implementation, but should have assignments
    assert len(ir.assigns) >= 1  # at least one assignment
    # Should have conditional execution (ternary operator)
    assert any('?' in assign.right for assign in ir.assigns)


def test_nested_control_flow_py_vs_sim(assert_python_matches_sim):
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'wire'},
        'b': {'width': 8, 'signed': 0, 'type': 'wire'},
        'result': {'width': 10, 'signed': 0, 'type': 'reg'},
        'i': {'width': 8, 'signed': 0, 'type': 'wire'},
    }
    assert_python_matches_sim(unit_nested_control, attr, [(1, 2), (5, 1), (9, 9)])

def test_boolean_operations():
    """Test boolean operations in conditions."""
    src = """
def foo(a, b, c):
    if a and b:
        result = c
    elif a or b:
        result = 0
    else:
        result = -1
    return result
"""
    attr = {
        'a': {'width': 1, 'signed': 0, 'type': 'wire'},
        'b': {'width': 1, 'signed': 0, 'type': 'wire'},
        'c': {'width': 8, 'signed': 0, 'type': 'wire'},
        'result': {'width': 8, 'signed': 0, 'type': 'reg'},
    }

    tree = ast.parse(src)
    ir = FunctionVisitor(attr).visit(tree)

    assert ir.name == "foo"
    # Should handle boolean operations in conditionals
    # The implementation uses ternary operators for if/elif/else
    assert len(ir.assigns) >= 1  # single assignment with conditional logic


def test_boolean_operations_py_vs_sim(assert_python_matches_sim):
    attr = {
        'a': {'width': 1, 'signed': 0, 'type': 'wire'},
        'b': {'width': 1, 'signed': 0, 'type': 'wire'},
        'c': {'width': 8, 'signed': 0, 'type': 'wire'},
        'result': {'width': 8, 'signed': 0, 'type': 'reg'},
    }
    assert_python_matches_sim(unit_boolean_ops, attr, [(0, 0, 9), (1, 0, 7), (1, 1, 12)])

def test_variable_reassignment():
    """Test that variables can be reassigned multiple times."""
    src = """
def foo(a):
    result = a
    result = a + 1
    result = a + 2
    return result
"""
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'wire'},
        'result': {'width': 9, 'signed': 0, 'type': 'reg'},
    }

    tree = ast.parse(src)
    ir = FunctionVisitor(attr).visit(tree)

    assert ir.name == "foo"
    # Should handle multiple assignments to the same variable
    assert len(ir.assigns) >= 1  # at least one assignment


def test_variable_reassignment_py_vs_sim(assert_python_matches_sim):
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'wire'},
        'result': {'width': 9, 'signed': 0, 'type': 'reg'},
    }
    assert_python_matches_sim(unit_var_reassign, attr, [(0,), (5,), (31,)])

def test_complex_expressions():
    """Test compound expressions with multiple operators."""
    src = """
def foo(a, b, c):
    result = (a + b) * c
    return result
"""
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'wire'},
        'b': {'width': 8, 'signed': 0, 'type': 'wire'},
        'c': {'width': 8, 'signed': 0, 'type': 'wire'},
        'result': {'width': 16, 'signed': 0, 'type': 'reg'},
    }

    tree = ast.parse(src)
    ir = FunctionVisitor(attr).visit(tree)

    assert ir.name == "foo"
    # Should handle complex expressions with multiple operators
    assert len(ir.assigns) >= 1  # single assignment with complex expression


def test_complex_expressions_py_vs_sim(assert_python_matches_sim):
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'wire'},
        'b': {'width': 8, 'signed': 0, 'type': 'wire'},
        'c': {'width': 8, 'signed': 0, 'type': 'wire'},
        'result': {'width': 16, 'signed': 0, 'type': 'reg'},
    }
    assert_python_matches_sim(unit_complex_expr, attr, [(1, 2, 3), (10, 5, 2), (7, 9, 4)])


def test_modulo_operations_not_supported():
    """Test that modulo operations raise NotImplementedError."""
    src = """
def foo(a, b):
    result = a % b  # Modulo operation
    return result
"""
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'wire'},
        'b': {'width': 8, 'signed': 0, 'type': 'wire'},
        'result': {'width': 16, 'signed': 0, 'type': 'reg'},
    }

    tree = ast.parse(src)
    
    # Modulo operations should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="Unsupported binary op.*Mod"):
        FunctionVisitor(attr).visit(tree)

def test_list_operations_not_supported():
    """Test that list literals raise NotImplementedError."""
    src = """
def foo(a):
    my_list = [1, 2, 3]  # List literal
    result = my_list[0]  # Indexing
    return result
"""
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'wire'},
        'result': {'width': 8, 'signed': 0, 'type': 'reg'},
    }

    tree = ast.parse(src)
    
    # List literals should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="Non-empty list literals are not supported"):
        FunctionVisitor(attr).visit(tree)
