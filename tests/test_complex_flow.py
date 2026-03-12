import ast
from visitor import FunctionVisitor

def unit_if_else(a, b, c):
    if c == 0:
        x = a + b
    else:
        x = a - b
    return x


def unit_for_unroll(a):
    acc = 0
    for i in range(3):
        acc = acc + a
    return acc


def test_if_else_function():
    """Test if/else control flow handling."""
    src = """
def foo(a, b, c):
    if c == 0:
        x = a + b
    else:
        x = a - b
    return x
"""
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'wire'},
        'b': {'width': 8, 'signed': 0, 'type': 'wire'},
        'c': {'width': 8, 'signed': 0, 'type': 'wire'},
        'x': {'width': 9, 'signed': 0, 'type': 'wire'},
    }

    tree = ast.parse(src)
    ir = FunctionVisitor(attr).visit(tree)

    assert ir.name == "foo"
    assert len(ir.assigns) > 0
    # Should have conditional assignments
    assert any('?' in assign.right for assign in ir.assigns)

def test_for_loop_unrolling():
    """Test for loop unrolling functionality."""
    src = """
def foo(a):
    acc = 0
    for i in range(3):
        acc = acc + a
    return acc
"""
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'reg'},
        'acc': {'width': 10, 'signed': 0, 'type': 'reg'},
    }

    tree = ast.parse(src)
    ir = FunctionVisitor(attr).visit(tree)

    assert ir.name == "foo"
    # Should have unrolled assignments for i=0,1,2
    assert len(ir.assigns) >= 3
    # Should have accumulator tracking
    assert any('acc_0' in assign.left for assign in ir.assigns)
    assert any('acc_1' in assign.left for assign in ir.assigns)
    assert any('acc_2' in assign.left for assign in ir.assigns)


def test_if_else_py_vs_sim(assert_python_matches_sim):
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'wire'},
        'b': {'width': 8, 'signed': 0, 'type': 'wire'},
        'c': {'width': 8, 'signed': 0, 'type': 'wire'},
        'x': {'width': 9, 'signed': 0, 'type': 'wire'},
    }
    assert_python_matches_sim(unit_if_else, attr, [(8, 3, 0), (8, 3, 1), (4, 7, 0)])


def test_for_loop_py_vs_sim(assert_python_matches_sim):
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'reg'},
        'acc': {'width': 10, 'signed': 0, 'type': 'reg'},
    }
    assert_python_matches_sim(unit_for_unroll, attr, [(1,), (7,), (15,)])

def test_while_loop_not_supported():
    """Test that while loops raise NotImplementedError."""
    src = """
def foo(a):
    acc = 0
    i = 0
    while i < 3:
        acc = acc + a
        i = i + 1
    return acc
"""
    attr = {
        'a': {'width': 8, 'signed': 0, 'type': 'reg'},
        'acc': {'width': 10, 'signed': 0, 'type': 'reg'},
        'i': {'width': 8, 'signed': 0, 'type': 'reg'},
    }

    tree = ast.parse(src)
    
    # While loops should raise NotImplementedError
    try:
        ir = FunctionVisitor(attr).visit(tree)
        assert False, "Expected NotImplementedError for while loops"
    except NotImplementedError as e:
        assert "while-loops are not supported" in str(e)
