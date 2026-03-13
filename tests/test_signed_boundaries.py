def signed_arith(a, b):
    s = a + b
    d = a - b
    p = a * b
    return s, d, p


def test_signed_overflow_underflow_boundaries(assert_python_matches_sim):
    """Signed edge-case differential test: Python reference vs Icarus simulation."""
    attr = {
        "a": {"signed": 1, "width": 8, "type": "wire"},
        "b": {"signed": 1, "width": 8, "type": "wire"},
        "s": {"signed": 1, "width": 9, "type": "wire"},
        "d": {"signed": 1, "width": 9, "type": "wire"},
        "p": {"signed": 1, "width": 16, "type": "wire"},
    }

    vectors = [
        (-128, -128),
        (-128, -1),
        (-128, 0),
        (-128, 1),
        (-128, 127),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 127),
        (127, -128),
        (127, -1),
        (127, 127),
    ]

    assert_python_matches_sim(signed_arith, attr, vectors)
