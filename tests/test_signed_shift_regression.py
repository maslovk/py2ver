def lpf_step_signed_shift(x, y_prev):
    """Reference Python behavior for signed right shift in LPF update."""
    err = x - y_prev
    delta = err >> 3
    y = y_prev + delta
    return y


def test_signed_right_shift_negative_error_regression(assert_python_matches_sim):
    """
    Regression guard for signed shift mismatch:
    Python expects arithmetic right shift for negative `err`, but current
    generated RTL behavior does not match for some vectors.
    """
    attr = {
        "x": {"signed": 1, "width": 12, "type": "wire"},
        "y_prev": {"signed": 1, "width": 16, "type": "wire"},
        "err": {"signed": 1, "width": 16, "type": "wire"},
        "delta": {"signed": 1, "width": 16, "type": "wire"},
        "y": {"signed": 1, "width": 16, "type": "wire"},
    }

    # Includes decreasing steps that force err < 0, where mismatch currently appears.
    vectors = [
        (1024, 0),
        (1024, 128),
        (256, 368),
        (0, 354),
        (-256, 310),
    ]

    assert_python_matches_sim(lpf_step_signed_shift, attr, vectors)
