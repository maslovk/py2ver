def lpf_step(x, y_prev):
    """
    First-order fixed-point low-pass filter step:
      y[n] = y[n-1] + ((x[n] - y[n-1]) >> 3)
    """
    err = x - y_prev
    delta = err >> 3
    y = y_prev + delta
    return y


def _to_signed(value, width):
    mask = (1 << width) - 1
    value &= mask
    sign_bit = 1 << (width - 1)
    return (value ^ sign_bit) - sign_bit


def test_fixed_point_lpf_sequence_python_vs_sim(assert_python_matches_sim):
    """Run a small LPF sequence and compare Python and Icarus outputs per sample."""
    attr = {
        "x": {"signed": 1, "width": 12, "type": "wire"},
        "y_prev": {"signed": 1, "width": 16, "type": "wire"},
        "err": {"signed": 1, "width": 16, "type": "wire"},
        "delta": {"signed": 1, "width": 16, "type": "wire"},
        "y": {"signed": 1, "width": 16, "type": "wire"},
    }

    # Simple step-like and settling stimulus.
    x_seq = [0, 0, 0, 256, 256, 512, 512, 768, 768, 1024, 1024, 1024]

    # Build vectors as (x, y_prev) using Python-side state progression.
    vectors = []
    y_state = 0
    for x in x_seq:
        vectors.append((x, y_state))
        y_state = _to_signed(lpf_step(x, y_state), 16)

    assert_python_matches_sim(lpf_step, attr, vectors)
