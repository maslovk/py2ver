def fir7_cheby_step(x, d1, d2, d3, d4, d5, d6):
    """
    7-tap symmetric FIR (Chebyshev-style fixed coefficients, Q6 scaling):
      h = [2, 8, 14, 16, 14, 8, 2] / 64
    State is provided explicitly as delayed samples d1..d6.
    Returns current output and next-state delays.
    """
    acc = (
        (x * 2)
        + (d1 * 8)
        + (d2 * 14)
        + (d3 * 16)
        + (d4 * 14)
        + (d5 * 8)
        + (d6 * 2)
    )
    y = acc >> 6

    nd1 = x
    nd2 = d1
    nd3 = d2
    nd4 = d3
    nd5 = d4
    nd6 = d5
    return y, nd1, nd2, nd3, nd4, nd5, nd6


def test_fir7_chebyshev_sequence_python_vs_sim(assert_python_matches_sim):
    attr = {
        "x": {"signed": 1, "width": 16, "type": "wire"},
        "d1": {"signed": 1, "width": 16, "type": "wire"},
        "d2": {"signed": 1, "width": 16, "type": "wire"},
        "d3": {"signed": 1, "width": 16, "type": "wire"},
        "d4": {"signed": 1, "width": 16, "type": "wire"},
        "d5": {"signed": 1, "width": 16, "type": "wire"},
        "d6": {"signed": 1, "width": 16, "type": "wire"},
        "acc": {"signed": 1, "width": 32, "type": "wire"},
        "y": {"signed": 1, "width": 16, "type": "wire"},
        "nd1": {"signed": 1, "width": 16, "type": "wire"},
        "nd2": {"signed": 1, "width": 16, "type": "wire"},
        "nd3": {"signed": 1, "width": 16, "type": "wire"},
        "nd4": {"signed": 1, "width": 16, "type": "wire"},
        "nd5": {"signed": 1, "width": 16, "type": "wire"},
        "nd6": {"signed": 1, "width": 16, "type": "wire"},
    }

    # Stimulus: impulse, positive step, negative step.
    x_seq = [0, 0, 1000, 0, 0, 0, 0, 0, 0, 800, 800, 800, 800, 0, -700, -700, -700, 0]

    d = [0, 0, 0, 0, 0, 0]
    vectors = []
    for x in x_seq:
        vectors.append((x, d[0], d[1], d[2], d[3], d[4], d[5]))
        _, nd1, nd2, nd3, nd4, nd5, nd6 = fir7_cheby_step(x, d[0], d[1], d[2], d[3], d[4], d[5])
        d = [nd1, nd2, nd3, nd4, nd5, nd6]

    assert_python_matches_sim(fir7_cheby_step, attr, vectors)
