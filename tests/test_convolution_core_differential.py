def conv3_core(x0, x1, x2, k0, k1, k2):
    """
    Simple 1D 3-tap convolution core:
      y = x0*k0 + x1*k1 + x2*k2
    """
    m0 = x0 * k0
    m1 = x1 * k1
    m2 = x2 * k2
    acc = m0 + m1 + m2
    return acc


def test_conv3_core_python_vs_sim(assert_python_matches_sim):
    attr = {
        "x0": {"signed": 1, "width": 12, "type": "wire"},
        "x1": {"signed": 1, "width": 12, "type": "wire"},
        "x2": {"signed": 1, "width": 12, "type": "wire"},
        "k0": {"signed": 1, "width": 12, "type": "wire"},
        "k1": {"signed": 1, "width": 12, "type": "wire"},
        "k2": {"signed": 1, "width": 12, "type": "wire"},
        "m0": {"signed": 1, "width": 24, "type": "wire"},
        "m1": {"signed": 1, "width": 24, "type": "wire"},
        "m2": {"signed": 1, "width": 24, "type": "wire"},
        "acc": {"signed": 1, "width": 26, "type": "wire"},
    }

    vectors = [
        (0, 0, 0, 1, 1, 1),
        (100, 50, 25, 1, 2, 3),
        (300, -200, 100, 4, -3, 2),
        (-512, 256, -128, -2, 5, -1),
        (1023, 1023, 1023, 1, 0, -1),
        (2047, -2048, 2047, 2, 2, 2),
    ]

    assert_python_matches_sim(conv3_core, attr, vectors)
