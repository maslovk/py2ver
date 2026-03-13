def sha2_round_core(a, b, c, d, e, f, g, h, w0, w1, k0, k1):
    """Two unrolled SHA-256 rounds with fixed W/K inputs per round."""
    m = 0xFFFFFFFF
    s1_0 = (((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^ ((e >> 25) | (e << 7))) & m
    ch0 = ((e & f) ^ ((~e) & g)) & m
    t1_0 = (h + s1_0 + ch0 + k0 + w0) & m
    s0_0 = (((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^ ((a >> 22) | (a << 10))) & m
    maj0 = ((a & b) ^ (a & c) ^ (b & c)) & m
    t2_0 = (s0_0 + maj0) & m

    a1 = (t1_0 + t2_0) & m
    b1 = a & m
    c1 = b & m
    d1 = c & m
    e1 = (d + t1_0) & m
    f1 = e & m
    g1 = f & m
    h1 = g & m

    s1_1 = (((e1 >> 6) | (e1 << 26)) ^ ((e1 >> 11) | (e1 << 21)) ^ ((e1 >> 25) | (e1 << 7))) & m
    ch1 = ((e1 & f1) ^ ((~e1) & g1)) & m
    t1_1 = (h1 + s1_1 + ch1 + k1 + w1) & m
    s0_1 = (((a1 >> 2) | (a1 << 30)) ^ ((a1 >> 13) | (a1 << 19)) ^ ((a1 >> 22) | (a1 << 10))) & m
    maj1 = ((a1 & b1) ^ (a1 & c1) ^ (b1 & c1)) & m
    t2_1 = (s0_1 + maj1) & m

    a2 = (t1_1 + t2_1) & m
    b2 = a1
    c2 = b1
    d2 = c1
    e2 = (d1 + t1_1) & m
    f2 = e1
    g2 = f1
    h2 = g1
    return a2, b2, c2, d2, e2, f2, g2, h2


def test_sha_two_round_core_python_vs_sim(assert_python_matches_sim):
    attr = {
        "a": {"signed": 0, "width": 32, "type": "wire"},
        "b": {"signed": 0, "width": 32, "type": "wire"},
        "c": {"signed": 0, "width": 32, "type": "wire"},
        "d": {"signed": 0, "width": 32, "type": "wire"},
        "e": {"signed": 0, "width": 32, "type": "wire"},
        "f": {"signed": 0, "width": 32, "type": "wire"},
        "g": {"signed": 0, "width": 32, "type": "wire"},
        "h": {"signed": 0, "width": 32, "type": "wire"},
        "w0": {"signed": 0, "width": 32, "type": "wire"},
        "w1": {"signed": 0, "width": 32, "type": "wire"},
        "k0": {"signed": 0, "width": 32, "type": "wire"},
        "k1": {"signed": 0, "width": 32, "type": "wire"},
        "s1_0": {"signed": 0, "width": 32, "type": "wire"},
        "ch0": {"signed": 0, "width": 32, "type": "wire"},
        "t1_0": {"signed": 0, "width": 32, "type": "wire"},
        "s0_0": {"signed": 0, "width": 32, "type": "wire"},
        "maj0": {"signed": 0, "width": 32, "type": "wire"},
        "t2_0": {"signed": 0, "width": 32, "type": "wire"},
        "a1": {"signed": 0, "width": 32, "type": "wire"},
        "b1": {"signed": 0, "width": 32, "type": "wire"},
        "c1": {"signed": 0, "width": 32, "type": "wire"},
        "d1": {"signed": 0, "width": 32, "type": "wire"},
        "e1": {"signed": 0, "width": 32, "type": "wire"},
        "f1": {"signed": 0, "width": 32, "type": "wire"},
        "g1": {"signed": 0, "width": 32, "type": "wire"},
        "h1": {"signed": 0, "width": 32, "type": "wire"},
        "s1_1": {"signed": 0, "width": 32, "type": "wire"},
        "ch1": {"signed": 0, "width": 32, "type": "wire"},
        "t1_1": {"signed": 0, "width": 32, "type": "wire"},
        "s0_1": {"signed": 0, "width": 32, "type": "wire"},
        "maj1": {"signed": 0, "width": 32, "type": "wire"},
        "t2_1": {"signed": 0, "width": 32, "type": "wire"},
        "a2": {"signed": 0, "width": 32, "type": "wire"},
        "b2": {"signed": 0, "width": 32, "type": "wire"},
        "c2": {"signed": 0, "width": 32, "type": "wire"},
        "d2": {"signed": 0, "width": 32, "type": "wire"},
        "e2": {"signed": 0, "width": 32, "type": "wire"},
        "f2": {"signed": 0, "width": 32, "type": "wire"},
        "g2": {"signed": 0, "width": 32, "type": "wire"},
        "h2": {"signed": 0, "width": 32, "type": "wire"},
    }

    vectors = [
        (
            0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
            0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
            0x61626380, 0x00000000, 0x428A2F98, 0x71374491,
        ),
        (
            0x01234567, 0x89ABCDEF, 0xFEDCBA98, 0x76543210,
            0x0F1E2D3C, 0x4B5A6978, 0x88776655, 0x44332211,
            0x11111111, 0x22222222, 0xB5C0FBCF, 0xE9B5DBA5,
        ),
    ]

    assert_python_matches_sim(sha2_round_core, attr, vectors)
