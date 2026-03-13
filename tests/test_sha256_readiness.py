def sha_w16_step(w0, w1, w9, w14):
    m = 0xFFFFFFFF
    s1 = ((w14 >> 17) | (w14 << 15)) ^ ((w14 >> 19) | (w14 << 13)) ^ (w14 >> 10)
    s0 = ((w1 >> 7) | (w1 << 25)) ^ ((w1 >> 18) | (w1 << 14)) ^ (w1 >> 3)
    w16 = (s1 + w9 + s0 + w0) & m
    return w16


def sha_w16_w17_step(w0, w1, w2, w9, w10, w14, w15):
    m = 0xFFFFFFFF
    s1_14 = ((w14 >> 17) | (w14 << 15)) ^ ((w14 >> 19) | (w14 << 13)) ^ (w14 >> 10)
    s0_1 = ((w1 >> 7) | (w1 << 25)) ^ ((w1 >> 18) | (w1 << 14)) ^ (w1 >> 3)
    w16 = (s1_14 + w9 + s0_1 + w0) & m

    s1_15 = ((w15 >> 17) | (w15 << 15)) ^ ((w15 >> 19) | (w15 << 13)) ^ (w15 >> 10)
    s0_2 = ((w2 >> 7) | (w2 << 25)) ^ ((w2 >> 18) | (w2 << 14)) ^ (w2 >> 3)
    w17 = (s1_15 + w10 + s0_2 + w1) & m
    return w16, w17


def sha4_round_core(a, b, c, d, e, f, g, h, w0, w1, w2, w3, k0, k1, k2, k3):
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

    s1_2 = (((e2 >> 6) | (e2 << 26)) ^ ((e2 >> 11) | (e2 << 21)) ^ ((e2 >> 25) | (e2 << 7))) & m
    ch2 = ((e2 & f2) ^ ((~e2) & g2)) & m
    t1_2 = (h2 + s1_2 + ch2 + k2 + w2) & m
    s0_2 = (((a2 >> 2) | (a2 << 30)) ^ ((a2 >> 13) | (a2 << 19)) ^ ((a2 >> 22) | (a2 << 10))) & m
    maj2 = ((a2 & b2) ^ (a2 & c2) ^ (b2 & c2)) & m
    t2_2 = (s0_2 + maj2) & m
    a3 = (t1_2 + t2_2) & m
    b3 = a2
    c3 = b2
    d3 = c2
    e3 = (d2 + t1_2) & m
    f3 = e2
    g3 = f2
    h3 = g2

    s1_3 = (((e3 >> 6) | (e3 << 26)) ^ ((e3 >> 11) | (e3 << 21)) ^ ((e3 >> 25) | (e3 << 7))) & m
    ch3 = ((e3 & f3) ^ ((~e3) & g3)) & m
    t1_3 = (h3 + s1_3 + ch3 + k3 + w3) & m
    s0_3 = (((a3 >> 2) | (a3 << 30)) ^ ((a3 >> 13) | (a3 << 19)) ^ ((a3 >> 22) | (a3 << 10))) & m
    maj3 = ((a3 & b3) ^ (a3 & c3) ^ (b3 & c3)) & m
    t2_3 = (s0_3 + maj3) & m
    a4 = (t1_3 + t2_3) & m
    b4 = a3
    c4 = b3
    d4 = c3
    e4 = (d3 + t1_3) & m
    f4 = e3
    g4 = f3
    h4 = g3
    return a4, b4, c4, d4, e4, f4, g4, h4


def test_sha_schedule_w16_known_and_differential(assert_python_matches_sim):
    assert sha_w16_step(0x61626380, 0x00000000, 0x00000000, 0x00000000) == 0x61626380

    attr = {
        "w0": {"signed": 0, "width": 32, "type": "wire"},
        "w1": {"signed": 0, "width": 32, "type": "wire"},
        "w9": {"signed": 0, "width": 32, "type": "wire"},
        "w14": {"signed": 0, "width": 32, "type": "wire"},
        "w16": {"signed": 0, "width": 32, "type": "wire"},
    }
    vectors = [
        (0x61626380, 0x00000000, 0x00000000, 0x00000000),
        (0x00000001, 0x00000002, 0x00000003, 0x00000004),
        (0xFFFFFFFF, 0x12345678, 0x89ABCDEF, 0x0F1E2D3C),
    ]
    assert_python_matches_sim(sha_w16_step, attr, vectors)


def test_sha_schedule_w16_w17_known_and_differential(assert_python_matches_sim):
    w16, w17 = sha_w16_w17_step(0x61626380, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000018)
    assert w16 == 0x61626380
    assert w17 == 0x000F0000

    attr = {
        "w0": {"signed": 0, "width": 32, "type": "wire"},
        "w1": {"signed": 0, "width": 32, "type": "wire"},
        "w2": {"signed": 0, "width": 32, "type": "wire"},
        "w9": {"signed": 0, "width": 32, "type": "wire"},
        "w10": {"signed": 0, "width": 32, "type": "wire"},
        "w14": {"signed": 0, "width": 32, "type": "wire"},
        "w15": {"signed": 0, "width": 32, "type": "wire"},
        "w16": {"signed": 0, "width": 32, "type": "wire"},
        "w17": {"signed": 0, "width": 32, "type": "wire"},
    }
    vectors = [
        (0x61626380, 0, 0, 0, 0, 0, 0x18),
        (0x00000001, 0x00000002, 0x00000003, 0x00000009, 0x0000000A, 0x0000000E, 0x0000000F),
    ]
    assert_python_matches_sim(sha_w16_w17_step, attr, vectors)


def test_sha_four_round_core_python_vs_sim(assert_python_matches_sim):
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
        "w2": {"signed": 0, "width": 32, "type": "wire"},
        "w3": {"signed": 0, "width": 32, "type": "wire"},
        "k0": {"signed": 0, "width": 32, "type": "wire"},
        "k1": {"signed": 0, "width": 32, "type": "wire"},
        "k2": {"signed": 0, "width": 32, "type": "wire"},
        "k3": {"signed": 0, "width": 32, "type": "wire"},
        "a4": {"signed": 0, "width": 32, "type": "wire"},
        "b4": {"signed": 0, "width": 32, "type": "wire"},
        "c4": {"signed": 0, "width": 32, "type": "wire"},
        "d4": {"signed": 0, "width": 32, "type": "wire"},
        "e4": {"signed": 0, "width": 32, "type": "wire"},
        "f4": {"signed": 0, "width": 32, "type": "wire"},
        "g4": {"signed": 0, "width": 32, "type": "wire"},
        "h4": {"signed": 0, "width": 32, "type": "wire"},
    }

    vectors = [
        (
            0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
            0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
            0x61626380, 0x00000000, 0x00000000, 0x00000000,
            0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
        ),
        (
            0x01234567, 0x89ABCDEF, 0xFEDCBA98, 0x76543210,
            0x0F1E2D3C, 0x4B5A6978, 0x88776655, 0x44332211,
            0x11111111, 0x22222222, 0x33333333, 0x44444444,
            0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
        ),
    ]
    assert_python_matches_sim(sha4_round_core, attr, vectors)
