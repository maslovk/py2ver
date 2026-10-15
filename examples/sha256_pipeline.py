# Four SHA-256 compression rounds, written with small helper functions and
# pipelined into 3 register stages so it meets 50 MHz on the DE0-Nano
# (unpipelined it reaches about 31 MHz; try --syn with and without PIPELINE).
import random

from py2ver import Py2ver

from _run import run

MASK = 0xFFFFFFFF
PIPELINE = 3


def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK


def round_(a, b, c, d, e, f, g, h, w, k):
    s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
    ch = (e & f) ^ ((~e) & g)
    t1 = (h + s1 + ch + k + w) & MASK
    s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
    maj = (a & b) ^ (a & c) ^ (b & c)
    t2 = (s0 + maj) & MASK
    return (t1 + t2) & MASK, a, b, c, (d + t1) & MASK, e, f, g


def sha4(a, b, c, d, e, f, g, h, w0, w1, w2, w3, k0, k1, k2, k3):
    a, b, c, d, e, f, g, h = round_(a, b, c, d, e, f, g, h, w0, k0)
    a, b, c, d, e, f, g, h = round_(a, b, c, d, e, f, g, h, w1, k1)
    a, b, c, d, e, f, g, h = round_(a, b, c, d, e, f, g, h, w2, k2)
    a, b, c, d, e, f, g, h = round_(a, b, c, d, e, f, g, h, w3, k3)
    return a, b, c, d, e, f, g, h


ports = "a b c d e f g h w0 w1 w2 w3 k0 k1 k2 k3".split()
attr = {n: {'signed': 0, 'width': 32} for n in ports}


if __name__ == "__main__":
    rng = random.Random(1)
    vectors = [tuple(rng.getrandbits(32) for _ in ports) for _ in range(20)]
    run(Py2ver(sha4, attr, pipeline=PIPELINE), vectors)
