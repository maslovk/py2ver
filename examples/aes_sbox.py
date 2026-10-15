# AES SubWord: four lookups in the 256-entry S-box. A module-level table
# indexed by a variable becomes a ROM (built from logic: about 315 logic
# elements per S-box on the DE0-Nano's Cyclone IV, at ~150 MHz).
import random

from py2ver import Py2ver

from _run import run


def _gf_mul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = ((a << 1) ^ 0x11B) if a & 0x80 else (a << 1)
        b >>= 1
    return r


def _make_sbox():
    out = []
    for x in range(256):
        inv = next((y for y in range(1, 256) if _gf_mul(x, y) == 1), 0) if x else 0
        s = inv
        for i in range(1, 5):
            s ^= ((inv << i) | (inv >> (8 - i))) & 0xFF
        out.append(s ^ 0x63)
    return tuple(out)


SBOX = _make_sbox()   # computed in Python, used in hardware as a table


def sub_word(w):
    out = 0
    for k in range(4):
        out |= SBOX[(w >> (8 * k)) & 0xFF] << (8 * k)
    return out


attr = {'w': {'signed': 0, 'width': 32}}


if __name__ == "__main__":
    rng = random.Random(0)
    run(Py2ver(sub_word, attr),
        [(0x00000000,), (0x53535353,), (0xFFFFFFFF,)] + [(rng.getrandbits(32),) for _ in range(20)])
