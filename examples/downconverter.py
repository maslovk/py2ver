# A digital downconverter: a real signal times a complex oscillator.
#
# Ordinary float Python with complex numbers: the oscillator z is complex
# state, rotated by a constant every sample. 'complex': True makes z a
# complex value in hardware (z_re / z_im); the formats are fixed point.
import cmath
import math

from py2ver import Py2ver

from _run import run

ROT = cmath.exp(-2j * math.pi * 0.125)      # 1/8 of the sample rate


def downconvert(x, z):
    y = x * z
    zn = z * ROT
    return y, zn


attr = {
    'x': {'signed': 1, 'width': 12, 'frac': 11},
    'z': {'signed': 1, 'width': 18, 'frac': 16, 'complex': True, 'next': 'zn', 'init': 1,
          'round': True},
    'y': {'signed': 1, 'width': 14, 'frac': 12},
}


if __name__ == "__main__":
    tone = [0.9 * math.cos(2 * math.pi * 0.125 * n) for n in range(24)]   # moved to 0 Hz
    run(Py2ver(downconvert, attr), tone, stream=True)
