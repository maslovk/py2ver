# A 256-bin histogram kept in block RAM.
#
# 'memory': 'ram' keeps the list as a RAM array: the hardware reads and
# writes one element per sample (two clocks each: read, then compute and
# write). In Quartus this is 4096 bits of block RAM and ~60 logic elements;
# the same list as registers ('next' instead of 'memory') needs ~11,000.
import random

from py2ver import Py2ver

from _run import run


def histogram(bin_, counts):
    c = counts[bin_] + 1
    counts[bin_] = c          # updated in place: a RAM write
    return c                  # the bin's new count


attr = {
    'bin_':   {'signed': 0, 'width': 8},
    'counts': {'signed': 0, 'width': 16, 'length': 256, 'memory': 'ram'},
}


if __name__ == "__main__":
    rng = random.Random(0)
    samples = [rng.randint(0, 255) for _ in range(30)] + [7] * 5
    run(Py2ver(histogram, attr), samples, stream=True)
