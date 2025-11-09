from py2ver import Py2ver
import logging
import random

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def foo(arg1, arg2):
    a = arg1 + arg2
    b = arg1 - arg2
    c = arg1 * arg2
    d = arg1 / arg2
    return a, b, c, d

attr = {
    'arg1': {'signed': 0, 'width': 8},
    'arg2': {'signed': 0, 'width': 8},
    'a':    {'signed': 0, 'width': 9},
    'b':    {'signed': 1, 'width': 9},
    'c':    {'signed': 0, 'width': 9},
    'd':    {'signed': 0, 'width': 9},
}

syn_attr = {
    'QUARTUS_DIR': '/opt/intelFPGA/22.1std/quartus/bin',
    # 'UART_PORT': '/dev/ttyUSB0',  # optional override
}

# --------------------------------------------------------------------
# Regression helper
# --------------------------------------------------------------------

def run_regression(func, attr, syn_attr, test_vectors, max_mismatches=10):
    """
    Build once, then run TB and HW on all test_vectors.
    test_vectors: iterable of tuples, e.g. [(5,8), (3,1), ...]
    """
    log = logging.getLogger("regression")

    p = Py2ver(func, attr)
    tb = p.TB()
    hw = p.HW(syn_attr)

    mismatches = []
    for i, vec in enumerate(test_vectors):
        tb_out = tb(*vec)
        hw_out = hw(*vec)
        if tb_out != hw_out:
            log.error("Mismatch #%d for inputs %s: TB=%s, HW=%s",
                      len(mismatches) + 1, vec, tb_out, hw_out)
            mismatches.append((vec, tb_out, hw_out))
            if len(mismatches) >= max_mismatches:
                log.error("Reached max_mismatches=%d, stopping regression.",
                          max_mismatches)
                break
        else:
            log.info("OK  #%d  inputs=%s  TB=HW=%s", i, vec, tb_out)

    log.info("Regression finished: %d tests, %d mismatches",
             len(test_vectors), len(mismatches))

    return mismatches

# --------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Some hand-picked cases
    directed = [
        (0, 1),
        (1, 1),
        (5, 8),
        (4, 4),
        (255, 1),
    ]

    # Plus some random ones within arg widths
    rnd = []
    for _ in range(20):
        a = random.randint(0, 255)
        b = random.randint(1, 255)  # avoid div by zero
        rnd.append((a, b))

    vectors = directed + rnd

    mismatches = run_regression(foo, attr, syn_attr, vectors)

    print("Mismatches:")
    for vec, tb_out, hw_out in mismatches:
        print(f"  inputs={vec}  TB={tb_out}  HW={hw_out}")
