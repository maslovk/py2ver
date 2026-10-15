"""
Shared runner for the examples: check py2ver's hardware against Python.

    python examples/basic.py               # simulation (cocotb + Icarus)
    python examples/basic.py --board-sim   # + simulated DE0-Nano over its UART
    python examples/basic.py --syn         # + Quartus area/timing (needs Quartus)
    python examples/basic.py --board       # + a connected DE0-Nano (Quartus + board)

Quartus is looked up in $QUARTUS_DIR (default /opt/intelFPGA/22.1std/quartus/bin),
the serial port in $UART_PORT (default /dev/ttyUSB0). Exits non-zero on any
mismatch.
"""

import argparse
import logging
import os
import sys

DEFAULT_QUARTUS_DIR = "/opt/intelFPGA/22.1std/quartus/bin"


def _args():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--board-sim", action="store_true",
                    help="also run on the simulated DE0-Nano (real UART timing, slower)")
    ap.add_argument("--syn", action="store_true", help="report area and timing with Quartus")
    ap.add_argument("--board", action="store_true", help="also run on a connected DE0-Nano")
    ap.add_argument("-v", "--verbose", action="store_true", help="show py2ver's log")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args


def _syn_attr():
    return {
        "QUARTUS_DIR": os.environ.get("QUARTUS_DIR", DEFAULT_QUARTUS_DIR),
        "UART_PORT": os.environ.get("UART_PORT", "/dev/ttyUSB0"),
    }


def _report(label, got, expected, vectors):
    bad = [(v, g, e) for v, g, e in zip(vectors, got, expected) if g != e]
    status = "OK" if not bad and len(got) == len(expected) else "MISMATCH"
    print(f"{label:14s} {status}  ({len(vectors)} vectors)")
    for v, g, e in bad[:5]:
        print(f"    inputs={v}  got={g}  python={e}")
    return not bad and len(got) == len(expected)


def run(p, vectors, stream=False):
    """Run vectors (samples, if stream=True) through every requested target."""
    args = _args()
    vectors = [v if isinstance(v, tuple) else (v,) for v in vectors]
    if stream:
        expected = p.PY_stream()(vectors)
        sim = p.TB_stream()(vectors)
    else:
        expected = p.PY_batch()(vectors)
        sim = p.TB_batch()(vectors)

    for v, e in list(zip(vectors, expected))[:4]:
        print(f"  {p.t_name}{v} -> {e}")
    ok = _report("simulation", sim, expected, vectors)

    if args.board_sim:
        ok &= _report("virtual board", p.HW_sim()(vectors), expected, vectors)
    if args.syn:
        print(p.SYN(_syn_attr()))
    if args.board:
        hw = p.HW(_syn_attr())
        ok &= _report("board", [hw(*v) for v in vectors], expected, vectors)

    sys.exit(0 if ok else 1)
