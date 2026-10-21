import contextlib
import io
import random
import shutil

import pytest

from py2ver import Py2ver
from py2ver.core import front_end
from py2ver.hwtypes import _namespace
from py2ver.sequencer import sequence

# Loop nests of the same form written once (steps.py): a loop over the
# program's steps, each its template's with its own numbers, lists and tables.

_T = tuple((k * 37) % 23 - 11 for k in range(256))


def layers(x):
    a = [0] * 64
    for i in range(64):
        a[i] = x[i % 16] * 3 + 1
    b = [0] * 64
    for i in range(62):                        # (these three: one form, their own numbers)
        b[i] = (a[i + 2] * 5 - 7) >> 1
    c = [0] * 48
    for i in range(48):
        c[i] = (b[i + 8] * 9 - 2) >> 2
    d = [0] * 64
    for i in range(8):                         # (these two: strides, counts)
        for j in range(8):
            d[i * 8 + j] = a[j * 8 + i] + b[i * 8 + j]
    e = [0] * 32
    for i in range(4):
        for j in range(8):
            e[i * 8 + j] = c[j * 4 + i] + d[i * 16 + j]
    f = [0] * 16
    for i in range(12):                        # (and these two: a table at two offsets)
        acc = 0
        for k in range(16):
            acc = acc + d[i * 4 + k] * _T[k * 16 + i + 1]
        f[i] = acc >> 3
    g = [0] * 8
    for i in range(6):
        acc2 = 0
        for k in range(8):
            acc2 = acc2 + e[i * 4 + k] * _T[k * 8 + i + 3]
        g[i] = acc2 >> 4
    h = [0] * 16
    for i in range(16):
        h[i] = (f[i] * 7 + 3) >> 1
    s = 0
    for i in range(8):
        s = s + g[i] * h[i]
    return s


X16 = {"x": {"signed": 1, "width": 8, "length": 16}}


def test_nests_of_one_form_are_written_once(monkeypatch):
    monkeypatch.setenv("PY2VER_STEPS", "1")          # (a small program: steps all the same)
    fe = front_end(layers, X16)
    info = sequence(fe.tree, fe.attr, _namespace(layers), sdram=0)[1]
    # (the nests and the lists in registers made, b filled first (its loop writes 62
    # of 64): b's and c's loops, d's and e's, f's and g's one template each; f and
    # g, a template's in the SDRAM, filled by loops, with b's)
    assert (info.steps, info.templates) == (14, 9)
    monkeypatch.setenv("PY2VER_STEPS", "0")
    assert sequence(fe.tree, fe.attr, _namespace(layers), sdram=0)[1].steps == 0


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_steps_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PY2VER_SIM", "icarus")
    monkeypatch.setenv("PY2VER_STEPS", "1")
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(layers, X16, sequential=True, sdram=0)
    assert p.schedule.steps == 14
    r = random.Random(4)
    vectors = [([r.randint(-128, 127) for _ in range(16)],) for _ in range(3)]
    assert p.PY_batch()(vectors) == [(layers(*v),) for v in vectors]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)
