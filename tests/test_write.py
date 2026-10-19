"""Py2ver(..., write=False): the Verilog in memory (verilog, hdl_files), hdl/
written only when a simulation or a project needs it, or by write_hdl()."""
import pytest

from py2ver import Py2ver, auto_fixed

import test_stages

I8 = {"width": 8, "signed": 0}


def mac(a, b, c):
    return a * b + c


def test_nothing_is_written_until_asked(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(mac, {"a": I8, "b": I8, "c": I8}, write=False)
    assert not (tmp_path / "hdl").exists()
    assert "module mac" in p.verilog and p.hdl_files == {"main.v": p.verilog}
    p.write_hdl()
    assert (tmp_path / "hdl" / "main.v").read_text() == p.verilog


def test_written_now_by_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(mac, {"a": I8, "b": I8, "c": I8})
    assert (tmp_path / "hdl" / "main.v").read_text() == p.verilog
    assert Py2ver(mac, {"a": I8, "b": I8, "c": I8}, write=False).verilog == p.verilog


TABLE = tuple((k * 37) % 251 for k in range(5000))


def table_mac(i, x):
    acc = 0
    for k in range(4):
        acc += TABLE[i + k] * x
    return acc


TABLE_ATTR = {"i": {"width": 12, "signed": 0}, "x": I8, "acc": {"width": 24, "signed": 0}}


@pytest.mark.sim
def test_a_large_tables_file_is_written_for_the_simulation(monkeypatch, tmp_path):
    """A table read from a file ($readmemh) is in hdl_files too, and on disk
    when the simulation needs it."""
    monkeypatch.chdir(tmp_path)
    p = Py2ver(table_mac, TABLE_ATTR, sequential=True, write=False)
    hexes = [n for n in p.hdl_files if n.endswith(".hex")]
    assert hexes and f'$readmemh("{tmp_path / "hdl" / hexes[0]}"' in p.verilog
    assert not (tmp_path / "hdl").exists()
    samples = [(i, x) for i in (0, 7, 4000) for x in (1, 200)]
    assert p.TB_stream()(samples) == p.PY_stream()(samples)
    assert (tmp_path / "hdl" / hexes[0]).read_text() == p.hdl_files[hexes[0]]


def test_stages_are_kept_in_memory_too(monkeypatch, tmp_path):
    formats = auto_fixed(test_stages.receiver, test_stages.BLOCKS, test_stages.STATE, width=16,
                         max_error=1e-3, round=True)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    p = Py2ver(test_stages.receiver, formats, stages=True, write=False)
    assert not (tmp_path / "a" / "hdl").exists()
    assert "module receiver" in p.hdl_files["main.v"]           # (the fused core)
    assert "module receiver_stages" in p.hdl_files["receiver_stages.v"]
    assert {f.name for f in p._stages["files"]} <= set(p.hdl_files)
    monkeypatch.chdir(tmp_path / "b")
    Py2ver(test_stages.receiver, formats, stages=True)          # (written as before)
    assert {f.name: f.read_text() for f in (tmp_path / "b" / "hdl").iterdir()} == p.hdl_files
