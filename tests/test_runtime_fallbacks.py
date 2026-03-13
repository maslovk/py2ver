from pathlib import Path

import py2ver as py2ver_mod
from py2ver import Py2ver


def rt_add_sub(a, b):
    s = a + b
    d = a - b
    return s, d


def test_tb_fun_returns_zero_tuple_when_results_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(py2ver_mod.tb_runner, "test_runner", lambda _top: None)

    attr = {
        "a": {"signed": 0, "width": 8, "type": "wire"},
        "b": {"signed": 0, "width": 8, "type": "wire"},
        "s": {"signed": 0, "width": 9, "type": "wire"},
        "d": {"signed": 1, "width": 9, "type": "wire"},
    }
    p = Py2ver(rt_add_sub, attr)

    out = p.TB()(7, 3)
    assert out == (0, 0)


def test_tb_fun_parses_string_payload_and_applies_signed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def _fake_test_runner(_top):
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        payload = "{'s': 5, 'd': 15}"  # d should become -1 for 4-bit signed output
        import pickle

        with (results_dir / "results.pickle").open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    monkeypatch.setattr(py2ver_mod.tb_runner, "test_runner", _fake_test_runner)

    attr = {
        "a": {"signed": 0, "width": 8, "type": "wire"},
        "b": {"signed": 0, "width": 8, "type": "wire"},
        "s": {"signed": 0, "width": 4, "type": "wire"},
        "d": {"signed": 1, "width": 4, "type": "wire"},
    }
    p = Py2ver(rt_add_sub, attr)

    out = p.TB()(2, 1)
    assert out == (5, -1)


def test_hw_fun_falls_back_to_tb_when_pyserial_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(py2ver_mod.tb_runner, "test_runner", lambda _top: None)

    attr = {
        "a": {"signed": 0, "width": 8, "type": "wire"},
        "b": {"signed": 0, "width": 8, "type": "wire"},
        "s": {"signed": 0, "width": 9, "type": "wire"},
        "d": {"signed": 1, "width": 9, "type": "wire"},
    }
    p = Py2ver(rt_add_sub, attr)

    monkeypatch.setattr(py2ver_mod, "serial", None)
    monkeypatch.setattr(p, "_hw_ok", True)
    monkeypatch.setattr(p, "tb_fun", lambda *_args: (123, 456))

    out = p.hw_fun(9, 2)
    assert out == (123, 456)
