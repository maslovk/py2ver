from pathlib import Path

import tb_runner


class _FakeRunner:
    def __init__(self):
        self.build_calls = []
        self.test_calls = []

    def build(self, **kwargs):
        self.build_calls.append(kwargs)

    def test(self, **kwargs):
        self.test_calls.append(kwargs)


def test_test_runner_verilog_defaults(monkeypatch, tmp_path):
    fake = _FakeRunner()

    # Force project root/build dir into a temporary workspace
    monkeypatch.setattr(tb_runner, "Path", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(tb_runner, "get_runner", lambda sim: fake)

    monkeypatch.delenv("HDL_TOPLEVEL_LANG", raising=False)
    monkeypatch.delenv("SIM", raising=False)

    proj_path = tmp_path.parent  # Path(__file__).resolve().parent in test_runner
    build_dir = proj_path / "sim_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "stale.txt").write_text("stale", encoding="utf-8")

    tb_runner.test_runner("foo")

    assert len(fake.build_calls) == 1
    assert len(fake.test_calls) == 1

    build_kwargs = fake.build_calls[0]
    assert build_kwargs["hdl_toplevel"] == "foo"
    assert build_kwargs["always"] is True
    assert build_kwargs["build_args"] == []
    assert build_kwargs["build_dir"] == str(proj_path / "sim_build")
    assert build_kwargs["sources"] == [proj_path / "hdl" / "main.v"]

    test_kwargs = fake.test_calls[0]
    assert test_kwargs["hdl_toplevel"] == "foo"
    assert test_kwargs["test_module"] == "tb"
    assert test_kwargs["build_dir"] == str(proj_path / "sim_build")
    assert test_kwargs["test_args"] == []


def test_test_runner_vhdl_xcelium_args(monkeypatch, tmp_path):
    fake = _FakeRunner()

    monkeypatch.setattr(tb_runner, "Path", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(tb_runner, "get_runner", lambda sim: fake)

    monkeypatch.setenv("HDL_TOPLEVEL_LANG", "vhdl")
    monkeypatch.setenv("SIM", "xcelium")

    tb_runner.test_runner("top_vhdl")

    proj_path = tmp_path.parent
    build_kwargs = fake.build_calls[0]
    assert build_kwargs["sources"] == [proj_path / "hdl" / "main.vhdl"]
    assert build_kwargs["build_args"] == ["-v93"]

    test_kwargs = fake.test_calls[0]
    assert test_kwargs["test_args"] == ["-v93"]
