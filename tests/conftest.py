# tests/conftest.py
import os, pathlib, sys, textwrap, types, shutil, pytest

# Ensure project root on sys.path (so `visitor`, `renderer` import)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from py2ver import Py2ver

@pytest.fixture
def toy_src_add_xor() -> str:
    # We’ll pass this source string directly to ast.parse in tests
    return textwrap.dedent("""\
        def foo(a, b):
            x = (a & 0xF) + (b & 0xF)
            y = (a ^ b) << 1
            return x, y
    """)

@pytest.fixture
def toy_func_add_xor(toy_src_add_xor):
    ns: dict = {}
    code = compile(toy_src_add_xor, "<toy>", "exec")
    exec(code, ns, ns)
    return ns["foo"]

@pytest.fixture
def attr_simple():
    return {
        "a": {"width": 8, "signed": 0, "type": "wire"},
        "b": {"width": 8, "signed": 0, "type": "wire"},
        "x": {"width": 8, "signed": 0, "type": "wire"},
        "y": {"width": 9, "signed": 0, "type": "wire"},  # allow <<1
    }


def _as_tuple(v):
    if isinstance(v, tuple):
        return v
    return (v,)


def _normalize_expected(py_out, p):
    out = []
    for idx, name in enumerate(p.output_args_list):
        raw = int(py_out[idx])
        width = int(p.output_args_width_list[idx])
        signed = int(p.attr.get(name, {}).get("signed", 0))
        mask = (1 << width) - 1 if width > 0 else 0
        raw &= mask
        if signed == 1 and width > 0:
            sign_bit = 1 << (width - 1)
            raw = (raw ^ sign_bit) - sign_bit
        out.append(raw)
    return tuple(out)


def pytest_collection_modifyitems(items):
    """A test using the simulation fixture is a simulation test; one using
    Quartus's, a quartus test."""
    for item in items:
        if "assert_python_matches_sim" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.sim)
        if "quartus" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.quartus)


# Quartus where the DE0-Nano's toolchain installs it (examples/_run.py's default)
DEFAULT_QUARTUS_DIR = "/opt/intelFPGA/22.1std/quartus/bin"


def quartus_dir():
    """Quartus's bin directory: $QUARTUS_DIR, quartus_sh on PATH, or its usual
    install; None if there is none."""
    on_path = shutil.which("quartus_sh")
    for d in (os.environ.get("QUARTUS_DIR"), os.path.dirname(on_path) if on_path else None,
              DEFAULT_QUARTUS_DIR):
        if d and os.path.exists(os.path.join(d, "quartus_sh")):
            return d
    return None


@pytest.fixture
def quartus(monkeypatch, tmp_path):
    """SYN()'s options with Quartus's directory, the test in its own directory;
    the test skipped if there is no Quartus."""
    d = quartus_dir()
    if d is None:
        pytest.skip(f"Quartus not found (set QUARTUS_DIR, put quartus_sh on PATH or install "
                    f"it in {DEFAULT_QUARTUS_DIR})")
    monkeypatch.chdir(tmp_path)
    return {"QUARTUS_DIR": d}


@pytest.fixture(autouse=True)
def _simulations_are_marked(request, monkeypatch):
    """A test not marked sim / board may not start a simulator, so that
    `pytest -m "not sim and not board"` stays quick (and its tests keep running
    in parallel with -n): mark a test that simulates."""
    if request.node.get_closest_marker("sim") or request.node.get_closest_marker("board"):
        return

    def refuse(*_args, **_kwargs):
        raise RuntimeError(f"{request.node.nodeid} runs a simulation: mark it @pytest.mark.sim")
    import py2ver.native
    import py2ver.tb_runner
    monkeypatch.setattr(py2ver.native, "_run", refuse)          # iverilog / vvp / verilator
    monkeypatch.setattr(py2ver.tb_runner, "get_runner", refuse)  # cocotb


@pytest.fixture
def assert_python_matches_sim(monkeypatch, tmp_path):
    """Assert Python reference output equals cocotb+Icarus output for vectors."""
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    pytest.importorskip("cocotb")
    pytest.importorskip("cocotb_tools")

    monkeypatch.chdir(tmp_path)                 # (its own directory: tests run in parallel)
    monkeypatch.setenv("SIM", "icarus")

    def _run(func, attr, vectors, **py2ver_kwargs):
        p = Py2ver(func, attr, **py2ver_kwargs)
        vectors = [tuple(vec) for vec in vectors]
        sim_outs = p.TB_batch()(vectors)
        assert len(sim_outs) == len(vectors)
        for vec, sim_out in zip(vectors, sim_outs):
            py_out = _normalize_expected(_as_tuple(func(*vec)), p)
            assert sim_out == py_out, f"mismatch for {func.__name__}{vec}: sim={sim_out} py={py_out}"

    return _run
