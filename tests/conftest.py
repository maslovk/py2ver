# tests/conftest.py
import pathlib, sys, textwrap, types, shutil, pytest

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


@pytest.fixture
def assert_python_matches_sim(monkeypatch):
    """Assert Python reference output equals cocotb+Icarus output for vectors."""
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    pytest.importorskip("cocotb")
    pytest.importorskip("cocotb_tools")

    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("SIM", "icarus")

    def _run(func, attr, vectors):
        p = Py2ver(func, attr)
        tb = p.TB()
        for vec in vectors:
            py_out = _normalize_expected(_as_tuple(func(*vec)), p)
            sim_out = tb(*vec)
            assert sim_out == py_out, f"mismatch for {func.__name__}{vec}: sim={sim_out} py={py_out}"

    return _run
