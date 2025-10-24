# tests/conftest.py
import pathlib, sys, textwrap, types, pytest

# Ensure project root on sys.path (so `visitor`, `renderer` import)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
