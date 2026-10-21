import ast
import inspect
import random

import pytest

from py2ver.pipeline import expr_cost, pipeline_ir
from py2ver import Py2ver
from test_sha256_readiness import sha4_round_core
from test_state_stream import LPF_ATTR, lpf_step
from py2ver.visitor import FunctionVisitor

SHA_ATTR = {
    n: {"signed": 0, "width": 32, "type": "wire"}
    for n in "a b c d e f g h w0 w1 w2 w3 k0 k1 k2 k3 a4 b4 c4 d4 e4 f4 g4 h4".split()
}


def mixed_ops(a, b, c):
    # Signed floor division and signed/unsigned mixing (see semantics.py)
    p = a * b
    q = p // (c + 1)
    if a > b:
        r = q - c
    else:
        r = q + c
    s = (r >> 2) ^ c
    return r, s


MIXED_ATTR = {
    "a": {"signed": 1, "width": 8, "type": "wire"},
    "b": {"signed": 1, "width": 8, "type": "wire"},
    "c": {"signed": 0, "width": 7, "type": "wire"},
    "p": {"signed": 1, "width": 16, "type": "wire"},
    "q": {"signed": 1, "width": 16, "type": "wire"},
    "r": {"signed": 1, "width": 16, "type": "wire"},
    "s": {"signed": 1, "width": 16, "type": "wire"},
}


def add_reg(a, b):
    s = a + b
    return s


def sum_loop(x):
    acc = 0
    for i in range(0, 6):
        acc = acc + (x * i)
    return acc


LOOP_ATTR = {
    "x": {"signed": 1, "width": 8, "type": "wire"},
    "acc": {"signed": 1, "width": 16, "type": "wire"},
}


def _pipelined(func, attr, stages):
    v = FunctionVisitor(attr)
    ir = v.visit(ast.parse(inspect.getsource(func)))
    return pipeline_ir(ir, v.attr, stages)


def _register_counts(ir):
    """Possible register counts on paths from any input to each signal."""
    inputs = {p.name for p in ir.inputs if p.kind == "input"}
    by_left = {a.left: a for a in ir.assigns}
    memo = {}

    def counts(name):
        if name in inputs:
            return {0}
        if name in memo:
            return memo[name]
        memo[name] = set()
        a = by_left.get(name)
        if a is None:
            return set()
        out = set()
        for ident in Py2ver._extract_idents(a.right):
            if ident in inputs or ident in by_left:
                out |= counts(ident)
        if a.is_reg:
            out = {k + 1 for k in out}
        memo[name] = out
        return out

    return {p.name: counts(p.name) for p in ir.outputs}


# ----------------------------- structure (no simulator) -----------------------------

@pytest.mark.parametrize("stages", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "func, attr",
    [(sha4_round_core, SHA_ATTR), (mixed_ops, MIXED_ATTR), (sum_loop, LOOP_ATTR)],
)
def test_every_input_to_output_path_has_exactly_n_registers(func, attr, stages):
    res = _pipelined(func, attr, stages)

    for out, regs in _register_counts(res.ir).items():
        assert regs == {stages}, f"{func.__name__}: output {out} sees {regs} registers"
    assert res.ir.has_clk


def test_constants_are_not_registered():
    res = _pipelined(sha4_round_core, SHA_ATTR, 3)

    assert not any(a.left.startswith("m__p") for a in res.ir.assigns)
    assert any(a.left == "m" and not a.is_reg for a in res.ir.assigns)


def test_stages_balance_estimated_delay():
    res = _pipelined(sha4_round_core, SHA_ATTR, 3)

    assert len(res.stage_delay) == 4
    assert max(res.stage_delay) < 0.6 * sum(res.stage_delay)


def test_delay_registers_keep_width_of_ssa_temporaries(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(sum_loop, LOOP_ATTR, pipeline=3)

    verilog = (tmp_path / "hdl" / "main.v").read_text(encoding="utf-8")
    delayed = [line for line in verilog.splitlines() if line.startswith("reg") and "__p" in line]
    assert delayed
    assert all("[0:0]" not in line and "signed" in line for line in delayed)
    assert p.tb_latency_cycles == 3
    assert set(p.out_latency.values()) == {3}


def test_zero_stages_leaves_design_unchanged(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(mixed_ops, MIXED_ATTR, pipeline=0)

    assert p.has_clk is False
    assert p.tb_latency_cycles == 0


def test_expr_cost():
    assert expr_cost("(a) + (b)") == 1.0
    assert expr_cost("((a) * (b)) / (c)") == 12.5
    assert expr_cost("((e) >>> (6)) | ((e) << (26))") == 0.25  # constant shifts are wiring
    assert expr_cost("(e) >>> (n)") == 1.0                      # variable shift
    assert expr_cost("((a > b)) ? (x) : (y)") == 1.5
    assert expr_cost("16'sd0") == 0.0


@pytest.mark.parametrize(
    "kwargs, attr, func, exc",
    [
        ({"pipeline": 2},
         {"a": {"width": 8}, "b": {"width": 8}, "s": {"width": 9, "type": "reg"}},
         None, NotImplementedError),
        ({"pipeline": -1}, MIXED_ATTR, mixed_ops, ValueError),
    ],
)
def test_pipeline_rejects_unsupported_designs(monkeypatch, tmp_path, kwargs, attr, func, exc):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(exc):
        Py2ver(func or add_reg, attr, **kwargs)


# ----------------------------- simulation (new vector every clock) -----------------------------

@pytest.mark.sim
@pytest.mark.parametrize("stages", [1, 2, 3])
def test_pipelined_sha_rounds_match_python_at_full_throughput(assert_python_matches_sim, stages):
    rng = random.Random(stages)
    vectors = [tuple(rng.getrandbits(32) for _ in range(16)) for _ in range(24)]

    assert_python_matches_sim(sha4_round_core, SHA_ATTR, vectors, pipeline=stages)


@pytest.mark.sim
def test_pipelined_mixed_ops_match_python(assert_python_matches_sim):
    rng = random.Random(7)
    vectors = [(rng.randint(-128, 127), rng.randint(-128, 127), rng.randint(0, 126)) for _ in range(40)]
    assert_python_matches_sim(mixed_ops, MIXED_ATTR, vectors, pipeline=2)


@pytest.mark.sim
def test_pipelined_loop_matches_python(assert_python_matches_sim):
    vectors = [(x,) for x in range(-128, 128, 9)]
    assert_python_matches_sim(sum_loop, LOOP_ATTR, vectors, pipeline=3)


# ----------------------------- real Quartus -----------------------------

def test_pipelining_raises_fmax_in_quartus(quartus):
    flat = Py2ver(sha4_round_core, SHA_ATTR).SYN(quartus)
    piped = Py2ver(sha4_round_core, SHA_ATTR, pipeline=3).SYN(quartus)

    assert flat.meets_timing is False      # ~31 MHz against a 50 MHz target
    assert piped.meets_timing is True      # ~88 MHz
    assert piped.fmax_mhz > 1.5 * flat.fmax_mhz
