import pytest

from py2ver import Array, Py2ver, Ram, Reg, S, State, U

S8 = {"signed": 1, "width": 8}
U8 = {"signed": 0, "width": 8}


# ----------------------------- designs: annotated vs attr -----------------------------

def foo_attr(arg1, arg2):
    a = arg1 + arg2
    b = arg1 - arg2
    c = arg1 * arg2
    d = arg1 / arg2
    return a, b, c, d


FOO_ATTR = {
    "arg1": {"signed": 0, "width": 8},
    "arg2": {"signed": 0, "width": 8},
    "a": {"signed": 0, "width": 9, "type": "reg"},
    "b": {"signed": 1, "width": 9},
    "c": {"signed": 0, "width": 9},
    "d": {"signed": 0, "width": 9},
}


def foo_ann(arg1: U[8], arg2: U[8]) -> (Reg(U[9]), S[9], U[9], U[9]):
    a = arg1 + arg2
    b = arg1 - arg2
    c = arg1 * arg2
    d = arg1 / arg2
    return a, b, c, d


def lpf_attr(x, y_prev):
    err = x - y_prev
    delta = err >> 3
    y = y_prev + delta
    return y


LPF_ATTR = {
    "x": {"signed": 1, "width": 12},
    "y_prev": {"signed": 1, "width": 16, "next": "y"},
    "err": {"signed": 1, "width": 16},
    "delta": {"signed": 1, "width": 16},
    "y": {"signed": 1, "width": 16},
}


def lpf_ann(x: S[12], y_prev: State(S[16], next="y")) -> S[16]:
    err: S[16] = x - y_prev
    delta: S[16] = err >> 3
    y = y_prev + delta
    return y


def hist_attr(bin_, counts):
    c = counts[bin_] + 1
    counts[bin_] = c
    return c


HIST_ATTR = {
    "bin_": {"signed": 0, "width": 4},
    "counts": {"signed": 0, "width": 8, "length": 16, "memory": "ram"},
}


def hist_ann(bin_: U[4], counts: Ram(U[8], 16)):
    c = counts[bin_] + 1
    counts[bin_] = c
    return c


def regfile_ann(i: U[2], v: U[8], mem: State(Array(U[8], 4), next="mem", init=[1, 2, 3, 4])):
    mem[i] += v
    return max(mem), mem


@pytest.mark.parametrize(
    "annotated, func, attr",
    [(foo_ann, foo_attr, FOO_ATTR), (lpf_ann, lpf_attr, LPF_ATTR), (hist_ann, hist_attr, HIST_ATTR)],
)
def test_annotations_generate_the_same_verilog_as_attr(monkeypatch, tmp_path, annotated, func, attr):
    monkeypatch.chdir(tmp_path)
    Py2ver(func, attr)
    expected = (tmp_path / "hdl" / "main.v").read_text().replace(func.__name__, "<top>")
    Py2ver(annotated)
    got = (tmp_path / "hdl" / "main.v").read_text().replace(annotated.__name__, "<top>")
    assert got == expected


def test_annotated_function_still_runs_as_python():
    assert foo_ann(3, 1) == (4, 2, 3, 3.0)
    assert lpf_ann(256, 0) == 32


def test_attr_merges_over_annotations(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(lpf_ann, {"x": {"width": 10}, "y_prev": {"init": 7}})
    assert p.attr["x"]["width"] == 10 and p.attr["x"]["signed"] == 1
    assert p.state_init["y_prev"] == 7


def test_attr_values_may_be_types(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Py2ver(lpf_attr, LPF_ATTR)
    expected = (tmp_path / "hdl" / "main.v").read_text()
    Py2ver(lpf_attr, {"x": S[12], "y_prev": State(S[16], next="y"),
                      "err": S[16], "delta": S[16], "y": S[16]})
    assert (tmp_path / "hdl" / "main.v").read_text() == expected


# ----------------------------- forms of annotation -----------------------------

def ret_expressions(a: S[8], b: U[8]) -> (S[12], U[3]):
    return a * b, b & 7


def bare_declaration(a: S[8], b: S[8]):
    y: S[10]
    if a > b:
        t: S[9]
        y = a + b
    else:
        y = a - b
    return y


def string_annotation(a: "U[8]", b: "U[8]") -> "U[9]":
    return a + b


def plain_int_hints(a: int, b: U[8]) -> int:
    y = a + b
    return y


def _halve(v: S[16]) -> S[16]:
    return v >> 1


def helper_with_hints(a: S[8]):
    y = _halve(a) + 1
    return y


def test_return_annotation_names_expressions(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(ret_expressions)
    assert p.output_args_list == ["ret0", "ret1"]
    assert p.output_args_width_list == [12, 3]
    assert p.attr["ret0"]["signed"] == 1 and p.attr["ret1"]["signed"] == 0


def test_forms(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(bare_declaration)
    assert p.attr["y"]["width"] == 10 and p.attr["t"]["width"] == 9
    p = Py2ver(string_annotation)
    assert p.output_args_width_list == [9]
    p = Py2ver(plain_int_hints, {"a": U8})
    assert p.attr["a"]["width"] == 8
    Py2ver(helper_with_hints)


def test_type_reprs():
    assert repr(S[16]) == "S[16]"
    assert repr(Ram(U[8], 16)) == "Ram(U[8], 16)"
    assert repr(State(Array(U[8], 4), next="m")) == "State(Array(U[8], 4), next='m', init=None)"
    assert Reg(U[9]).attr() == {"signed": 0, "width": 9, "type": "reg"}


# ----------------------------- errors -----------------------------

def conflicting(a: U[8]):
    a: U[9] = a + 1
    return a


def wrong_return_count(a: U[8]) -> (U[8], U[8]):
    return a + 1


def annotated_returned_param(a: U[8]) -> U[8]:
    return a


def non_type_local(a: U[8]):
    y: int = a + 1
    return y


def _typed_local_helper(v):
    t: U[8] = v + 1
    return t


def helper_typed_local(a: U[8]):
    y = _typed_local_helper(a)
    return y


@pytest.mark.parametrize(
    "func, exc, match",
    [
        (conflicting, ValueError, "annotated as both"),
        (wrong_return_count, ValueError, "2 types for 1 returned"),
        (annotated_returned_param, ValueError, "takes its type from the parameter"),
        (non_type_local, TypeError, "not a type such as"),
        (helper_typed_local, NotImplementedError, "annotate variables in the top function"),
    ],
)
def test_bad_annotations_raise_clearly(monkeypatch, tmp_path, func, exc, match):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(exc, match=match):
        Py2ver(func)


def test_bad_widths():
    with pytest.raises(ValueError):
        U[0]
    with pytest.raises(TypeError):
        Array(8, 4)
    with pytest.raises(NotImplementedError):
        Array(Array(U[8], 2), 2)


# ----------------------------- simulation -----------------------------

@pytest.mark.sim
def test_annotated_designs_match_python_in_simulation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    xs = [0, 256, 256, 512, -2048, 2047]
    p = Py2ver(lpf_ann)
    assert p.TB_stream()(xs) == p.PY_stream()(xs)

    p = Py2ver(regfile_ann)
    samples = [(0, 5), (3, 250), (1, 1), (3, 10)]
    assert p.TB_stream()(samples) == p.PY_stream()(samples)

    p = Py2ver(hist_ann)
    bins = [1, 1, 15, 1, 0]
    assert p.TB_stream()(bins) == p.PY_stream()(bins)

    p = Py2ver(bare_declaration)
    vectors = [(-128, 127), (127, 127), (5, -3), (-1, -1)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)
