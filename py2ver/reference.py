"""Py2ver's Python side (a mixin of Py2ver, see core.py)."""

from __future__ import annotations

import functools
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .containers import flatten, nest
from .fixed import quantize


def expand_args(call_inputs: List[str], cplx_params: set, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Arguments as the caller gives them -> as the ports take them (a complex
    argument as its real and imaginary parts)."""
    if not cplx_params:
        return tuple(args)
    out: List[Any] = []
    for name, v in zip(call_inputs, args):
        if name in cplx_params and isinstance(v, (list, tuple)):     # a complex list
            vs = [complex(x) for x in v]
            out += [[x.real for x in vs], [x.imag for x in vs]]
        elif name in cplx_params:
            v = complex(v)
            out += [v.real, v.imag]
        else:
            out.append(v)
    return tuple(out) + tuple(args[len(call_inputs):])


def _flatten_outputs(ret: Any) -> Tuple[Any, ...]:
    """
    Python results as output values, in port order: a complex value becomes
    (real, imag), a list its elements, and a list of complex values its real
    parts, then its imaginary parts (<xs>_re_*, <xs>_im_*).
    """
    out: List[Any] = []

    def add(v: Any) -> None:
        if isinstance(v, complex):
            out.extend([v.real, v.imag])
        elif isinstance(v, (list, tuple)) and any(isinstance(e, complex) for e in v):
            out.extend([complex(e).real for e in v] + [complex(e).imag for e in v])
        elif isinstance(v, (list, tuple)):
            for e in v:
                add(e)
        else:
            out.append(v)

    for v in ret if isinstance(ret, tuple) else (ret,):
        add(v)
    return tuple(out)


class ReferenceMixin:
    """Py2ver's Python side: the function (or its bit-exact model) run on the inputs the hardware gets, and values to and from the ports' raw bits."""

    @staticmethod
    def _wrap(n: int, nbits: int, signed: bool) -> int:
        n &= (1 << nbits) - 1
        if signed:
            sign_bit = 1 << (nbits - 1)
            n = (n ^ sign_bit) - sign_bit
        return n

    @staticmethod
    def tosigned(n: int, nbits: int) -> int:
        mask = (1 << nbits) - 1
        n &= mask
        sign_bit = 1 << (nbits - 1)
        return (n ^ sign_bit) - sign_bit

    def PY_batch(self) -> Callable[[List[Tuple[Any, ...]]], List[Tuple[int, ...]]]:
        return self._fx_rows(self.py_batch_fun)

    def py_batch_fun(self, vectors: List[Tuple[Any, ...]]) -> List[Tuple[int, ...]]:
        """
        Python reference for tb_batch_fun(): the function's results (for a
        fixed-point design, its bit-exact model's), wrapped to each output's
        declared width and sign (int() truncates /).
        """
        if self.state_map or self.rams or self._seq_mems:
            raise ValueError("This design has state ('next' or 'memory' in attr); use PY_stream().")
        func = self._model or self._original()
        return [self._py_row(func(*self._py_args(vec))) for vec in vectors]

    def _original(self) -> Callable:
        """The function as written, taking the flat lists the ports take for its
        lists of lists (it gets them nested)."""
        if not self._shapes:
            return self.func
        names = [self._py_name.get(p, p) for p in self._shapes]
        shapes = {self._py_name.get(p, p): s for p, s in self._shapes.items()}
        sig = inspect.signature(self.func)

        @functools.wraps(self.func)
        def call(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            for name in names:
                if name in bound.arguments:
                    bound.arguments[name] = nest(flatten(bound.arguments[name]), shapes[name])
            return self.func(*bound.args, **bound.kwargs)
        return call

    def _value_names(self) -> List[str]:
        """The names of the values the function returns, in order: the ports'
        and the 'internal' ones' (a register's next value without a port); the
        ports' alone where they aren't all the returned names (a list returned
        through the SDRAM)."""
        names = getattr(self, "_returned_names", None)
        if names is None or any(n not in names for n in self.output_args_list):
            return list(self.output_args_list)
        return names

    def _py_row(self, ret: Any) -> Tuple[int, ...]:
        # (every returned value by its name: the ports' kept, an 'internal' one not)
        values = dict(zip(self._value_names(), _flatten_outputs(ret)))
        return tuple(
            self._wrap(int(values[n]), w, int(self.attr.get(n, {}).get("signed", 0)) == 1)
            for n, w in zip(self.output_args_list, self.output_args_width_list)
        )

    def hardware_random(self):
        """A context manager: inside it, the original code's random numbers (numpy
        Generators' .random, torch's dropout) are the ones the hardware draws, from
        their start, so the original can be compared with the hardware step by step."""
        from .numpy_loops import _drawing
        return _drawing(getattr(self._numpy, "generators", None) or {})

    def state_values(self, row: Tuple[Any, ...] | None = None) -> Dict[str, Any]:
        """
        numpy designs with state: the object's attributes in one result row
        of PY_stream() / TB_stream(), as arrays of their shapes (e.g. the
        weights after a hardware training step): {'W1': array, ...}. A
        sequenced design keeps its state in memories, not in its results:
        without a row, the state after the last PY_stream(), as
        {(object, attribute): array or number} (the object the original's).
        """
        import numpy as np
        if row is None and self._numpy is not None and getattr(self._numpy, "memories", None):
            if not self._last_memories:
                raise ValueError("Run PY_stream() first")
            out = {}
            for mem, (label, init, owner) in self._numpy.memories.items():
                frac = self._fx_frac(mem)
                raw = self._last_memories[mem]
                vals = [v / (1 << frac) if frac is not None else v for v in raw]
                out[(owner, label.split(".", 1)[1])] = (np.array(vals, dtype=float).reshape(np.shape(init))
                                                     if isinstance(init, np.ndarray) else vals[0])
            return out
        if self._numpy is None or not self._numpy.state:
            raise ValueError("Only for a numpy design with state")
        out, i = {}, len(self.output_args_list) - sum(v.size for v in self._numpy.state.values())
        for a, init in self._numpy.state.items():
            out[a] = np.array(row[i:i + init.size], dtype=float).reshape(init.shape)
            i += init.size
        return out

    def PY_float(self) -> Callable[[List[Any]], List[Tuple[Any, ...]]]:
        """
        The original function as written (floats), on the same inputs as the
        hardware after they are quantized: the ideal a fixed-point design
        approximates. Takes vectors like PY_batch(), or samples like
        PY_stream() for a design with state.
        """
        return self.py_float_fun

    def py_float_fun(self, vectors: List[Any]) -> List[Tuple[Any, ...]]:
        from .auto import float_reference
        if self._adapt is not None:
            vectors = [self._adapt(v) for v in vectors]
        return float_reference(self.func, self._py_attr, vectors)

    def quantization_error(self, vectors: List[Any]) -> Dict[str, Tuple[float, float]]:
        """
        Per output: (max abs error, RMS error) of the bit-exact results
        (PY_batch / PY_stream, which the hardware matches) against PY_float().
        """
        stream = bool(self.state_map or self.rams or self._seq_mems)
        exact = (self.PY_stream() if stream else self.PY_batch())(vectors)
        ideal = self.py_float_fun(vectors)
        out: Dict[str, Tuple[float, float]] = {}
        for i, name in enumerate(self.output_args_list):
            errs = [abs(float(e[i]) - float(r[i])) for e, r in zip(exact, ideal)]
            if errs:
                out[self._py_name.get(name, name)] = (
                    max(errs), (sum(x * x for x in errs) / len(errs)) ** 0.5)
        return out

    def PY_stream(self) -> Callable[[List[Any]], List[Tuple[int, ...]]]:
        return self._fx_rows(self.py_stream_fun)

    @staticmethod
    def _raw_init(meta: Any) -> Any:
        """A fixed-point state/memory's 'init' (numbers) as raw values."""
        if not (isinstance(meta, dict) and meta.get("frac") is not None and "init" in meta):
            return meta
        q = lambda v: quantize(v, meta["frac"], meta["width"], bool(meta.get("signed")),  # noqa: E731
                               bool(meta.get("round")), bool(meta.get("saturate")))
        init = meta["init"]
        return dict(meta, init=[q(v) for v in init] if isinstance(init, (list, tuple)) else q(init))

    def _fx_frac(self, port: str) -> Optional[int]:
        """Fraction bits of an output port (xs_3 is an element of xs, a_out
        returns a, xs_3_out returns an element of xs)."""
        names = [port]
        for name in list(names):
            base = name[:-4] if name.endswith("_out") else name
            m = re.fullmatch(r"(.+)_(\d+)", base)
            names += [base] + ([m.group(1)] if m else [])
        for name in names:
            t = self._fx_types.get(name)
            if t is not None:
                t = t[1] if t[0] == "l" else t
                return t[1] if t is not None and t[0] == "f" else None
        return None

    def _py_value(self, param: str, value: Any) -> Any:
        """An argument for the ports / the bit-exact model: a fixed-point
        parameter's numbers as raw values, quantized like assignments to it."""
        if param in self._shapes:            # a list of lists: flat, row-major
            value = flatten(value)
        m = self._fx_params.get(param)
        if m is None:
            return value
        conv = lambda v: quantize(v, m["frac"], m["width"], bool(m.get("signed")),  # noqa: E731
                                  bool(m.get("round")), bool(m.get("saturate")))
        return [conv(v) for v in value] if isinstance(value, (list, tuple)) else conv(value)

    def _expand(self, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
        return expand_args(self._call_inputs, self._cplx_params, args)

    def _py_args(self, vec: Tuple[Any, ...]) -> Tuple[Any, ...]:
        vec = self._expand(tuple(vec))
        params = [p for p, _ in self.arg_groups]
        return tuple(self._py_value(p, v) for p, v in zip(params, vec)) + tuple(vec[len(params):])

    def _fx_rows(self, fn: Callable, single: bool = False) -> Callable:
        """fn with its results' fixed-point outputs as numbers (raw / 2**frac),
        and (numpy designs) taking arguments as the original does: one call's
        if `single`, else a list of them first."""
        if not self._fx_out and self._adapt is None:
            return fn
        fracs = [self._fx_out.get(n) for n in self.output_args_list]

        def row(r):
            return tuple(v if f is None else int(v) / (1 << f) for v, f in zip(r, fracs))

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if self._adapt is not None:
                args = (self._adapt(args) if single
                        else ([self._adapt(v) for v in args[0]],) + tuple(args[1:]))
            out = fn(*args, **kwargs)
            return [row(r) for r in out] if isinstance(out, list) else row(out)

        return wrapped

    def _flatten_args(self, in_arg: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Function arguments (a list argument as a list) -> one value per input port."""
        groups = [(p, ports) for p, ports in self.arg_groups
                  if (ports or [p])[0] in self.input_args_list]   # state lives in registers
        in_arg = self._expand(tuple(in_arg))
        out: List[Any] = []
        for value, (param, ports) in zip(in_arg, groups):
            value = self._py_value(param, value)
            if ports is None:
                out.append(value)
                continue
            value = list(value)
            if len(value) != len(ports):
                raise ValueError(f"'{param}' needs {len(ports)} values, got {len(value)}")
            out.extend(value)
        return tuple(out)

    def _input_vector(self, in_arg: Tuple[Any, ...]) -> Tuple[int, ...]:
        # Missing trailing arguments default to 0
        flat = self._flatten_args(tuple(in_arg))
        return tuple(
            int(flat[i]) if i < len(flat) else 0
            for i in range(len(self.input_args_list))
        )

    def _stream_vectors(self, samples: List[Any]) -> List[Tuple[int, ...]]:
        if not (self.state_map or self.rams or self.schedule):
            raise ValueError("Streams need state ('next' or 'memory' in attr); use TB_batch().")
        # A sample is a tuple of arguments; anything else is the only argument
        return [self._input_vector(s if isinstance(s, tuple) else (s,)) for s in samples]

    def py_stream_fun(self, samples: List[Any]) -> List[Tuple[int, ...]]:
        """
        Python reference for tb_stream_fun(): calls the function (for a
        fixed-point design, its bit-exact model) once per sample, feeding each
        next-state output back into its state argument with the same
        width/sign wrapping as the hardware registers.
        """
        samples = list(samples)
        self._stream_vectors(samples)  # validates
        groups = dict(self.arg_groups)

        model = self._model is not None
        func = self._model if model else self._original()
        py = (lambda name: name) if model else (lambda name: self._py_name.get(name, name))  # noqa: E731
        params = [p for p in inspect.signature(func).parameters
                  if p not in {py(m) for m in self._seq_mems}]
        if params != [py(p) for p, _ in self.arg_groups]:
            raise ValueError(f"PY_stream() needs arguments {list(groups)}; function has {params}")
        inputs = [p for p, _ in self.arg_groups
                  if p not in self.state_params and p not in self.rams]

        def _meta(name: str) -> Tuple[int, bool]:
            m = self.attr.get(name, {})
            return int(m.get("width", 1)), int(m.get("signed", 0)) == 1

        def keep(name: str, v: Any) -> Any:
            return self._wrap(int(v), *_meta(name))

        def _state_value(param: str, value_of) -> Any:
            ports = groups[param]
            if ports is None:
                return keep(param, value_of(param))
            return [keep(port, value_of(port)) for port in ports]

        state = {p: _state_value(p, self.state_init.__getitem__) for p in self.state_params}
        # Memories: the function updates the list in place; values wrap like the RAM
        rams = {}
        for mem, ram in self.rams.items():
            init = self._ram_init[mem]
            vals = list(init) if isinstance(init, (list, tuple)) else [init] * ram["length"]
            w, sgn = int(ram["elem"]["width"]), int(ram["elem"]["signed"]) == 1
            rams[mem] = [self._wrap(int(v), w, sgn) for v in vals]
        # (a sequenced design's memories: their raw 'init', wrapped as kept)
        mems = {mem: list(m["init"]) for mem, m in self._seq_mems.items()}
        outs: List[Tuple[Any, ...]] = []
        for sample in samples:
            args = self._expand(sample if isinstance(sample, tuple) else (sample,))
            kwargs = dict(zip(inputs, args))
            for p in inputs[len(args):]:                     # missing trailing args: 0
                kwargs[p] = 0 if groups[p] is None else [0] * len(groups[p])
            kwargs = {p: self._py_value(p, v) for p, v in kwargs.items()}
            call = {py(k): v for k, v in {**kwargs, **state, **rams, **mems}.items()}
            ret = _flatten_outputs(func(**call))
            for mem, ram in self.rams.items():
                w, sgn = int(ram["elem"]["width"]), int(ram["elem"]["signed"]) == 1
                rams[mem][:] = [self._wrap(int(v), w, sgn) for v in rams[mem]]
            for mem, m in self._seq_mems.items():
                mems[mem][:] = [self._wrap(int(v), m["width"], bool(m["signed"]))
                                for v in mems[mem]]
            row = {name: keep(name, v) for name, v in zip(self._value_names(), ret)}
            outs.append(tuple(row[n] for n in self.output_args_list))
            state = {p: _state_value(p, lambda port: row[self.state_map[port]])
                     for p in self.state_params}
        self._last_memories = {mem: list(v) for mem, v in mems.items()}
        return outs
