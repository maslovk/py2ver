"""Py2ver's testbenches (a mixin of Py2ver, see core.py)."""

from __future__ import annotations

import ast
import logging
import pickle
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from . import tb_runner
from .defaults import DEFAULT_TB_PERIOD, RESULTS_PATH

log = logging.getLogger(__name__)


def _drop_bytecode(directory: Path, module: str) -> None:
    """
    Remove the cached bytecode of a testbench module before rewriting it:
    Python checks a .pyc only by the source's size and mtime (in seconds), so
    a rewrite of the same size within a second would run the old testbench.
    """
    for p in (directory / "__pycache__").glob(f"{module}.*.pyc"):
        try:
            log.debug("Removing stale pycache file: %s", p)
            p.unlink()
        except OSError as e:
            log.warning("Failed to remove %s: %s", p, e)


class SimulationMixin:
    """Py2ver's testbenches: the design simulated (a plain Verilog testbench on Icarus or Verilator, or cocotb), its stream wrapper and outputs' latencies."""

    @staticmethod
    def _extract_idents(expr: str) -> List[str]:
        # verilog-ish identifiers
        return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)

    def _compute_latencies(self) -> Dict[str, int]:
        """
        Conservative pipeline latency in *cycles* for each signal.

        wire assign: lat(lhs) = max(lat(deps))
        reg  assign: lat(lhs) = 1 + max(lat(deps))

        This correctly yields:
          d <= expr(inputs)  => lat(d)=1
          e <= d             => lat(e)=2
        """
        ir = getattr(self, "ir", None)
        if ir is None:
            return {n: 0 for n in self.output_args_list}

        # Known names: ports + anything ever assigned
        input_names = set(self.input_args_list)
        output_names = set(self.output_args_list)
        lhs_names = {a.left for a in ir.assigns}
        known = input_names | output_names | lhs_names

        # Seed: inputs are 0 cycles
        lat: Dict[str, int] = {n: 0 for n in known}
        for n in input_names:
            lat[n] = 0

        # Precompute deps per assignment
        assigns: List[tuple[str, bool, List[str]]] = []
        for a in ir.assigns:
            deps = []
            for ident in self._extract_idents(a.right):
                if ident.startswith("$"):
                    continue
                if ident == "clk":
                    continue
                if ident in known:
                    deps.append(ident)
            assigns.append((a.left, bool(a.is_reg), deps))

        # Fixed-point relaxation.
        # Works even if assigns are out-of-order.
        # Safe even with self-references from guarded updates (we ignore d==lhs).
        max_iters = len(known) + 10
        for _ in range(max_iters):
            changed = False
            for left, is_reg, deps in assigns:
                dep_lat = 0
                for d in deps:
                    if d == left:
                        continue  # guarded update pattern: lhs = cond ? rhs : lhs
                    dep_lat = max(dep_lat, lat.get(d, 0))
                new_lat = dep_lat + (1 if is_reg else 0)
                if new_lat > lat.get(left, 0):
                    lat[left] = new_lat
                    changed = True
            if not changed:
                break

        return lat

    def _compute_tb_latency(self) -> Tuple[int, Dict[str, int]]:
        """
        Returns:
          (cycles_to_wait, per_output_latency)
        """
        if self.state_map:
            # Stream mode: outputs are combinational in (inputs, state) and are
            # sampled every clock before the state registers update.
            return 0, {n: 0 for n in self.output_args_list}
        if self.schedule:
            # Several clocks per sample (pre_step), its registers in loops: the
            # results are there a clock after the last
            return 1, {n: 1 for n in self.output_args_list}
        per_sig = self._compute_latencies()
        per_out = {n: int(per_sig.get(n, 0)) for n in self.output_args_list}
        cycles = max(per_out.values(), default=0)
        return cycles, per_out

    def _stream_mode(self) -> str:
        if getattr(self, "ram_phases", 1) > 1:
            return "ram"
        if self.pipeline_stages:
            return "pipe"
        if self.has_clk and not (self.state_map or self.rams):
            raise NotImplementedError(
                "The stream wrapper needs a combinational, stateful or pipelined design; "
                "'reg' signals have no balanced latency (use pipeline=N instead)."
            )
        return "reg_out"

    def _serial_ports(self, attr: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lists the stream wrapper moves one element per transfer ('serial' in
        attr): {'in': {param: [ports]}, 'out': {name: [ports]}, 'm', 'l'}
        (m inputs per call, l results per call).
        """
        sin = {p: ports for p, ports in self.arg_groups
               if ports and attr.get(p, {}).get("serial") and ports[0] in self.input_args_list}
        for p, _ in self.arg_groups:
            if attr.get(p, {}).get("serial") and p not in sin:
                raise ValueError(f"'serial' input '{p}' must be a list input (not state)")
        sout: Dict[str, List[str]] = {}
        for name, meta in attr.items():
            if not (isinstance(meta, dict) and meta.get("serial")) or name in dict(self.arg_groups):
                continue
            ports = [n for n in self.output_args_list if re.fullmatch(rf"{re.escape(name)}_\d+", n)]
            if not ports:
                raise ValueError(f"'serial' output '{name}' must be a returned list")
            sout[name] = sorted(ports, key=lambda n: int(n.rsplit("_", 1)[1]))
        lengths_in = {len(v) for v in sin.values()}
        lengths_out = {len(v) for v in sout.values()}
        if len(lengths_in) > 1 or len(lengths_out) > 1:
            raise ValueError("Serial inputs (and serial outputs) must all have the same length")
        return {"in": sin, "out": sout, "m": lengths_in.pop() if sin else 1,
                "l": lengths_out.pop() if sout else 1}

    def stream_wrapper(self, write: bool = True) -> Path:
        """
        Write hdl/<name>_stream.v (with write=False: only into hdl_files): the
        module behind a valid/ready interface (clk, rst, s_valid/s_ready/s_<input>,
        m_valid/m_ready/m_<output>). Lists marked 'serial' take or give one
        element per transfer.
        """
        mode = self._stream_mode()

        def meta(name: str) -> Tuple[int, bool]:
            m = self.attr.get(name, {})
            return int(m.get("width", 1)), int(m.get("signed", 0)) == 1

        def rng(name: str) -> str:
            w, sgn = meta(name)
            return f"{'signed ' if sgn else ''}[{w - 1}:0]"

        def joint(ports: List[str]) -> str:
            """A range holding every element: signed if any is (unsigned ones then need a bit more)."""
            signed = any(meta(n)[1] for n in ports)
            w = max(meta(n)[0] + (1 if signed and not meta(n)[1] else 0) for n in ports)
            return f"{'signed ' if signed else ''}[{w - 1}:0]"

        serial = self._serial["in"] or self._serial["out"]
        ctx = {
            "module_name": f"{self.t_name}_blocks" if serial else f"{self.t_name}_stream",
            "name": self.t_name,
            "mode": mode,
            "stages": self.pipeline_stages,
            "in_decls": [f"{rng(n)} s_{n}" for n in self.input_args_list],
            "out_decls": [f"{rng(n)} m_{n}" for n in self.output_args_list],
            "in_conns": [(n, rng(n)) for n in self.input_args_list],
            "out_conns": [(n, rng(n)) for n in self.output_args_list],
            "has_clk": self.has_clk,
            "has_rst": any(p.kind == "rst" for p in self.ir.inputs),
            "has_ce": self._has_ce,
            "has_vin": self._has_vin,
            "phases": getattr(self, "ram_phases", 1),
        }
        text = self.get_template("stream_wrapper.txt").render(ctx)
        if serial:
            s_in = [p for ports in self._serial["in"].values() for p in ports]
            s_out = [p for ports in self._serial["out"].values() for p in ports]
            plain_in = [n for n in self.input_args_list if n not in s_in]
            plain_out = [n for n in self.output_args_list if n not in s_out]
            sin = [{"name": p, "range": rng(ports[0])} for p, ports in self._serial["in"].items()]
            sout = [{"name": n, "range": joint(ports)} for n, ports in self._serial["out"].items()]
            text += self.get_template("serial_wrapper.txt").render({
                "name": self.t_name,
                "m_in": self._serial["m"], "l_out": self._serial["l"],
                "serial_in": list(self._serial["in"]), "serial_out": list(self._serial["out"]),
                "in_decls": [f"{s['range']} s_{s['name']}" for s in sin]
                            + [f"{rng(n)} s_{n}" for n in plain_in],
                "out_decls": [f"{s['range']} m_{s['name']}" for s in sout]
                             + [f"{rng(n)} m_{n}" for n in plain_out],
                "block_in": [{"port": n, "range": rng(n)} for n in self.input_args_list],
                "block_out": [{"port": n, "range": rng(n)} for n in self.output_args_list],
                "serial_in_ports": sin, "serial_out_ports": sout,
                "plain_in": [{"port": n, "range": rng(n)} for n in plain_in],
                "plain_out": [{"port": n, "range": rng(n)} for n in plain_out],
            })
        path = self.hdl_dir / f"{self.t_name}_stream.v"
        self.hdl_files[path.name] = text
        if write:
            self.write_hdl()
            log.info("Stream wrapper (%s) written to %s", mode, path)
        return path

    def _transfers(self, vecs: List[Tuple[int, ...]]) -> Tuple[List[Tuple[int, ...]], List[str]]:
        """Input vectors of calls -> the stream wrapper's transfers (a serial list
        one element per transfer), and the wrapper's input port names."""
        ser = self._serial
        if not ser["in"]:
            return vecs, list(self.input_args_list)
        s_in = [p for ports in ser["in"].values() for p in ports]
        plain = [n for n in self.input_args_list if n not in s_in]
        idx = {n: i for i, n in enumerate(self.input_args_list)}
        out = [tuple(vec[idx[ports[k]]] for ports in ser["in"].values())
               + tuple(vec[idx[n]] for n in plain)
               for vec in vecs for k in range(ser["m"])]
        return out, list(ser["in"]) + plain

    def TB_handshake(self) -> Callable[..., List[Tuple[int, ...]]]:
        return self._fx_rows(self.tb_handshake_fun)

    def tb_handshake_fun(self, vectors: List[Any], valid: float = 0.7, ready: float = 0.7,
                         seed: int = 0) -> List[Tuple[int, ...]]:
        """
        Simulate the stream wrapper with inputs offered with probability `valid`
        and results accepted with probability `ready` each clock. Returns the
        results in the order they were taken (one per input). Vectors are
        samples for a design with state (as for TB_stream), else as for TB_batch.
        """
        vectors = list(vectors)
        if not vectors:
            return []
        if self.state_map or self.rams or self.schedule:
            vecs = self._stream_vectors(vectors)
        else:
            vecs = [self._input_vector(tuple(v)) for v in vectors]
        masks = [(1 << w) - 1 for w in self.input_args_width_list]
        vecs = [tuple(v & m for v, m in zip(vec, masks)) for vec in vecs]
        calls = len(vecs)
        out_names = self.output_args_list
        ser = self._serial
        s_out = [p for ports in ser["out"].values() for p in ports]
        vecs, in_names = self._transfers(vecs)
        if ser["out"]:
            out_names = list(ser["out"]) + [n for n in out_names if n not in s_out]

        wrapper = self.stream_wrapper()
        results_path = self.outdir / RESULTS_PATH
        if results_path.exists():
            results_path.unlink()
        tb_src = self.get_template("tb_handshake.txt").render({
            "name": self.t_name,
            "vectors": repr(vecs),
            "valid_p": float(valid),
            "ready_p": float(ready),
            "seed": int(seed),
            "max_cycles": (20 + getattr(self, "ram_phases", 1)) * len(vecs) * ser["l"]
                          + 100 + 4 * self.pipeline_stages,
            "in_names": in_names,
            "out_args": out_names,
            "n_results": calls * ser["l"],
            "out_meta": {**self.out_meta, **{n: {"signed": int(any(
                self.attr.get(p, {}).get("signed", 0) for p in ports))}
                for n, ports in ser["out"].items()}},
        })
        _drop_bytecode(self.outdir, "tb_handshake")
        (self.outdir / "tb_handshake.py").write_text(tb_src, encoding="utf-8")
        tb_runner.run_cocotb(
            f"{self.t_name}_stream", [self.hdl_dir / "main.v", wrapper], "tb_handshake",
            self.outdir, build_name="sim_build_handshake", timescale=("1ns", "1ps"),
        )
        rows: List[Dict[str, int]] = []
        if results_path.exists():
            with results_path.open("rb") as handle:
                payload = pickle.load(handle)
            rows = payload["outs"]
            # Clocks from reset release to the last result taken
            self.handshake_cycles = payload["cycles"]
        if len(rows) != calls * ser["l"]:
            raise RuntimeError(
                f"Stream wrapper returned {len(rows)} results for {calls * ser['l']} expected"
            )
        if ser["out"]:
            # l element transfers -> one call's outputs
            blocks = []
            for c in range(calls):
                group = rows[c * ser["l"]:(c + 1) * ser["l"]]
                row = dict(group[0])
                for name, ports in ser["out"].items():
                    for k, port in enumerate(ports):
                        row[port] = group[k][name]
                blocks.append(row)
            rows = blocks
        return self._rows_to_outputs(rows, calls)

    def TB(self) -> Callable[..., Tuple[int, ...]]:
        return self._fx_rows(self.tb_fun, single=True)

    def TB_batch(self) -> Callable[[List[Tuple[Any, ...]]], List[Tuple[int, ...]]]:
        return self._fx_rows(self.tb_batch_fun)

    def TB_stream(self) -> Callable[[List[Any]], List[Tuple[int, ...]]]:
        return self._fx_rows(self.tb_stream_fun)

    def _render_tb(self, template: str, vecs: List[Tuple[int, ...]]) -> None:
        # Write raw bit patterns of each port's width (cocotb rejects e.g. -1
        # on a 1-bit port)
        masks = [(1 << w) - 1 for w in self.input_args_width_list]
        vecs = [tuple(v & m for v, m in zip(vec, masks)) for vec in vecs]
        tb_out = self.get_template(template).render({
            "period": DEFAULT_TB_PERIOD,
            "in_names": self.input_args_list,
            "vectors": repr(vecs),
            "out_args": self.output_args_list,
            "has_clk": self.has_clk,
            "out_meta": self.out_meta,
            "latency_cycles": int(getattr(self, "tb_latency_cycles", 0)),
            "out_latency": dict(getattr(self, "out_latency", {})),
            "pipeline_stages": self.pipeline_stages,
            "has_ce": self._has_ce,
            "ram_phases": getattr(self, "ram_phases", 1),
            "has_vin": self._has_vin,
            # With state and a pipeline, results come out this many clocks later
            "stream_latency": self.pipeline_stages if self._has_vin else 0,
        })
        (self.outdir / "tb.py").write_text(tb_out, encoding="utf-8")

    def gen_tb(self, vectors: List[Tuple[Any, ...]]) -> None:
        vecs = [self._input_vector(tuple(v)) for v in vectors]
        log.info("gen_tb: generating tb.py with %d vector(s)", len(vecs))
        self._render_tb("tb.txt", vecs)

    def gen_tb_stream(self, samples: List[Any]) -> None:
        vecs = self._stream_vectors(samples)
        log.info("gen_tb_stream: generating tb.py with %d sample(s)", len(vecs))
        self._render_tb("tb_stream.txt", vecs)

    def tb_fun(self, *in_arg: Any) -> Tuple[int, ...]:
        return self.tb_batch_fun([in_arg])[0]

    def tb_batch_fun(self, vectors: List[Tuple[Any, ...]]) -> List[Tuple[int, ...]]:
        """Simulate all input vectors in a single cocotb run, one result per vector."""
        if self.state_map or self.rams or self._seq_mems:
            raise ValueError(
                "This design has state ('next' or 'memory' in attr); use TB_stream()."
            )
        vectors = [tuple(v) for v in vectors]
        log.info("TB called with %d vector(s)", len(vectors))
        if not vectors:
            return []
        if self.schedule:              # several clocks per vector: the stream testbench
            return self.tb_stream_fun(vectors)
        native_rows = self._simulate_native([self._input_vector(v) for v in vectors])
        if native_rows is not None:
            return native_rows
        return self._simulate(len(vectors), lambda: self.gen_tb(vectors))

    def tb_stream_fun(self, samples: List[Any]) -> List[Tuple[int, ...]]:
        """
        Reset the state registers, then apply one sample per clock in a single
        cocotb run. Returns all outputs (in return order) for each sample,
        sampled before the clock edge that loads the next state.
        """
        samples = list(samples)
        self._stream_vectors(samples)  # validate before touching the filesystem
        log.info("TB stream called with %d sample(s)", len(samples))
        if not samples:
            return []
        native_rows = self._simulate_native(self._stream_vectors(samples))
        if native_rows is not None:
            return native_rows
        return self._simulate(len(samples), lambda: self.gen_tb_stream(samples))

    def _simulate_native(self, vecs: List[Tuple[int, ...]]) -> List[Tuple[int, ...]] | None:
        """The vectors through a plain Verilog testbench on Icarus or Verilator (see
        native.py), or None where the cocotb testbench is used."""
        from . import native
        clocks = len(vecs) * int(getattr(self, "ram_phases", 1)) + self.pipeline_stages
        sim = native.choose_simulator(self, clocks)
        if sim == "cocotb" or native.mode_of(self) is None:
            return None
        log.info("Simulating %d vector(s) with %s (native testbench)", len(vecs), sim)
        return self._rows_to_outputs(native.run(self, vecs, sim), len(vecs))

    def _simulate(self, n_vectors: int, write_tb: Callable[[], None]) -> List[Tuple[int, ...]]:
        """Run one cocotb simulation of the tb.py written by write_tb()."""
        results_path = self.outdir / RESULTS_PATH

        # Clean previous results
        if results_path.exists():
            try:
                results_path.unlink()
            except OSError as e:
                log.warning(e)

        # Ensure a clean sim build directory so tb.py isn't stale
        sim_build = self.outdir / "sim_build"
        if sim_build.exists():
            try:
                shutil.rmtree(sim_build)
            except OSError as e:
                log.warning("Failed to remove sim_build: %s", e)

        self.write_hdl()
        _drop_bytecode(self.outdir, "tb")
        write_tb()

        # Run cocotb-based simulation
        tb_runner.test_runner(self.t_name, self.outdir)

        # Load results written by tb.py (a list of dicts; a single dict is
        # accepted as a one-vector batch)
        rows: List[Dict[str, int]] = []
        if results_path.exists():
            with results_path.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, str):
                payload = ast.literal_eval(payload)
            rows = [payload] if isinstance(payload, dict) else list(payload)
        else:
            log.warning("No TB results found at %s", results_path)

        return self._rows_to_outputs(rows, n_vectors)

    def _rows_to_outputs(self, rows: List[Dict[str, int]], n_vectors: int) -> List[Tuple[int, ...]]:
        outs: List[Tuple[int, ...]] = []
        for i in range(n_vectors):
            d = rows[i] if i < len(rows) else {}
            row: List[int] = []
            for n, w in zip(self.output_args_list, self.output_args_width_list):
                v = d.get(n, 0)
                if self.attr.get(n, {}).get("signed") == 1:
                    v = self.tosigned(v, w)
                row.append(v)
            outs.append(tuple(row))
        return outs
