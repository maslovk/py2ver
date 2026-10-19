"""Py2ver on the DE0-Nano (a mixin of Py2ver, see core.py)."""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from . import tb_runner
from .defaults import DEFAULT_BAUD_RATE, DEFAULT_CLK_FREQ, DEFAULT_TEMPLATE_DIR
from .simulate import _drop_bytecode
from .syn import _SDRAM_SDC
from .syn_report import read_report

try:
    import serial  # pyserial
except Exception:
    serial = None  # handled gracefully later

log = logging.getLogger(__name__)

# The virtual board with the SDRAM: the DE0-Nano's top and a model of the chip on its pins
_BOARD_SIM = """`timescale 1ns/1ps
module py2ver_board(input CLOCK_50, input [1:0] KEY, input [3:0] SW, input GPIO_0_RX,
                    output GPIO_0_TX);
    wire [12:0] a; wire [1:0] ba, dqm; wire cas_n, cke, dclk, cs_n, ras_n, we_n;
    wire [15:0] dq;
    DE0_Nano board(.CLOCK_50(CLOCK_50), .KEY(KEY), .SW(SW), .LED(), .DRAM_ADDR(a), .DRAM_BA(ba),
        .DRAM_CAS_N(cas_n), .DRAM_CKE(cke), .DRAM_CLK(dclk), .DRAM_CS_N(cs_n), .DRAM_DQ(dq),
        .DRAM_DQM(dqm), .DRAM_RAS_N(ras_n), .DRAM_WE_N(we_n), .GPIO_0_RX(GPIO_0_RX),
        .GPIO_0_TX(GPIO_0_TX));
    sdram_model chip(.clk(dclk), .cke(cke), .cs_n(cs_n), .ras_n(ras_n), .cas_n(cas_n),
        .we_n(we_n), .ba(ba), .sa(a), .dqm(dqm), .dq(dq));
endmodule
"""


class BoardMixin:
    """Py2ver on the DE0-Nano: the board's project (UART, the core), the bytes over its UART, the virtual board and programming the FPGA."""

    @property
    def _hw_root(self) -> Path:
        return self.outdir / "hw"

    @property
    def _project_dir(self) -> Path:
        return self._hw_root / self.t_name

    def _pack_inputs(self, *in_arg: Any) -> bytes:
        """
        Pack input arguments into a little-endian bit-vector with per-signal
        byte-aligned segments.

        Matches DE0_Nano.v RX side:

          localparam FOO_SEG_WIDTH = ((FOO_WIDTH+7)/8)*8;

          // RX_WIDTH is sum of FOO_SEG_WIDTH
          assign { ..., input1_seg, input0_seg } = rx_data_int;
          assign input0_int = input0_seg[FOO0_WIDTH-1:0];
          assign input1_int = input1_seg[FOO1_WIDTH-1:0];

        So we do:
          - For each input i:
              * segment width SEG_WIDTH[i]
              * write value into lower 'width[i]' bits of that segment
              * upper bits remain 0
          - Concatenate segments in order inputs[0] (LSB) .. inputs[N-1] (MSB).
        """
        acc = 0
        bit_offset = 0

        for idx, (name, width, seg_width) in enumerate(
            zip(self.input_args_list,
                self.input_args_width_list,
                self.input_seg_widths)
        ):
            v = in_arg[idx] if idx < len(in_arg) else 0
            mask = (1 << width) - 1 if width > 0 else 0
            v = int(v) & mask
            # Place v in the LSBs of this segment
            acc |= (v << bit_offset)
            bit_offset += seg_width  # move by SEG_WIDTH, not raw width

        num_bytes = (self.rx_width + 7) // 8  # should already be multiple of 8
        return acc.to_bytes(num_bytes, byteorder="little")

    def _unpack_outputs(self, data: bytes) -> Tuple[int, ...]:
        """
        Unpack output bytes (little-endian) into a tuple of ints in the
        order of self.output_args_list, applying signedness.

        Matches DE0_Nano.v TX side:

          localparam FOO_SEG_WIDTH = ((FOO_WIDTH+7)/8)*8;

          assign tx_data_int = {
              { (FOO_N_SEG_WIDTH-FOO_N_WIDTH){1'b0} }, foo_n_out_int,
              ...
              { (FOO_0_SEG_WIDTH-FOO_0_WIDTH){1'b0} }, foo_0_out_int
          };

        i.e., segments concatenated outputs[0] at LSB, outputs[1] above, etc.
        """
        if not data:
            return tuple(0 for _ in self.output_args_list)

        acc = int.from_bytes(data, byteorder="little")
        outs: List[int] = []
        bit_offset = 0

        for name, width, seg_width in zip(
            self.output_args_list,
            self.output_args_width_list,
            self.output_seg_widths
        ):
            seg_mask = (1 << seg_width) - 1 if seg_width > 0 else 0
            seg = (acc >> bit_offset) & seg_mask
            bit_offset += seg_width

            # Extract the actual value from the LSB bits of the segment
            mask_val = (1 << width) - 1 if width > 0 else 0
            v = seg & mask_val

            if self.attr.get(name, {}).get("signed") == 1:
                v = self.tosigned(v, width)
            outs.append(v)

        return tuple(outs)

    def HW(self, syn_attr: Dict[str, Any]) -> Callable[..., Tuple[int, ...]]:
        """
        Synthesis / hardware flow.

        syn_attr must contain:
            'QUARTUS_DIR': path to quartus/bin
        and may contain:
            'UART_PORT':    serial port (default '/dev/ttyUSB0')
            'UART_BAUD':    baud rate (default DEFAULT_BAUD_RATE)
            'UART_TIMEOUT': read timeout in seconds (default 1.0)
            'FALLBACK_TO_SIM': return simulation results instead of raising
                               when the board cannot be used (default False)
            'ALLOW_TIMING_FAILURE': run even if the board design misses
                               50 MHz timing (results may be wrong)

        Returns:
            hw_fun(*args) -> tuple of outputs (same shape as TB()). For designs
            with state, each call is one sample and advances the state.
        """
        self.syn_attr = syn_attr
        self._sdram_loaded = False                   # (the SDRAM's tables: loaded on the first call)
        self._hw_ok = self.createHWproject()
        if not self._hw_ok:
            self._hw_unavailable(getattr(self, "_hw_error", "board project could not be built (see log)"))
        return self._fx_rows(self.hw_fun, single=True)

    def _hw_unavailable(self, reason: str, *in_arg: Any) -> Tuple[int, ...]:
        """Raise, or fall back to simulation if FALLBACK_TO_SIM is set."""
        if self.syn_attr.get("FALLBACK_TO_SIM") and not self.state_map:
            log.warning("HW unavailable (%s); returning simulation result.", reason)
            return self.tb_fun(*in_arg) if in_arg else ()
        raise RuntimeError(f"HW unavailable: {reason}")

    def hw_fun(self, *in_arg: Any) -> Tuple[int, ...]:
        """
        Hardware runner.

        Flow:
          - If Quartus or UART is unavailable, fall back to tb_fun().
          - Otherwise:
              * pack inputs into RX_WIDTH bits (byte-aligned segments)
              * write RX_BYTES to UART
              * read back TX_BYTES
              * unpack to tuple of outputs
        """
        if not getattr(self, "_hw_ok", False):
            return self._hw_unavailable(
                getattr(self, "_hw_error", "board project could not be built"), *in_arg
            )

        if serial is None:
            return self._hw_unavailable("pyserial is not installed", *in_arg)

        port = self.syn_attr.get("UART_PORT", "/dev/ttyUSB0")
        baud = int(self.syn_attr.get("UART_BAUD", DEFAULT_BAUD_RATE))
        timeout = float(self.syn_attr.get("UART_TIMEOUT", 1.0))

        if self._sdram:
            return self._hw_sdram_call(port, baud, timeout, in_arg)

        input_bytes = self._pack_inputs(*self._flatten_args(in_arg))
        output_bytes_expected = (self.tx_width + 7) // 8

        try:
            with serial.Serial(port, baudrate=baud, timeout=timeout) as ser:
                log.info("Opened UART port %s @ %d baud", port, baud)

                try:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                except Exception:
                    pass

                # --- Send input payload ---
                log.debug("Sending %d input bytes: %s",
                          len(input_bytes), input_bytes.hex(" "))
                ser.write(input_bytes)
                ser.flush()

                # --- Read response ---
                resp = b""
                while len(resp) < output_bytes_expected:
                    chunk = ser.read(output_bytes_expected - len(resp))
                    if not chunk:
                        break
                    resp += chunk

                if len(resp) == 0:
                    log.error("UART: No data received.")
                else:
                    log.info("Received %d bytes: %s",
                             len(resp), resp.hex(" "))

                if len(resp) != output_bytes_expected:
                    return self._hw_unavailable(
                        f"UART: expected {output_bytes_expected} bytes, got {len(resp)}",
                        *in_arg,
                    )

                # --- Decode ---
                outs = self._unpack_outputs(resp)
                log.info("Decoded HW outputs: %s", outs)
                return outs

        except (OSError, serial.SerialException) as e:
            return self._hw_unavailable(f"UART communication failed ({e})", *in_arg)

    def _hw_sdram_call(self, port: str, baud: int, timeout: float, in_arg) -> Tuple[int, ...]:
        """A sample on a board with the SDRAM: its tables (once), its input lists,
        then 'R' (see de0_nano_sdram_top.txt)."""
        first = not getattr(self, "_sdram_loaded", False)
        steps = self._sdram_steps(self._input_vector(tuple(in_arg)), first)
        try:
            with serial.Serial(port, baudrate=baud, timeout=max(timeout, 30.0)) as ser:
                replies = []
                for data, n in steps:
                    ser.write(data)
                    ser.flush()
                    replies.append(ser.read(n))
                    if len(replies[-1]) != n:
                        return self._hw_unavailable(
                            f"UART: expected {n} byte(s), got {len(replies[-1])}", *in_arg)
                self._sdram_loaded = True
                return self._sdram_results(replies)
        except (OSError, serial.SerialException) as e:
            return self._hw_unavailable(f"UART communication failed ({e})", *in_arg)

    def hw_run(self) -> int:
        """
        Legacy HW runner placeholder (kept for compatibility).
        Not used by the new HW()/hw_fun path.
        """
        return 1

    def HW_sim(self) -> Callable[[List[Tuple[Any, ...]]], List[Tuple[int, ...]]]:
        """
        Virtual board: simulate the full DE0-Nano project (UART, latency
        handling, core) and talk to it over its UART pins with the same bytes
        hw_fun() sends. Returns a batch callable like TB_batch().
        """
        return self._fx_rows(self.hw_sim_fun)

    def hw_sim_fun(self, vectors: List[Tuple[Any, ...]]) -> List[Tuple[int, ...]]:
        vectors = [tuple(v) for v in vectors]
        if not vectors:
            return []
        if self._sdram:
            return self._hw_sim_sdram(vectors)
        project_dir = self.gen_hw_project()
        tx_bytes = (self.tx_width + 7) // 8
        requests = [list(self._pack_inputs(*self._flatten_args(v))) for v in vectors]

        tb_src = self.get_template("tb_hw.txt").render({
            "baud": DEFAULT_BAUD_RATE,
            "tx_bytes": tx_bytes,
            "requests": repr(requests),
        })
        _drop_bytecode(project_dir, "tb_hw")
        (project_dir / "tb_hw.py").write_text(tb_src, encoding="utf-8")

        sources = (
            [project_dir / "DE0_Nano.v", project_dir / "delayed_registers.sv",
             project_dir / f"{self.t_name}.v"]
            + sorted((project_dir / "uart").glob("*.sv"))
        )
        tb_runner.run_cocotb(
            "DE0_Nano", sources, "tb_hw", project_dir, timescale=("1ns", "1ps")
        )

        results_path = project_dir / "results" / "results.pickle"
        if not results_path.exists():
            raise RuntimeError(f"Virtual board produced no results ({results_path})")
        with results_path.open("rb") as handle:
            replies = pickle.load(handle)

        outs: List[Tuple[int, ...]] = []
        for vec, reply in zip(vectors, replies):
            if len(reply) != tx_bytes or not all(isinstance(b, int) for b in reply):
                raise RuntimeError(
                    f"Virtual board reply for {vec}: expected {tx_bytes} bytes, got {reply}"
                )
            outs.append(self._unpack_outputs(bytes(reply)))
        return outs

    SIM_BAUD = 3_125_000        # the virtual SDRAM board's UART (16 clocks a bit at 50 MHz)

    def _hw_sim_sdram(self, vectors: List[Tuple[Any, ...]]) -> List[Tuple[int, ...]]:
        """The virtual board with the SDRAM: the board's project and a model of the
        chip on its pins, the same UART commands as hw_fun() (at SIM_BAUD)."""
        project_dir = self.gen_hw_project(baud=self.SIM_BAUD)
        samples = [[(list(d), n) for d, n in self._sdram_steps(self._input_vector(v), k == 0)]
                   for k, v in enumerate(vectors)]
        P = int(self.ram_phases)
        tb_src = self.get_template("tb_hw_sdram.txt").render({
            "baud": self.SIM_BAUD, "samples": repr(samples), "wait_bits": (4 * P + 50000) // 128})
        _drop_bytecode(project_dir, "tb_hw_sdram")
        (project_dir / "tb_hw_sdram.py").write_text(tb_src, encoding="utf-8")
        model = DEFAULT_TEMPLATE_DIR / "hw" / "sdram" / "sdram_model.v"
        board = project_dir / "py2ver_board.v"
        board.write_text(_BOARD_SIM, encoding="utf-8")
        sources = ([board, project_dir / "DE0_Nano.v", project_dir / f"{self.t_name}.v", model]
                   + sorted((project_dir / "uart").glob("*.sv")))
        tb_runner.run_cocotb("py2ver_board", sources, "tb_hw_sdram", project_dir,
                             timescale=("1ns", "1ps"))
        with (project_dir / "results" / "results.pickle").open("rb") as handle:
            replies = pickle.load(handle)
        outs = []
        for vec, steps, reply in zip(vectors, samples, replies):
            want = [n for _, n in steps]
            acks_ok = all(r == [0x57] for (d, _), r in zip(steps, reply) if d[0] == 0x57)
            if [len(r) for r in reply] != want or not acks_ok:
                raise RuntimeError(f"Virtual board replies for {vec}: expected {want} bytes "
                                   f"('W' acks, the outputs, the returned lists), got {reply}")
            outs.append(self._sdram_results([bytes(r) for r in reply]))
        return outs

    def gen_hw_project(self, baud: int = DEFAULT_BAUD_RATE) -> Path:
        """Write the DE0-Nano Quartus project (board top, UART, core). No compile."""
        project_dir = self._project_dir
        if project_dir.is_dir():
            shutil.rmtree(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        project_name = self.t_name

        (project_dir / f"{project_name}.qpf").write_text(
            f"PROJECT_REVISION = {project_name}\n", encoding="utf-8"
        )

        qsf_path = project_dir / f"{project_name}.qsf"
        self.get_template("hw/de0_nano_qsf.txt").stream({}).dump(str(qsf_path))

        # ---------------------- Emit UART split modules ----------------------
        uart_dir = project_dir / "uart"
        uart_dir.mkdir(exist_ok=True)

        self.get_template("hw/uart/uart_baud_gen.txt") \
            .stream({}).dump(str(uart_dir / "uart_baud_gen.sv"))
        self.get_template("hw/uart/uart_rx_8n1.txt") \
            .stream({}).dump(str(uart_dir / "uart_rx_8n1.sv"))
        self.get_template("hw/uart/uart_tx_8n1.txt") \
            .stream({}).dump(str(uart_dir / "uart_tx_8n1.sv"))
        self.get_template("hw/uart/uart_fifo4x8.txt") \
            .stream({}).dump(str(uart_dir / "uart_fifo4x8.sv"))
        # PRBS7 TX test leaf
        self.get_template("hw/uart/uart_prbs7_tx_test.txt") \
            .stream({}).dump(str(uart_dir / "uart_prbs7_tx_test.sv"))
        # Fixed-pattern TX generator
        self.get_template("hw/uart/uart_fixed_pattern_tx.txt") \
            .stream({}).dump(str(uart_dir / "uart_fixed_pattern_tx.sv"))
        # Loopback wrapper
        self.get_template("hw/uart/uart_loopback.txt") \
            .stream({
                "CLK_HZ": DEFAULT_CLK_FREQ,
                "BAUD": DEFAULT_BAUD_RATE
            }).dump(str(uart_dir / "uart_loopback.sv"))

        # uart_transceiver module
        self.get_template("hw/uart/uart_transceiver_top.txt") \
            .stream({}).dump(str(uart_dir / "uart_transceiver.sv"))

        # Append UART leafs to QSF
        with qsf_path.open("a", encoding="utf-8") as qsf:
            qsf.write("\n# UART split modules\n")
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_baud_gen.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_rx_8n1.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_tx_8n1.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_fifo4x8.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_prbs7_tx_test.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_fixed_pattern_tx.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_loopback.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_transceiver.sv\n')

        # SDC / wrappers / board top
        self.get_template("hw/de0_nano_sdc.txt") \
            .stream({}).dump(str(project_dir / "DE0_Nano.SDC"))

        if self._sdram:
            # A design with the SDRAM: the board's top takes commands (see the template)
            self.get_template("hw/de0_nano_sdram_top.txt").stream(
                self._sdram_board(baud)).dump(str(project_dir / "DE0_Nano.v"))
            with open(project_dir / "DE0_Nano.SDC", "a", encoding="utf-8") as f:
                f.write(_SDRAM_SDC.replace("[get_ports clk]", "[get_ports CLOCK_50]"))
        else:
            self.get_template("hw/delayed_registers.txt") \
                .stream(self.syn_data).dump(str(project_dir / "delayed_registers.sv"))

            # DE0_Nano top you provided
            self.get_template("hw/de0_nano_top.txt") \
                .stream(self.syn_data).dump(str(project_dir / "DE0_Nano.v"))

        # Copy generated core HDL
        self._copy_hdl(project_dir, f"{project_name}.v")

        log.info("Quartus project created at %s", project_dir)

        return project_dir

    def createHWproject(self) -> bool:
        """
        Create Quartus project, emit all HDL, run compile, and attempt
        to program the FPGA.

        Returns:
            True on successful compile (programming errors are logged but
            do not flip this to False), False on any early/compile error.
        """
        syn_tool_dir = self.syn_attr.get("QUARTUS_DIR")
        if not syn_tool_dir:
            self._hw_error = "missing 'QUARTUS_DIR'"
            log.error(self._hw_error)
            return False

        quartus_sh = os.path.join(syn_tool_dir, "quartus_sh")
        if not os.path.exists(quartus_sh):
            self._hw_error = f"quartus_sh not found: {quartus_sh}"
            log.error(self._hw_error)
            return False

        project_dir = self.gen_hw_project(int(self.syn_attr.get("UART_BAUD", DEFAULT_BAUD_RATE)))
        project_name = self.t_name

        # ---- Compile ----
        log.info("Running Quartus compile...")
        result = subprocess.run(
            [quartus_sh, "--flow", "compile", project_name],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self._hw_error = f"Quartus compile failed (rc={result.returncode}); see log"
            log.error("Quartus compile failed: rc=%d", result.returncode)
            log.error("quartus_sh stdout:\n%s", result.stdout)
            log.error("quartus_sh stderr:\n%s", result.stderr)
            return False

        # Board reports are written to the project directory
        self.hw_report = read_report(
            project_dir, project_name, self.t_name,
            1e9 / DEFAULT_CLK_FREQ, out_dir=project_dir,
        )
        log.info("Board build:\n%s", self.hw_report)
        if self.hw_report.meets_timing is False and not self.syn_attr.get("ALLOW_TIMING_FAILURE"):
            self._hw_error = (
                f"board design misses {DEFAULT_CLK_FREQ // 1_000_000} MHz timing "
                f"(Fmax {self.hw_report.fmax_mhz} MHz), so results would be unreliable; "
                "use pipeline=N, or set ALLOW_TIMING_FAILURE"
            )
            log.error(self._hw_error)
            return False

        self._program_fpga(syn_tool_dir, project_dir, project_name)
        return True

    def _sdram_board(self, baud: int) -> Dict[str, Any]:
        """The SDRAM board top's template context."""
        ins, lo = [], 0
        for n, w in self._port_inputs():
            ins.append({"name": n, "width": w, "lo": lo})
            lo += (w + 7) // 8 * 8
        outs = [{"name": n, "width": w, "seg": (w + 7) // 8 * 8}
                for n, w in zip(self.output_args_list, self.output_args_width_list)
                if n not in self._ext_outputs]
        # (a reply to 'R' of a byte at least: the sample done, its lists ready to read)
        return {"core": self.t_name, "inputs": ins, "outputs": outs, "rx_bytes": lo // 8,
                "tx_bytes": max(1, sum(o["seg"] for o in outs) // 8), "phases": int(self.ram_phases),
                "baud": int(baud)}

    def _sdram_steps(self, vec: Tuple[int, ...], first: bool) -> List[Tuple[bytes, int]]:
        """The UART commands of a sample (vec: its raw inputs, as input_args_list):
        the tables first (first=True), its input lists into the SDRAM, then 'R'.
        [(bytes, bytes of the reply)]."""
        from .native import sdram_images, _runs
        ext = getattr(self, "_ext_inputs", {})
        steps: List[Tuple[bytes, int]] = []

        def write(base: int, words: List[int]) -> None:
            for at in range(0, len(words), 4096):
                part = words[at:at + 4096]
                head = b"W" + (base + at).to_bytes(3, "big") + len(part).to_bytes(3, "big")
                steps.append((head + b"".join((w & 0xFFFF).to_bytes(2, "big") for w in part), 1))
        if first:
            for base, values in sdram_images(self):
                write(base, values)
        names = list(self.input_args_list)
        addrs = [ext[n] for n in names if n in ext]
        values = [int(vec[i]) for i, n in enumerate(names) if n in ext]
        for start, base, count in _runs(addrs):
            write(base, values[start:start + count])
        acc, off = 0, 0
        port = {n for n, _ in self._port_inputs()}
        for i, (n, w) in enumerate(zip(names, self.input_args_width_list)):
            if n in port:
                acc |= (int(vec[i]) & ((1 << w) - 1)) << off
                off += (w + 7) // 8 * 8
        steps.append((b"R" + acc.to_bytes(off // 8, "little"), self._sdram_board(0)["tx_bytes"]))
        for base, count in self._sdram_reads():                 # (the returned lists)
            steps.append((b"X" + base.to_bytes(3, "big") + count.to_bytes(3, "big"), 2 * count))
        return steps

    def _sdram_reads(self) -> List[Tuple[int, int]]:
        """(address, words) of the 'X' reads of a sample's returned lists in the SDRAM."""
        from .native import _runs
        addrs = [self._ext_outputs[n] for n in self.output_args_list if n in self._ext_outputs]
        return [(base + at, min(4096, count - at)) for _, base, count in _runs(addrs)
                for at in range(0, count, 4096)]

    def _sdram_results(self, replies: List[bytes]) -> Tuple[int, ...]:
        """A sample's outputs from the board's replies to its steps: the ports' from
        'R' (the UART's segments), the returned lists' from the 'X' reads after it."""
        nx = len(self._sdram_reads())
        acc = int.from_bytes(replies[len(replies) - 1 - nx], "little")
        words = b"".join(replies[len(replies) - nx:])
        values: Dict[str, int] = {}
        off = 0
        ext = [n for n in self.output_args_list if n in self._ext_outputs]
        for n, w in zip(self.output_args_list, self.output_args_width_list):
            if n not in self._ext_outputs:
                values[n] = (acc >> off) & ((1 << w) - 1)
                off += (w + 7) // 8 * 8
        widths = dict(zip(self.output_args_list, self.output_args_width_list))
        for i, n in enumerate(ext):
            values[n] = int.from_bytes(words[2 * i:2 * i + 2], "big") & ((1 << widths[n]) - 1)
        return tuple(self.tosigned(values[n], widths[n]) if self.attr.get(n, {}).get("signed") == 1
                     else values[n] for n in self.output_args_list)

    @staticmethod
    def _program_fpga(syn_tool_dir: str, project_dir: Path, project_name: str) -> None:
        out_dir = project_dir / "output_files"
        candidates = [
            out_dir / f"{project_name}.sof",
            next(iter(out_dir.glob("*.sof")), None) if out_dir.exists() else None,
            project_dir / f"{project_name}.sof",
        ]
        sof_file = next((p for p in candidates if isinstance(p, Path) and p and p.exists()), None)

        if not sof_file:
            log.warning("No .sof found, skipping programming.")
            return

        quartus_pgm = os.path.join(syn_tool_dir, "quartus_pgm")
        if not os.path.exists(quartus_pgm):
            log.error("quartus_pgm not found at %s", quartus_pgm)
            return

        log.info("Programming FPGA with %s...", sof_file)
        result = subprocess.run(
            [quartus_pgm, "-m", "jtag", "-o", f"p;{sof_file}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.error("quartus_pgm failed: %s", result.stderr)
        else:
            log.info("FPGA programmed successfully ✅")
