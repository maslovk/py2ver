"""
A design on the DE0-Nano's own pins (a mixin of Py2ver, see core.py).

A port's attr names the board's pin it goes to, as the board's pin assignments
(templates/hw/de0_nano_qsf.txt) call it:

    attr = {'rx': 'GPIO_0_RX',                  # (a pin, its width the pins')
            'data': 'LED[7:0]',                 # bit 0 on LED[0]
            ...}
    p = Py2ver(uart_rx, attr)
    p.PINS({'QUARTUS_DIR': ...})       # compiled for the board, not programmed

The board's top (DE0_Nano) clocks the design with CLOCK_50, holds it in reset
at power-up and while KEY[0] is pressed (unless a port takes KEY[0]), its clock
enable high; an input pin reaches its port through two registers (a pin
changes whenever it likes: the second register's value is settled), an output
port drives its pin; a GPIO pin no output drives floats. Ports without a pin:
inputs 0, outputs unconnected.
"""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .defaults import DEFAULT_SYN_PERIOD_NS, DEFAULT_TEMPLATE_DIR
from .syn_report import SynReport

log = logging.getLogger(__name__)

# (the board's pins by kind: driven by the design, read, or either)
_OUT_GROUPS = {"LED"}
_IN_GROUPS = {"KEY", "SW", "GPIO_0_IN", "GPIO_1_IN", "GPIO_2_IN"}

# The JTAG UART's pins: bytes to and from the PC over the USB-Blaster (juart-terminal),
# each (its width, the design's input or output, its wire in the board's top)
_JTAG = {"JTAG_RX_DATA": (8, "input", "jtag_rx_data"),      # a byte from the PC
         "JTAG_RX_VALID": (1, "input", "jtag_rx_valid"),    # (high: JTAG_RX_DATA is one)
         "JTAG_RX_READY": (1, "output", "jtag_rx_ready"),   # (optional: the design takes it)
         "JTAG_TX_DATA": (8, "output", "jtag_tx_data"),     # a byte to the PC
         "JTAG_TX_VALID": (1, "output", "jtag_tx_valid"),   # (high a clock: send it)
         "JTAG_TX_READY": (1, "input", "jtag_tx_ready")}    # (high: there's room for one)
JTAG_RX_FIFO, JTAG_TX_FIFO = 8, 16                           # bytes each way


def board_pins() -> Dict[str, List[int]]:
    """The DE0-Nano's pins as its assignments name them: a group's indices (LED:
    0..7; GPIO_0: 2..33, its pins 0 and 1 being GPIO_0_RX and GPIO_0_TX), a
    single pin's none."""
    qsf = (DEFAULT_TEMPLATE_DIR / "hw" / "de0_nano_qsf.txt").read_text(encoding="utf-8")
    pins: Dict[str, List[int]] = {}
    for m in re.finditer(r"set_location_assignment\s+\S+\s+-to\s+(\w+)(?:\[(\d+)\])?", qsf):
        name, idx = m.group(1), m.group(2)
        pins.setdefault(name, [])
        if idx is not None:
            pins[name].append(int(idx))
    return {k: sorted(v) for k, v in pins.items()}


def _spec(spec: str, pins: Dict[str, List[int]]):
    m = re.fullmatch(r"(\w+)(?:\[(\d+)(?::(\d+))?\])?", spec.strip())
    if not m or m.group(1) not in pins:
        raise ValueError(f"No DE0-Nano pin {spec!r} (pins: {', '.join(sorted(pins))}; the "
                         f"JTAG UART's: {', '.join(_JTAG)})")
    return m.group(1), m.group(2), m.group(3)


def pin_width(spec: str) -> int:
    """How many pins a pin attr names: 'LED[7:0]' 8, 'LED[2]' 1, 'LED' its group's
    8, 'GPIO_0_TX' 1; the JTAG UART's: JTAG_RX_DATA 8."""
    if spec.strip() in _JTAG:
        return _JTAG[spec.strip()][0]
    name, a, b = _spec(spec, board_pins())
    if a is None:
        return max(1, len(board_pins()[name]))
    return 1 if b is None else abs(int(a) - int(b)) + 1


def parse_pin(spec: str, width: int, pins: Dict[str, List[int]]) -> List[str]:
    """A port's pin attr as one pin a bit, its bit 0 first: 'LED[0]', 'LED[3:0]' (bit 0
    on LED[0]), 'LED' (the group from its first pin), 'GPIO_0_TX' (a single pin);
    a JTAG UART pin (JTAG_RX_DATA, ...) as itself."""
    if spec.strip() in _JTAG:
        need = _JTAG[spec.strip()][0]
        if width != need:
            raise ValueError(f"{spec}: {need} bit{'s' if need > 1 else ''} for a port of {width}")
        return [spec.strip()]
    name, a, b = _spec(spec, pins)
    have = pins[name]
    if not have:                                       # (a single pin)
        if a is not None or width != 1:
            raise ValueError(f"{name} is a single pin: a port of 1 bit")
        return [name]
    if a is None:
        if width > len(have):
            raise ValueError(f"{spec}: {len(have)} pins for a port of {width} bits (give it "
                             f"{width} pins, or the port its width in attr)")
        lo, hi = have[0], have[0] + width - 1
    elif b is None:
        lo = hi = int(a)
    else:
        hi, lo = int(a), int(b)
    if hi < lo:
        raise ValueError(f"{spec}: [high:low]")
    if hi - lo + 1 != width:
        raise ValueError(f"{spec}: {hi - lo + 1} pins for a port of {width} bits (give it "
                         f"{width} pins, or the port its width in attr)")
    missing = [i for i in range(lo, hi + 1) if i not in have]
    if missing:
        singles = [k for k, v in pins.items() if not v and k.startswith(name + "_")]
        raise ValueError(f"{spec}: the board has no {name}[{missing[0]}] ({name} is "
                         f"{name}[{have[-1]}:{have[0]}]"
                         + (f"; its other pins: {', '.join(singles)}" if singles else "") + ")")
    return [f"{name}[{i}]" for i in range(lo, hi + 1)]


def _jtag_uart(used: Dict[str, str]) -> List[str]:
    """The JTAG UART in the board's top: Intel's alt_jtag_atlantic (its handshake as
    the JTAG UART IP's: a byte to the PC on r_dat with r_val the clock after
    r_ena; one from the PC on t_dat while t_ena, t_dav saying there's room) and a
    FIFO each way. From the PC: a byte delivered while the design's
    JTAG_RX_READY is high (always, if no port takes it), JTAG_RX_VALID high that
    clock; to the PC: JTAG_TX_DATA taken each clock JTAG_TX_VALID is high,
    JTAG_TX_READY high while there's room (for two: the design's outputs come a
    clock late). used: the JTAG pins the design takes."""
    rx, tx = JTAG_RX_FIFO, JTAG_TX_FIFO
    rb, tb = (rx - 1).bit_length(), (tx - 1).bit_length()
    defaults = {"JTAG_RX_READY": "1'b1", "JTAG_TX_DATA": "8'd0", "JTAG_TX_VALID": "1'b0"}
    return [f"    // The JTAG UART: bytes to and from the PC over the USB-Blaster "
            f"(juart-terminal); {rx} bytes",
            f"    // waiting from the PC at most, {tx} to it"] + [
        f"    assign {_JTAG[k][2]} = {v};" for k, v in defaults.items() if k not in used] + [
        "    wire jtag_t_ena, jtag_r_ena, jtag_t_pause;",
        "    wire [7:0] jtag_t_dat;",
        "    reg jtag_t_dav = 1'b0, jtag_r_val = 1'b0;",
        "    reg [7:0] jtag_r_dat = 8'd0;",
        "    alt_jtag_atlantic #(.INSTANCE_ID(0), .LOG2_RXFIFO_DEPTH(6), .LOG2_TXFIFO_DEPTH(6),",
        "                        .SLD_AUTO_INSTANCE_INDEX(\"YES\")) u_jtag_uart (",
        "        .clk(clk), .rst_n(!rst),",
        "        .r_dat(jtag_r_dat), .r_val(jtag_r_val), .r_ena(jtag_r_ena),",
        "        .t_dat(jtag_t_dat), .t_dav(jtag_t_dav), .t_ena(jtag_t_ena), .t_pause(jtag_t_pause));",
        "    // (from the PC)",
        f"    reg [7:0] jrx_mem [0:{rx - 1}];",
        f"    reg [{rb - 1}:0] jrx_head = 0, jrx_tail = 0;",
        f"    reg [{rb}:0] jrx_count = 0;",
        f"    wire jrx_put = jtag_t_ena && jrx_count != {rx};",
        "    assign jtag_rx_valid = jrx_count != 0 && jtag_rx_ready;",
        "    assign jtag_rx_data = jrx_mem[jrx_head];",
        "    always @(posedge clk) begin",
        "        if (jrx_put) begin jrx_mem[jrx_tail] <= jtag_t_dat; jrx_tail <= jrx_tail + 1'b1; end",
        "        if (jtag_rx_valid) jrx_head <= jrx_head + 1'b1;",
        "        jrx_count <= jrx_count + jrx_put - jtag_rx_valid;",
        f"        jtag_t_dav <= (jrx_count + jrx_put - jtag_rx_valid) <= {rx - 2};   "
        f"// (room for two: t_dav comes a clock late)",
        "        if (rst) begin jrx_head <= 0; jrx_tail <= 0; jrx_count <= 0; jtag_t_dav <= 1'b0; end",
        "    end",
        "    // (to the PC)",
        f"    reg [7:0] jtx_mem [0:{tx - 1}];",
        f"    reg [{tb - 1}:0] jtx_head = 0, jtx_tail = 0;",
        f"    reg [{tb}:0] jtx_count = 0;",
        f"    assign jtag_tx_ready = jtx_count <= {tx - 2};",
        f"    wire jtx_put = jtag_tx_valid && jtx_count != {tx};",
        "    wire jtx_take = jtag_r_ena && jtx_count != 0;",
        "    always @(posedge clk) begin",
        "        if (jtx_put) begin jtx_mem[jtx_tail] <= jtag_tx_data; jtx_tail <= jtx_tail + 1'b1; end",
        "        jtag_r_val <= jtx_take;",
        "        if (jtx_take) begin jtag_r_dat <= jtx_mem[jtx_head]; jtx_head <= jtx_head + 1'b1; end",
        "        jtx_count <= jtx_count + jtx_put - jtx_take;",
        "        if (rst) begin jtx_head <= 0; jtx_tail <= 0; jtx_count <= 0; jtag_r_val <= 1'b0; end",
        "    end"]


class PinsMixin:
    """Py2ver on the DE0-Nano's own pins: PINS() and its project."""

    _pins: Dict[str, str] = {}

    @property
    def _pins_dir(self) -> Path:
        return self.outdir / "syn" / f"{self.t_name}_pins"

    def _pin_map(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """({input port: its pins}, {output port: its pins})."""
        pins = board_pins()

        def parse(p) -> List[str]:
            try:
                return parse_pin(self._pins[p.name], p.width, pins)
            except ValueError as e:
                bits = "1 bit" if p.width == 1 else f"{p.width} bits"
                raise ValueError(f"the port '{p.name}' ({bits}): {e}") from None
        ins = {p.name: parse(p) for p in self.ir.inputs
               if p.kind == "input" and p.name in self._pins}
        outs = {p.name: parse(p) for p in self.ir.outputs if p.name in self._pins}
        unknown = set(self._pins) - set(ins) - set(outs)
        if unknown:
            raise ValueError(f"A pin for no port: {sorted(unknown)} (ports: "
                             f"{[p.name for p in self.ir.inputs + self.ir.outputs]})")
        opens = getattr(self, "_opens", {})
        taken: Dict[str, str] = {}
        for port, ps in {**ins, **outs}.items():
            for pin in ps:
                # (an open-drain pin: its drive and its line on one pin)
                if pin in taken and opens.get(taken[pin]) != port and opens.get(port) != taken[pin]:
                    raise ValueError(f"{pin} is both {taken[pin]}'s and {port}'s")
                taken[pin] = port
        for port, ps in outs.items():
            for pin in ps:
                if pin.split("[")[0] in _IN_GROUPS:
                    raise ValueError(f"{pin} is an input pin: not for the output {port}")
                if port in opens and pin.split("[")[0] in _OUT_GROUPS:
                    raise ValueError(f"{pin} can't float: not for the open-drain pin {port} (a "
                                     f"GPIO pin, or I2C_SDAT / I2C_SCLK)")
        for port, ps in ins.items():
            for pin in ps:
                if pin.split("[")[0] in _OUT_GROUPS:
                    raise ValueError(f"{pin} is an output pin: not for the input {port}")
        # (the JTAG UART's pins: each an input of the design's, or an output)
        for kind, ports in (("input", ins), ("output", outs)):
            for port, ps in ports.items():
                if ps and ps[0] in _JTAG and _JTAG[ps[0]][1] != kind:
                    raise ValueError(f"{ps[0]} is for an {_JTAG[ps[0]][1]} of the design's: not "
                                     f"for the {kind} {port}")
        return ins, outs

    def board_top(self, signaltap: Any = None) -> str:
        """The Verilog of DE0_Nano: the board's pins, the design between them
        (signaltap: SignalTap recording them too, see _signaltap)."""
        if self._sdram:
            raise ValueError("A design with the SDRAM goes on the board with HW()")
        ins, outs = self._pin_map()
        pins = board_pins()
        jtag = {ps[0]: port for port, ps in {**ins, **outs}.items() if ps and ps[0] in _JTAG}
        groups = sorted({pin.split("[")[0] for ps in [*ins.values(), *outs.values()] for pin in ps
                         if pin not in _JTAG}
                        | ({"KEY"} if "KEY[0]" not in {p for ps in ins.values() for p in ps}
                           else set()))
        opens = getattr(self, "_opens", {})
        open_pins = {pin for o in opens for pin in outs.get(o, [])}
        decl = ["    input CLOCK_50"]
        for g in groups:
            idx = pins[g]
            rng = f"[{idx[-1]}:{idx[0]}] " if idx else ""
            kind = ("output" if g in _OUT_GROUPS else "input" if g in _IN_GROUPS
                    else "inout" if not idx and g in open_pins
                    else "output" if not idx and any(g in ps for ps in outs.values())
                    else "input" if not idx else "inout")
            decl.append(f"    {kind} {rng}{g}")
        own_key = "KEY[0]" in {p for ps in ins.values() for p in ps}
        lines = ["// The DE0-Nano with the design between its pins (py2ver PINS())",
                 "module DE0_Nano(", ",\n".join(decl), ");",
                 "    wire clk = CLOCK_50;",
                 "    // (held in reset at power-up" + (")" if own_key else ", and while KEY[0] is pressed)"),
                 "    reg [3:0] por = 4'd0;",
                 "    always @(posedge clk) if (por != 4'd15) por <= por + 4'd1;",
                 f"    wire rst = (por != 4'd15){'' if own_key else ' || !KEY[0]'};"]
        if jtag:
            lines += ["    // (the JTAG UART's side of the design)"]
            lines += [f"    wire {f'[{w - 1}:0] ' if w > 1 else ''}{wire};"
                      for w, _, wire in _JTAG.values()]
        conns = [".clk(clk)", ".rst(rst)", ".ce(1'b1)"] if self.has_clk else []
        for p in self.ir.inputs:
            if p.kind != "input":
                continue
            rng = f"[{p.width - 1}:0] " if p.width > 1 else ""
            if p.name in ins and ins[p.name][0] in _JTAG:   # (the JTAG UART's: its clock)
                conns.append(f".{p.name}({_JTAG[ins[p.name][0]][2]})")
            elif p.name in ins:
                bits = ins[p.name]
                cat = bits[0] if len(bits) == 1 else "{" + ", ".join(reversed(bits)) + "}"
                idle = 1 if p.name in opens.values() else 0      # (a line: high, let go)
                lines += [f"    // {p.name} from {self._pins[p.name]}, through two registers",
                          f"    reg {rng}{p.name}_s1 = {idle}, {p.name}_s2 = {idle};",
                          f"    always @(posedge clk) begin {p.name}_s1 <= {cat}; "
                          f"{p.name}_s2 <= {p.name}_s1; end"]
                conns.append(f".{p.name}({p.name}_s2)")
            else:
                conns.append(f".{p.name}({p.width}'d0)")
        for p in self.ir.outputs:
            rng = f"[{p.width - 1}:0] " if p.width > 1 else ""
            lines.append(f"    wire {rng}{p.name}_w;")
            conns.append(f".{p.name}({p.name}_w)")
            for i, pin in enumerate(outs.get(p.name, [])):
                if pin in _JTAG:                    # (to the JTAG UART)
                    lines.append(f"    assign {_JTAG[pin][2]} = {p.name}_w;")
                    continue
                if p.name in opens:                 # (open drain: low, or let go)
                    lines.append(f"    assign {pin} = {p.name}_w ? 1'bz : 1'b0;")
                    continue
                lines.append(f"    assign {pin} = {p.name}_w" + (f"[{i}];" if p.width > 1 else ";"))
        # (a GPIO pin no output drives: floating; an LED: off)
        driven = {pin for ps in outs.values() for pin in ps}
        for g in groups:
            if g in _OUT_GROUPS:
                lines += [f"    assign {g}[{i}] = 1'b0;" for i in pins[g] if f"{g}[{i}]" not in driven]
        for g in groups:
            if g.startswith("GPIO") and g not in _IN_GROUPS and pins[g]:
                free = [i for i in pins[g] if f"{g}[{i}]" not in driven]
                if len(free) == len(pins[g]):
                    lines.append(f"    assign {g} = {len(pins[g])}'bz;")
                else:
                    lines += [f"    assign {g}[{i}] = 1'bz;" for i in free]
        lines += [f"    {self.t_name} u_design(", "        " + ",\n        ".join(conns) + ");"]
        if jtag:
            lines += _jtag_uart(jtag)
        if signaltap:
            lines += self._signaltap(signaltap, ins)[0]
        lines += ["endmodule", ""]
        return "\n".join(lines)

    def _signaltap(self, spec: Any, ins: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
        """SignalTap in the board's top: (its Verilog, the map of its bits). spec:
        True, or {'signals': [port names], 'depth': samples}; by default every
        input on a pin (as the design gets it, after its two registers) and
        every output (with probes=True, the process's state too)."""
        spec = {} if spec is True else dict(spec)
        sources = {p.name: (_JTAG[ins[p.name][0]][2] if ins[p.name][0] in _JTAG
                            else f"{p.name}_s2", p.width) for p in self.ir.inputs
                   if p.kind == "input" and p.name in ins}
        sources.update({p.name: (f"{p.name}_w", p.width) for p in self.ir.outputs})
        names = list(spec.get("signals") or sources)
        unknown = [n for n in names if n not in sources]
        if unknown:
            raise ValueError(f"SignalTap: no port {unknown} to record (the inputs on pins and "
                             f"the outputs: {', '.join(sources)})")
        bits = sum(sources[n][1] for n in names)
        if bits > 4096:
            raise ValueError(f"SignalTap: {bits} bits; at most 4096 ('signals': fewer)")
        # (a depth of the buffer the device's memory holds easily, 256 Kbit at most of
        # its 594: 4,096 samples of 61 bits, 82 us at 50 MHz; 16,384 of 13 bits)
        depth = int(spec.get("depth")
                    or max(128, min(32768, 1 << ((262144 // bits).bit_length() - 1))))
        if depth & (depth - 1) or not 64 <= depth <= 131072:
            raise ValueError(f"SignalTap: a depth of {depth}: a power of two, 64 to 131072")
        crc = int(hashlib.md5(" ".join(f"{n}:{sources[n][1]}" for n in names).encode())
                  .hexdigest()[:8], 16)
        # (bit 0 the first signal's bit 0)
        tap = "{" + ", ".join(sources[n][0] for n in reversed(names)) + "}"
        mapping, k = [], 0
        for n in names:
            w = sources[n][1]
            mapping.append(f"{n}\t{k if w == 1 else f'{k + w - 1}:{k}'}\t{w}")
            k += w
        params = {"sld_data_bits": bits, "sld_trigger_bits": bits, "sld_sample_depth": depth,
                  "sld_ram_block_type": '"AUTO"', "sld_storage_qualifier_mode": '"OFF"',
                  "sld_trigger_level": 1, "sld_trigger_in_enabled": 0,
                  "sld_enable_advanced_trigger": 0, "sld_trigger_level_pipeline": 1,
                  "sld_trigger_pipeline": 0, "sld_ram_pipeline": 0, "sld_counter_pipeline": 0,
                  "sld_node_info": 806383104, "sld_incremental_routing": 0,
                  "sld_node_crc_bits": 32, "sld_node_crc_hiword": crc >> 16,
                  "sld_node_crc_loword": crc & 0xFFFF}
        lines = [f"    // SignalTap: {', '.join(names)} ({bits} bits), {depth} samples a clock "
                 f"apart; its",
                 "    // trigger set and run in Quartus's SignalTap (File > Create/Update > "
                 "Create SignalTap",
                 "    // File from Design Instance(s)); signaltap.txt: which bits are which",
                 "    sld_signaltap #(",
                 ",\n".join(f"        .{k}({v})" for k, v in params.items()),
                 "    ) u_signaltap (",
                 f"        .acq_clk(clk),",
                 f"        .acq_data_in({tap}),",
                 f"        .acq_trigger_in({tap}));"]
        return lines, ["signal\tbits\twidth"] + mapping

    def gen_pins_project(self, signaltap: Any = None) -> Path:
        """The Quartus project of the design on the board's pins (PINS()): the
        board's assignments, its clock's constraint, DE0_Nano and the design
        (with signaltap, SignalTap and signaltap.txt, its bits)."""
        project_dir = self._pins_dir
        if project_dir.is_dir():
            shutil.rmtree(project_dir)
        project_dir.mkdir(parents=True)
        name = "DE0_Nano"
        (project_dir / f"{name}.qpf").write_text(f"PROJECT_REVISION = {name}\n", encoding="utf-8")
        qsf = (DEFAULT_TEMPLATE_DIR / "hw" / "de0_nano_qsf.txt").read_text(encoding="utf-8")
        qsf += ("\n# The design (py2ver PINS())\n"
                "set_global_assignment -name VERILOG_FILE DE0_Nano.v\n"
                f"set_global_assignment -name VERILOG_FILE {self.t_name}.v\n")
        (project_dir / f"{name}.qsf").write_text(qsf, encoding="utf-8")
        shutil.copy(DEFAULT_TEMPLATE_DIR / "hw" / "de0_nano_sdc.txt", project_dir / "DE0_Nano.SDC")
        (project_dir / "DE0_Nano.v").write_text(self.board_top(signaltap), encoding="utf-8")
        if signaltap:
            (project_dir / "signaltap.txt").write_text(
                "\n".join(self._signaltap(signaltap, self._pin_map()[0])[1]) + "\n",
                encoding="utf-8")
        self._copy_hdl(project_dir, f"{self.t_name}.v")
        log.info("The design on the board's pins: %s", project_dir)
        return project_dir

    def PROGRAM(self, syn_attr: Optional[Dict[str, Any]] = None, cable: str = "1") -> None:
        """Load the PINS() build (syn/<name>_pins/DE0_Nano.sof) into the board's
        FPGA over JTAG (its memory: a power cycle forgets it; the board's flash is
        left as it is). cable: as jtagconfig names it, or its number."""
        from .jtag_uart import find_tool
        sof = self._pins_dir / "DE0_Nano.sof"
        if not sof.exists():
            raise FileNotFoundError(f"{sof}: run PINS() first")
        pgm = find_tool("quartus_pgm", (syn_attr or {}).get("QUARTUS_DIR"))
        result = subprocess.run([pgm, "-c", cable, "-m", "JTAG", "-o", f"p;{sof}"],
                                capture_output=True, text=True)
        if result.returncode != 0 or "Configuration succeeded" not in result.stdout:
            errors = [l for l in result.stdout.splitlines() if "Error" in l]
            raise RuntimeError(f"quartus_pgm failed: " + "\n".join(errors[-5:] or
                                                                  result.stdout.splitlines()[-5:]))
        log.info("Programmed %s", sof)

    def jtag_uart(self, syn_attr: Optional[Dict[str, Any]] = None, cable: Optional[str] = None):
        """A connection to the board's JTAG UART (jtag_uart.JtagUart): write()
        bytes to the design running there, read() what it sends."""
        from .jtag_uart import JtagUart
        return JtagUart((syn_attr or {}).get("QUARTUS_DIR"), cable)

    def jtag_reference(self, data: bytes, settle: int = 100_000) -> bytes:
        """The bytes the design sends the PC for data sent to it over the JTAG
        UART, by its generator: a testbench playing the board's JTAG UART (a
        byte given while the design is ready, each byte sent taken, room for it
        always), ending settle clocks after the last byte sent back. What the
        board must answer: link.read(len(it))."""
        from . import process
        if self.process is None:
            raise ValueError("jtag_reference(): a process's (a generator's)")
        by = {pin: self._py_name.get(port, port) for port, pin in self._pins.items()
              if pin in _JTAG}
        missing = [p for p in ("JTAG_RX_DATA", "JTAG_RX_VALID", "JTAG_TX_DATA", "JTAG_TX_VALID")
                   if p not in by]
        if missing:
            raise ValueError(f"jtag_reference(): no port on {', '.join(missing)}")

        def tb(dut):
            got, k, quiet = [], 0, 0
            while quiet < settle:
                if "JTAG_TX_READY" in by:
                    setattr(dut, by["JTAG_TX_READY"], 1)
                give = k < len(data) and (getattr(dut, by["JTAG_RX_READY"])
                                          if "JTAG_RX_READY" in by else 1)
                setattr(dut, by["JTAG_RX_VALID"], int(bool(give)))
                if give:
                    setattr(dut, by["JTAG_RX_DATA"], data[k])
                    k += 1
                if getattr(dut, by["JTAG_TX_VALID"]):
                    got.append(getattr(dut, by["JTAG_TX_DATA"]))
                    quiet = 0
                elif k == len(data):
                    quiet += 1
                yield
            return bytes(got)
        return process.drive(self.process.func, tb, self._process_attr,
                             limit=settle + 100 * len(data) + 1_000_000).value

    def PINS(self, syn_attr: Optional[Dict[str, Any]] = None, signaltap: Any = None) -> SynReport:
        """
        Compile the design on the DE0-Nano's own pins (each port's attr 'pin')
        with Quartus, its area and timing; nothing is programmed. syn_attr: as
        SYN()'s ('QUARTUS_DIR'). The project: syn/<name>_pins. signaltap: True
        (or {'signals': [...], 'depth': n}) for SignalTap recording the design's
        pins (with probes=True, its state too) on the board.
        """
        syn_attr = dict(syn_attr or {})
        quartus_sh = self._quartus_sh(syn_attr)
        project_dir = self.gen_pins_project(signaltap)
        syn_attr.setdefault("CLK_PERIOD_NS", DEFAULT_SYN_PERIOD_NS)
        # (the board's assignments keep the reports beside the project)
        return self._run_quartus(quartus_sh, project_dir, "DE0_Nano", self.t_name, syn_attr,
                                 out_dir=project_dir)
