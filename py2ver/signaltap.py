"""
SignalTap captures from the board, scripted (CAPTURE(), see pins.py): the board's
project built with PINS(signaltap=True) and programmed, a .stp made from its
SignalTap instance (quartus_stp --create_signaltap_hdl_file), a trigger set, a run,
and the record read back with each port by its name.

What Quartus wants, as found on a DE0-Nano with Quartus 22.1std:
- run and export_data_log need a named signal set and trigger: the generated .stp
  names neither, so they are named here;
- the export holds only what the .stp's data view lists: every bit is put there;
- a run stores its record in the .stp, so each capture starts from a fresh one;
- a trigger condition is a basic level, bits tested one by one ('bit' == high,
  low, rising edge, falling edge) and joined by && (no parentheses; & as &amp; in
  the XML); one it can't read is only a warning, and the trigger then fires at once.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

INSTANCE = "signaltap_megafunction_1"           # (the name quartus_stp gives the HDL instance)
_EDGES = {"rising": "rising edge", "falling": "falling edge"}


@dataclass
class Capture:
    """A SignalTap record: each sample's values by port name, the trigger's sample."""
    rows: List[Dict[str, int]]
    trigger: int
    widths: Dict[str, int] = field(default_factory=dict)
    # (a process's state probe: each state's line in the source, explain()'s)
    states: Dict[str, List[str]] = field(default_factory=dict)

    def show(self, limit: int = 40) -> str:
        """The record's changes as lines: the sample (from the trigger), what
        changed, and a state probe's state by its line in the generator."""
        out, prev = [], {}
        for k, r in self.changes()[:limit]:
            moved = {n: v for n, v in r.items() if prev.get(n) != v}
            text = ", ".join(f"{n}={v}" for n, v in moved.items())
            for n, lines in self.states.items():
                if n in moved and moved[n] < len(lines):
                    text += f"   [{n}: {lines[moved[n]]}]"
            out.append(f"{k:>7}  {text}")
            prev = r
        return "\n".join(out)

    def changes(self) -> List[Tuple[int, Dict[str, int]]]:
        """(sample, values) where any value changes (the first sample too); a sample
        counted from the trigger's (negative: before it)."""
        out, prev = [], None
        for k, r in enumerate(self.rows):
            if r != prev:
                out.append((k - self.trigger, r))
            prev = r
        return out

    def write_vcd(self, path: str, period_ns: int = 20) -> None:
        """The record as waveforms (GTKWave, Surfer): the trigger at time 0."""
        from .process import _vcd_id
        names = list(self.widths)
        ids = {n: _vcd_id(k + 1) for k, n in enumerate(names)}
        out = ["$timescale 1ns $end", "$scope module capture $end",
               f"$var wire 1 {_vcd_id(0)} clk $end"]
        out += [f"$var wire {self.widths[n]} {ids[n]} {n} $end" for n in names]
        out += ["$upscope $end", "$enddefinitions $end"]
        last: Dict[str, int] = {}
        clk = _vcd_id(0)
        for k, r in enumerate(self.rows):             # (sample k at k clocks; the trigger's
            out += [f"#{k * period_ns}", f"1{clk}"]    # marked below)
            if k == self.trigger:
                out.append("$comment trigger $end")
            for n in names:
                if last.get(n) != r[n]:
                    last[n] = r[n]
                    w = self.widths[n]
                    out.append(f"{r[n]}{ids[n]}" if w == 1 else f"b{r[n]:b} {ids[n]}")
            out += [f"#{k * period_ns + period_ns // 2}", f"0{clk}"]
        Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


def read_map(path: Path) -> List[Tuple[str, int, int]]:
    """signaltap.txt: (signal, its first bit, its width) each."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        name, bits, width = line.split("\t")
        out.append((name, int(bits.split(":")[-1]), int(width)))
    return out


def condition(trigger: Optional[Dict[str, Any]], fields: List[Tuple[str, int, int]]) -> str:
    """A basic trigger condition: each signal's value (an int: every bit of it), or
    'rising' / 'falling' (a bit's edge); all of them at once."""
    if not trigger:
        return ""
    where = {n: (lo, w) for n, lo, w in fields}
    parts = []
    for name, want in trigger.items():
        if name not in where:
            raise ValueError(f"SignalTap: no signal '{name}' recorded (recorded: "
                             f"{', '.join(where)})")
        lo, w = where[name]
        if want in _EDGES:
            if w != 1:
                raise ValueError(f"SignalTap: '{name}' {want}: an edge is a bit's ({w} bits)")
            parts.append(f"'acq_trigger_in[{lo}]' == {_EDGES[want]}")
            continue
        want = int(want)
        if not 0 <= want < (1 << w):
            raise ValueError(f"SignalTap: '{name}' == {want}: {w} bits")
        parts += [f"'acq_trigger_in[{lo + i}]' == {'high' if (want >> i) & 1 else 'low'}"
                  for i in range(w)]
    return " &amp;&amp; ".join(parts)


def make_stp(project: Path, stp_tool: str, name: str, cond: str) -> Path:
    """A fresh .stp for the project's SignalTap instance: named, every bit in its
    views, cond its trigger (none: at once)."""
    stp = project / f"{name}.stp"
    stp.unlink(missing_ok=True)
    r = subprocess.run([stp_tool, "DE0_Nano", "--create_signaltap_hdl_file", f"--stp_file={stp.name}"],
                       cwd=project, capture_output=True, text=True)
    if r.returncode != 0 or not stp.exists():
        raise RuntimeError(f"quartus_stp couldn't make {stp}: " + "\n".join(
            l for l in r.stdout.splitlines() if "rror" in l)[-800:])
    t = stp.read_text(encoding="utf-8")
    n = len(re.findall(r'name="acq_data_in\[\d+\]"', t))
    nets = "\n".join(
        f'          <net data_index="{i}" is_data_input="true" is_node_valid="true" '
        f'is_signal_inverted="no" is_storage_input="false" is_trigger_input="true" '
        f'name="acq_data_in[{i}]" storage_index="-1" tap_mode="classic" trigger_index="{i}" '
        f'type="unknown"/>' for i in range(n))
    t = t.replace('<signal_set name="">', '<signal_set name="signal_set_1">', 1)
    t = t.replace(' name="" record_data_gap', ' name="trigger_1" record_data_gap', 1)
    t = t.replace("<data_view/>", f"<data_view>\n{nets}\n        </data_view>", 1)
    t = t.replace("<setup_view/>", f"<setup_view>\n{nets}\n        </setup_view>", 1)
    if cond:
        level = (f'<level enabled="yes" name="condition1" type="basic">{cond}\n'
                 '            <power_up enabled="yes">\n            </power_up>\n'
                 '            <op_node/>\n          </level>')
        t, k = re.subn(r'<level type="basic">\s*<power_up>\s*</power_up>\s*</level>', level, t,
                       count=1)
        if not k:
            raise RuntimeError(f"{stp}: no trigger level to set (quartus_stp's .stp changed?)")
    stp.write_text(t, encoding="utf-8")
    return stp


def capture(project: Path, stp_tool: str, trigger: Optional[Dict[str, Any]] = None,
            during: Optional[Callable[[], Any]] = None, timeout: float = 30.0,
            arm_s: float = 4.0, names: Optional[Dict[str, str]] = None) -> Capture:
    """Capture on the board running the project's build: armed, during() called
    arm_s seconds later (what makes the trigger happen: bytes sent, ...), the
    record when it has triggered (or TimeoutError after timeout seconds).
    names: a signal's name in the record (byte_ -> byte)."""
    fields = read_map(project / "signaltap.txt")
    stp = make_stp(project, stp_tool, "capture", condition(trigger, fields))
    log = f"log_{int(time.time() * 1000)}"
    for f in ("capture.csv", "capture.vcd"):
        (project / f).unlink(missing_ok=True)
    tcl = project / "capture.tcl"
    tcl.write_text(
        f"open_session -name {stp.name}\n"
        "set hw [lindex [get_hardware_names] 0]\n"
        "set dev [lindex [get_device_names -hardware_name $hw] 0]\n"
        f"run -instance {INSTANCE} -signal_set signal_set_1 -trigger trigger_1 -data_log {log} "
        f"-timeout {int(timeout)} -hardware_name $hw -device_name $dev\n"
        f"export_data_log -instance {INSTANCE} -signal_set signal_set_1 -trigger trigger_1 "
        f"-data_log {log} -filename capture.csv -format csv\n"
        f"export_data_log -instance {INSTANCE} -signal_set signal_set_1 -trigger trigger_1 "
        f"-data_log {log} -filename capture.vcd -format vcd\n"
        "close_session\n", encoding="utf-8")
    proc = subprocess.Popen([stp_tool, "-t", tcl.name], cwd=project, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    out: List[str] = []
    reader = threading.Thread(target=lambda: out.extend(proc.stdout), daemon=True)
    reader.start()
    if during is not None:
        time.sleep(arm_s)                       # (armed: quartus_stp takes a second or two)
        during()
    proc.wait(timeout=timeout + 60)
    reader.join(timeout=5)
    text = "".join(out)
    if "TIMEOUT state" in text:
        raise TimeoutError(f"SignalTap: the trigger {trigger} didn't happen in {timeout:g} s")
    if proc.returncode != 0 or not (project / "capture.csv").exists():
        raise RuntimeError("SignalTap capture failed: " + "\n".join(
            l for l in text.splitlines() if "rror" in l)[-1000:])
    return read_capture(project, fields, names or {})


def read_capture(project: Path, fields: List[Tuple[str, int, int]],
                 names: Dict[str, str]) -> Capture:
    """The exported record (capture.csv; capture.vcd for where the trigger is)."""
    lines = (project / "capture.csv").read_text(encoding="utf-8").splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith("time unit"))
    rows = []
    for line in lines[head + 1:]:
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 2 or "X" in cells[1:-1]:
            continue                             # (the first: not yet sampled)
        bits = cells[1:]
        rows.append({names.get(n, n): int("".join(reversed(bits[lo:lo + w])), 2)
                     for n, lo, w in fields})
    start = re.search(r"Sample (-?\d+) \(Start", (project / "capture.vcd").read_text(
        encoding="utf-8", errors="replace"))
    first = int(start.group(1)) if start else 0
    # (the trigger: where the VCD says the record starts before it, -512; its export
    # doesn't always say, and SignalTap's default puts it an eighth of the way in)
    return Capture(rows, -first if first < 0 else len(rows) // 8,
                   {names.get(n, n): w for n, _, w in fields})
