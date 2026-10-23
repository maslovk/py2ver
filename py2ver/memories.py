"""Py2ver's memories as Verilog (a mixin of Py2ver, see core.py)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import vprint
from .ast_ir import Assignment, ModuleIR

ROM_FILE_MIN = 4096         # a table longer than this is read from a file ($readmemh)


class MemoriesMixin:
    """Py2ver's memories as Verilog: tables read asynchronously, tables and register files in block RAM, block-RAM memories and the SDRAM's controller."""

    # The SDRAM's pins (the DE0-Nano's names, lower case), as a module with sdram= has them
    SDRAM_PORTS = [("output", 13, "dram_addr"), ("output", 2, "dram_ba"), ("output", 1, "dram_cas_n"),
                   ("output", 1, "dram_cke"), ("output", 1, "dram_clk"), ("output", 1, "dram_cs_n"),
                   ("inout", 16, "dram_dq"), ("output", 2, "dram_dqm"), ("output", 1, "dram_ras_n"),
                   ("output", 1, "dram_we_n"), ("output", 1, "adv"),
                   # (the host's port: reads and writes while the design is paused)
                   ("input", 1, "host_req"), ("input", 1, "host_we"), ("input", 24, "host_addr"),
                   ("input", 16, "host_wdata"), ("output", 1, "host_ready"),
                   ("output", 1, "host_rvalid"), ("output", 16, "host_rdata")]

    @staticmethod
    def _sdram_items(sdram: Dict[str, Any]) -> List[str]:
        """The SDRAM's controller and the return path of its reads: the design's
        access goes to a register first (the controller's request: its address
        and its row's check from a register, not from the design's logic), the
        design's clock enable ce is ce_in, held low while that request waits (a
        row opened, a refresh) and until the SDRAM is set up; its reads (ext_req
        and not ext_we) get their data on ext_q `latency` advancing clocks later,
        in order, from a FIFO. adv: the design advanced this clock."""
        L = int(sdram["latency"])
        return [
            "// The SDRAM: the lists and tables that don't fit the block RAM (sdram=)",
            "wire __ext_ready, __ext_rvalid, __ext_init;",
            "wire [15:0] __ext_rdata;",
            "// (the design's access, a clock later: the controller's request)",
            "reg __p_req = 1'b0, __p_we = 1'b0;",
            "reg [23:0] __p_addr = 24'd0;",
            "reg [15:0] __p_wdata = 16'd0;",
            "wire __ext_stall = !__ext_init || (__p_req && !__ext_ready);",
            "wire ce = ce_in && !__ext_stall;",
            "assign adv = ce;",
            "assign dram_clk = ~clk;",
            "always @(posedge clk) begin",
            "  if (rst) __p_req <= 1'b0;",
            "  else if (ce) begin",
            "    __p_req <= ext_req; __p_we <= ext_we; __p_addr <= ext_addr; __p_wdata <= ext_wdata;",
            "  end else if (__p_req && __ext_ready) __p_req <= 1'b0;   // (done while paused)",
            "end",
            "// (the host reads and writes through it while the design is paused: ce_in low,",
            "// once the design's last access is done)",
            "wire __host = !ce_in && !__p_req;",
            "assign host_ready = __host && __ext_init && __ext_ready;",
            "wire __req = (__host ? host_req : __p_req) && __ext_init;",
            "wire __we = __host ? host_we : __p_we;",
            "py2ver_sdram __sdram(.clk(clk), .rst(rst), .req(__req),",
            "    .we(__we), .addr(__host ? host_addr : __p_addr),",
            "    .wdata(__host ? host_wdata : __p_wdata), .ready(__ext_ready),",
            "    .rvalid(__ext_rvalid), .rdata(__ext_rdata), .init_done(__ext_init),",
            "    .sa(dram_addr), .ba(dram_ba), .cs_n(dram_cs_n), .ras_n(dram_ras_n),",
            "    .cas_n(dram_cas_n), .we_n(dram_we_n), .cke(dram_cke), .dqm(dram_dqm), .dq(dram_dq));",
            f"// A read's data, in order, on ext_q {L} advancing clocks after its address",
            "// (a register: taken from the FIFO the clock before)",
            "// (a read's data for whoever issued it: the design's into the FIFO)",
            "reg [15:0] __ext_fifo [0:7];",
            "reg [2:0] __ext_wp, __ext_rp;",
            f"reg [{L - 1}:0] __ext_due;",
            "reg [15:0] ext_q;",
            "reg __tag [0:7];",
            "reg [2:0] __tag_wp, __tag_rp;",
            "wire __for_host = __tag[__tag_rp];",
            "assign host_rvalid = __ext_rvalid && __for_host;",
            "assign host_rdata = __ext_rdata;",
            "always @(posedge clk) begin",
            "  if (rst) begin",
            "    __ext_wp <= 3'd0; __ext_rp <= 3'd0; __ext_due <= 0;",
            "    __tag_wp <= 3'd0; __tag_rp <= 3'd0;",
            "  end else begin",
            "    if (__req && __ext_ready && !__we) begin",
            "      __tag[__tag_wp] <= __host;",
            "      __tag_wp <= __tag_wp + 3'd1;",
            "    end",
            "    if (__ext_rvalid) __tag_rp <= __tag_rp + 3'd1;",
            "    if (__ext_rvalid && !__for_host) begin",
            "      __ext_fifo[__ext_wp] <= __ext_rdata;",
            "      __ext_wp <= __ext_wp + 3'd1;",
            "    end",
            "    if (ce) begin",
            f"      __ext_due <= {{__ext_due[{L - 2}:0], ext_req && !ext_we}};",
            f"      if (__ext_due[{L - 2}]) begin",
            "        ext_q <= __ext_fifo[__ext_rp];",
            "        __ext_rp <= __ext_rp + 3'd1;",
            "      end",
            "    end",
            "  end",
            "end",
        ]

    def _schedule_rams(self, ir: ModuleIR) -> Tuple[int, Dict[str, Any]]:
        """
        Clocks per sample with block RAMs, and the clock of each read/write.
        A read is issued once the reads its address depends on have their data;
        the outputs are computed when all read data is in; then the writes
        follow one per clock, in program order (so the last write to an
        address wins, as in Python).
        """
        if not self.rams:
            return 1, {}
        deps = {a.left: set(self._extract_idents(a.right)) for a in ir.assigns}
        rdata_of = {rd: (mem, j) for mem, ports in self._ram_ports.items()
                    for j, (_ra, rd) in enumerate(ports["reads"])}

        def reach(name: str, seen: set) -> set:
            for d in deps.get(name, ()):
                if d not in seen:
                    seen.add(d)
                    reach(d, seen)
            return seen

        issue: Dict[str, int] = {}

        def issue_clock(rdata: str) -> int:
            if rdata not in issue:
                mem, j = rdata_of[rdata]
                raddr = self._ram_ports[mem]["reads"][j][0]
                before = [r for r in reach(raddr, {raddr}) if r in rdata_of]
                issue[rdata] = max((issue_clock(r) + 1 for r in before), default=0)
            return issue[rdata]

        for rd in rdata_of:
            issue_clock(rd)
        compute = max((c + 1 for c in issue.values()), default=0)
        writes = max((len(p["writes"]) for p in self._ram_ports.values()), default=0)
        return compute + max(writes, 1), {"issue": issue, "compute": compute}

    @staticmethod
    def _rom_registers(ir: ModuleIR, rom_reads: Dict[str, Dict[str, Any]],
                       attr: Dict[str, Dict[str, Any]]) -> ModuleIR:
        """Tables whose value only goes into a register, read by a memory block.

        A memory block gives its value a clock after the address, so a table
        read used in the same clock stays logic (see _rom_arrays). But when the
        value only goes into a register r, as in `x = PATH[k]` followed by a
        yield, the memory block's output register can hold it instead:

            r__q       the memory block's output, loaded when r takes the table's value
            r__table   whether r's value is r__q (set when r takes the table's value,
                       cleared when it takes any other)
            r__held    r's register, for the other values
            r          a wire: r__table ? r__q : r__held

        This applies when the table's value is r's next value, or one of the
        choices of it, and nothing else reads the value or that next value."""
        regs = {}       # (a register -> its update's reset value, next value and hold's cast)
        for a in ir.assigns:
            m = re.fullmatch(r"rst \? (.+) : \(ce \? (\w+) : (\$(?:un)?signed)\((\w+)\)\)", a.right)
            if a.is_reg and m and m.group(4) == a.left:
                regs[a.left] = (m.group(1), m.group(2), m.group(3))
        tables = [t for t, rom in rom_reads.items() if "entries" in rom]
        if not regs or not tables:
            return ir
        by_left = {a.left: a for a in ir.assigns}
        readers: Dict[str, List[str]] = {}       # (a name -> the assignments that read it)
        for a in ir.assigns:
            for n in set(re.findall(r"(?<![$\w'])[A-Za-z_]\w*", a.right)):
                readers.setdefault(n, []).append(a.left)
        outputs = {p.name for p in ir.outputs}

        def copied(name: str) -> str | None:
            """The name that `name = other` copies, if name is such a copy."""
            a = by_left.get(name)
            if a is None or a.is_reg or not re.fullmatch(r"[A-Za-z_]\w*", a.right):
                return None
            return a.right

        def copies(name: str) -> set:
            """name and the names that copy it (or copy a copy)."""
            out, todo = {name}, [name]
            while todo:
                for r in readers.get(todo.pop(), []):
                    if r not in out and copied(r) in out:
                        out.add(r)
                        todo.append(r)
            return out

        def choices(node: Any, change: Any) -> Any:
            """The selection with each of its choices (not its conditions) changed."""
            if node[0] == "tern":
                return ("tern", node[1], choices(node[2], change), choices(node[3], change))
            return change(node)

        def leaves(node: Any) -> List[Any]:
            found: List[Any] = []
            choices(node, lambda n: found.append(n) or n)
            return found

        removed: set = set()
        new_assigns: List[Assignment] = []
        items: List[str] = []
        for r, (init, nxt, cast) in regs.items():
            if r in outputs or r not in attr:
                continue
            # Back from the next value through its copies, to the table's value or
            # to the selection that makes it
            chain = [nxt]
            while copied(chain[-1]) is not None and chain[-1] not in rom_reads:
                chain.append(copied(chain[-1]))
            rom = next((t for t in tables if chain[-1] in copies(t)), None)
            select = None
            if rom is not None:                   # (the next value is the table's value)
                reads = [rom]
                chain = chain[:min(k for k, c in enumerate(chain) if c in copies(rom))]
                tree = None
            else:
                a = by_left.get(chain[-1])
                tree = vprint.tree_or_none(a.right) if a is not None and not a.is_reg else None
                if tree is None:
                    continue
                # (the choices may read one table at several places: one memory block)
                reads = [t for t in tables if any(n[0] == "atom" and n[1] in copies(t)
                                                  for n in leaves(tree))]
                same = {(tuple(rom_reads[t]["entries"]), rom_reads[t]["width"],
                         bool(rom_reads[t].get("signed")), rom_reads[t]["address_bits"])
                        for t in reads}
                if len(same) != 1:
                    continue
                select = chain[-1]
            read_of = {c: t for t in reads for c in copies(t)}    # (a copy -> its read)
            table = set(read_of)
            if select is not None:
                # (the table's value only a choice, not in a condition or a sum)
                named = re.findall(r"(?<![$\w'])[A-Za-z_]\w*", by_left[select].right)
                if sum(n in table for n in named) != sum(n[0] == "atom" and n[1] in table
                                                         for n in leaves(tree)):
                    continue
            # Nothing else reads the table's value, or the next value on its way to r
            user = select or (chain[-1] if chain else r)
            if {x for t in table for x in readers.get(t, []) if x not in table} != {user}:
                continue
            if any(readers.get(c) != [b] for c, b in zip(chain, [r] + chain)):
                continue
            info = rom_reads[reads[0]]
            w, ab = int(info["width"]), int(info["address_bits"])
            # (each read's address: its index, as _rom_arrays finds it)
            address = {}
            for t in reads:
                index = set(re.findall(r"(?<![$\w'])[A-Za-z_]\w*", by_left[t].right))
                if len(index) != 1:
                    break
                idx = index.pop()
                iw = int(attr.get(idx, attr.get(rom_reads[t]["index"], {})).get("width", ab + 1))
                address[t] = f"{idx}[{ab - 1}:0]" if iw > ab else idx
            if len(address) != len(reads):
                continue

            def is_table(n: Any) -> bool:
                return n[0] == "atom" and n[1] in table

            held, flag, q = f"{r}__held", f"{r}__table", f"{r}__q"
            if select is None:
                load = keep = "1'b1"
                addr = address[rom]
                held_next = held                # (r__held is unused: it holds)
                removed |= set(chain)
            else:
                own = copies(r)
                one, zero = ("atom", "1'b1"), ("atom", "1'b0")
                load = vprint.text(choices(tree, lambda n: one if is_table(n) else zero))
                keep = vprint.text(choices(tree, lambda n: one if is_table(n) else (
                    ("atom", flag) if n[0] == "atom" and n[1] in own else zero)))
                # (the address of the read chosen; with no read chosen, any)
                first = ("atom", address[reads[0]])
                addr = vprint.text(choices(tree, lambda n: ("atom", address[read_of[n[1]]])
                                           if is_table(n) else first)) if len(reads) > 1 \
                    else address[reads[0]]
                # (r__held takes the other choices; when r takes the table's, any value)
                new_assigns.append(Assignment(
                    select, vprint.text(choices(tree, lambda n: ("atom", r) if is_table(n) else n)),
                    False, by_left[select].line))
                removed.add(select)
                held_next = nxt
            rw = int(attr[r].get("width", 1))
            if rw > w:                      # (r's value: the entry extended by its sign)
                fill = f"{{{rw - w}{{{q}[{w - 1}]}}}}" if info.get("signed") else f"{rw - w}'d0"
                value = f"{{{fill}, {q}}}"
            else:
                value = f"{q}[{rw - 1}:0]" if rw < w else q
            entries = list(info["entries"])
            entries += [entries[-1]] * ((1 << ab) - len(entries))
            mask = (1 << w) - 1
            arr = f"{r}__rom"
            items += [f"// Table in a memory block: {len(info['entries'])} x {w} bits, "
                      f"its entry for {r} a clock after the address",
                      f"reg [{w - 1}:0] {arr} [0:{len(entries) - 1}];",
                      "initial begin"]
            items += [f"  {arr}[{k}] = {w}'d{int(v) & mask};" for k, v in enumerate(entries)]
            items += ["end",
                      f"reg [{w - 1}:0] {q};",
                      f"wire [{ab - 1}:0] {q}__address = {addr};",
                      f"always @(posedge clk) if (ce && {flag}__load) {q} <= {arr}[{q}__address];"]
            attr[held] = dict(attr[r])
            for n in (flag, f"{flag}__load", f"{flag}__next"):
                attr[n] = {"width": 1, "signed": 0}
            new_assigns += [
                Assignment(f"{flag}__load", load, False),
                Assignment(f"{flag}__next", keep, False),
                Assignment(held, f"rst ? {init} : (ce ? {held_next} : {cast}({held}))", True),
                Assignment(flag, f"rst ? 1'b0 : (ce ? {flag}__next : $unsigned({flag}))", True),
                Assignment(r, f"{flag} ? {value} : {held}", False)]
            removed |= {r} | table
        if not removed:
            return ir
        return ModuleIR(name=ir.name, inputs=ir.inputs, outputs=ir.outputs,
                        assigns=[a for a in ir.assigns if a.left not in removed] + new_assigns,
                        has_clk=ir.has_clk, items=list(ir.items) + items)

    @staticmethod
    def _rom_arrays(ir: ModuleIR, rom_reads: Dict[str, Dict[str, Any]],
                    attr: Dict[str, Dict[str, Any]]) -> ModuleIR:
        """Table reads still in the assignments (not block RAM) as arrays read by
        their index: compact Verilog, and readable by Verilator (an expression of
        1024 cases is not). The array is padded to a power of two; reads of the
        same table share one array."""
        reads = {a.left: a for a in ir.assigns if a.left in rom_reads}
        if not reads:
            return ir
        items: List[str] = []
        arrays: Dict[Any, Tuple[str, int]] = {}  # table -> (array, its comment's place in items)
        for name, a in reads.items():
            rom = rom_reads[name]
            # (the index as the assignment reads it: renamed by pipelining, or a
            # sign-adjusted copy, but the one signal on the right)
            names = set(re.findall(r"(?<![$\w'])[A-Za-z_]\w*", a.right))
            if len(names) != 1:
                raise RuntimeError(f"Can't find the index of table read {name}: {a.right}")
            idx = names.pop()
            w, ab = int(rom["width"]), int(rom["address_bits"])
            sig = "signed " if rom.get("signed") else ""
            mask = (1 << w) - 1
            arr = f"{name}__table"
            iw = int(attr.get(idx, attr.get(rom["index"], {})).get("width", ab + 1))
            addr = f"{idx}[{ab - 1}:0]" if iw > ab else idx
            key = (tuple(rom.get("signals", rom.get("entries", ()))), "signals" in rom, w, sig, ab)
            if key in arrays:
                arr, at = arrays[key]
                items[at] += f", {name}"
                items.append(f"wire {sig}[{w - 1}:0] {name} = {arr}[{addr}];")
                continue
            arrays[key] = (arr, len(items))
            if "signals" in rom:                  # (a long list argument's ports)
                sigs = list(rom["signals"])
                items += [f"// A list argument read by index: {len(sigs)} x {w} bits, "
                          f"read by {name}",
                          f"wire {sig}[{w - 1}:0] {arr} [0:{(1 << ab) - 1}];"]
                items += [f"assign {arr}[{k}] = {v};" for k, v in enumerate(sigs)]
                items += [f"assign {arr}[{k}] = {w}'d0;" for k in range(len(sigs), 1 << ab)]
                items.append(f"wire {sig}[{w - 1}:0] {name} = {arr}[{addr}];")
                continue
            entries = list(rom["entries"])
            entries += [entries[-1]] * ((1 << ab) - len(entries))
            items += [f"// Table: {len(rom['entries'])} x {w} bits, read by {name}",
                      f"reg {sig}[{w - 1}:0] {arr} [0:{len(entries) - 1}];",
                      "initial begin"]
            items += [f"  {arr}[{k}] = {w}'d{int(v) & mask};" for k, v in enumerate(entries)]
            items += ["end", f"wire {sig}[{w - 1}:0] {name} = {arr}[{addr}];"]
        return ModuleIR(name=ir.name, inputs=ir.inputs, outputs=ir.outputs,
                        assigns=[a for a in ir.assigns if a.left not in reads],
                        has_clk=ir.has_clk, items=list(ir.items) + items)

    @staticmethod
    def _rom_items(roms: Dict[str, Dict[str, Any]], attr: Dict[str, Dict[str, Any]],
                   hdl_dir: Path, files: Dict[str, str]) -> List[str]:
        """Function tables in block RAM: q is the entry the index addressed a clock
        before. Two ports reading the same table share one array (a dual-port ROM:
        the same block RAM). A large table's contents go in files (name -> text),
        read from hdl_dir."""
        items: List[str] = []
        groups: Dict[Any, List[str]] = {}
        for q, rom in roms.items():
            key = (tuple(rom["entries"]), int(rom["width"]), bool(rom["signed"]))
            groups.setdefault(key, []).append(q)
        for qs in groups.values():
            for pair in (qs[i:i + 2] for i in range(0, len(qs), 2)):
                rom = roms[pair[0]]
                entries, w, aw = rom["entries"], int(rom["width"]), int(rom["address_bits"])
                sig = "signed " if rom["signed"] else ""
                mask = (1 << w) - 1
                arr = f"{pair[0]}__rom"
                ports = " and ".join(pair)
                items.append(f"// Table in block RAM: {len(entries)} x {w} bits"
                             + (f", read by {ports}" if len(pair) > 1 else ""))
                items.append(f"reg {sig}[{w - 1}:0] {arr} [0:{len(entries) - 1}];")
                items += [f"reg {sig}[{w - 1}:0] {q};" for q in pair]
                if len(entries) > ROM_FILE_MIN:
                    # (a large table from a file next to main.v: simulators and Quartus
                    # read it, instead of an assignment an entry to compile)
                    path = hdl_dir / f"{arr}.hex"
                    digits = (w + 3) // 4
                    files[path.name] = "".join(f"{int(v) & mask:0{digits}x}\n" for v in entries)
                    items.append(f'initial $readmemh("{path.as_posix()}", {arr});')
                else:
                    items.append("initial begin")
                    items += [f"  {arr}[{k}] = {w}'d{int(v) & mask};" for k, v in enumerate(entries)]
                    items.append("end")
                idxs = []
                for q in pair:
                    idx = roms[q]["index"]
                    if int(attr.get(idx, {}).get("width", aw + 1)) > aw:   # (never out of range)
                        idx = f"{idx}[{aw - 1}:0]"
                    idxs.append(idx)
                if len(pair) == 1:
                    items.append(f"always @(posedge clk) if (ce) {pair[0]} <= {arr}[{idxs[0]}];")
                    continue
                # Two ports: read every clock (with an enable Quartus won't make it one
                # dual-port ROM), at the address of the last enabled clock while ce is
                # low, so q holds as with an enable
                for q, idx in zip(pair, idxs):
                    items += [f"reg [{aw - 1}:0] {q}__a;",
                              f"always @(posedge clk) if (ce) {q}__a <= {idx};"]
                items.append("always @(posedge clk) begin "
                             + " ".join(f"{q} <= {arr}[ce ? {idx} : {q}__a];"
                                        for q, idx in zip(pair, idxs)) + " end")
        return items

    @staticmethod
    def _rf_items(rfs: Dict[str, Dict[str, Any]], attr: Dict[str, Dict[str, Any]]) -> List[str]:
        """A shared multiplier input's register file: a simple dual-port block RAM,
        written by we/waddr/wdata, q the entry raddr addressed a clock before."""
        items: List[str] = []
        for q, rf in rfs.items():
            w, depth, ab = int(rf["width"]), int(rf["depth"]), int(rf["address_bits"])
            sig = "signed " if rf["signed"] else ""
            arr = f"{q}__mem"

            def addr(name: str) -> str:
                return f"{name}[{ab - 1}:0]" if int(attr.get(name, {}).get("width", ab)) > ab \
                    else name
            items += [f"// Register file in block RAM: {depth} x {w} bits",
                      f"(* ramstyle = \"M9K\" *) reg {sig}[{w - 1}:0] {arr} [0:{depth - 1}];",
                      f"reg {sig}[{w - 1}:0] {q};"]
            if any(rf.get("init", ())):          # (a memory: its starting values)
                mask = (1 << w) - 1
                items += ["initial begin"] + [f"  {arr}[{k}] = {w}'d{int(v) & mask};"
                                              for k, v in enumerate(rf["init"])] + ["end"]
            else:
                items += [f"integer __i_{arr};",
                          f"initial for (__i_{arr} = 0; __i_{arr} < {depth}; __i_{arr} = __i_{arr} + 1)"
                          f" {arr}[__i_{arr}] = {w}'d0;"]
            items += ["always @(posedge clk) if (ce) begin",
                      f"  if ({rf['we']}) {arr}[{addr(rf['waddr'])}] <= {rf['wdata']};",
                      f"  {q} <= {arr}[{addr(rf['raddr'])}];",
                      "end"]
        return items

    def _ram_items(self) -> List[str]:
        """
        Verilog for the block-RAM memories. Each read port gets its own copy of
        the memory (all copies take every write), and the writes of a sample
        share one write port, one per clock, so every copy stays a simple
        dual-port block RAM.
        """
        if not self.rams:
            return []
        items: List[str] = []
        compute = self._ram_schedule["compute"]
        for mem, ram in self.rams.items():
            n, elem = ram["length"], ram["elem"]
            w = int(elem["width"])
            aw = max(1, (n - 1).bit_length())
            sig = "signed " if int(elem["signed"]) == 1 else ""
            reads = self._ram_ports[mem]["reads"]
            writes = self._ram_ports[mem]["writes"]
            init = self._ram_init[mem]
            values = list(init) if isinstance(init, (list, tuple)) else [init] * n
            if len(values) != n:
                raise ValueError(f"'init' of RAM '{mem}' needs {n} values")
            mask = (1 << w) - 1
            items.append(f"// RAM {mem}: {n} x {w} bits, {max(len(reads), 1)} copy(ies)")

            if writes:
                def by_clock(k: int) -> str:
                    expr = "0"
                    for i in reversed(range(len(writes))):
                        expr = f"({self._phase_is(compute + i)} ? {writes[i][k]} : {expr})"
                    return expr
                items.append(f"wire [{aw - 1}:0] {mem}__wa = {by_clock(1)};")
                items.append(f"wire [{w - 1}:0] {mem}__wd = {by_clock(2)};")
                enables = " || ".join(f"({self._phase_is(compute + i)} && {we})"
                                      for i, (we, _a, _d) in enumerate(writes))
                items.append(f"wire {mem}__wen = {enables};")

            for c in range(max(len(reads), 1)):
                arr = f"{mem}__m{c}"
                items.append(f"reg {sig}[{w - 1}:0] {arr} [0:{n - 1}];")
                if any(values):
                    items.append("initial begin")
                    items += [f"  {arr}[{k}] = {w}'d{int(v) & mask};" for k, v in enumerate(values)]
                    items.append("end")
                else:
                    items.append(f"integer __i_{arr};")
                    items.append(f"initial for (__i_{arr} = 0; __i_{arr} < {n}; "
                                 f"__i_{arr} = __i_{arr} + 1) {arr}[__i_{arr}] = {w}'d0;")
                items.append("always @(posedge clk) begin")
                if c < len(reads):
                    raddr, rdata = reads[c]
                    items.insert(-1, f"reg {sig}[{w - 1}:0] {rdata};")
                    clock = self._ram_schedule["issue"][rdata]
                    items.append(f"  if (ce && {self._phase_is(clock)}) {rdata} <= {arr}[{raddr}];")
                if writes:
                    items.append(f"  if (ce && {mem}__wen) {arr}[{mem}__wa] <= {mem}__wd;")
                items.append("end")
        return items
