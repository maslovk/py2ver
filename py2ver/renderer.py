# renderer.py
"""
The module as Verilog, laid out to be read:

  - an ANSI port list (clk, rst, ce and vin first, then the inputs and the
    outputs), its columns aligned
  - the registers and wires, aligned, in the order the logic assigns them
  - the registers' updates as `if (rst) ... else if (enable) ...`, a block per
    reset and enable shared, wherever that is provably the same as the
    selection the passes wrote (see _register_update)
  - expressions with only the parentheses they need, long ones over several
    lines (see vprint.py)
  - a list read at a variable index (a selection tree, `i < 2 ? (i < 1 ? a :
    b) : ...`) as a case table: a function for a table of numbers, written
    once however often it is read; a reg and its always block for a list of
    signals (see _lookup)
  - above the logic each Python statement became, the statement, with its
    line in the file
"""
import ast
import dataclasses
import os
import re
import textwrap
from typing import Any, Dict, List, Sequence, Tuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import vprint
from .ast_ir import Assignment, ModuleIR

IND = "  "
_CONTROL = ("clk", "rst", "ce", "vin")
LOOKUP_MIN = 4          # a list read at a variable index: a case table from this many elements
LOOKUP_MAX = 1024       # (and up to this many entries)


def make_env(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _range(width: int) -> str:
    return f"[{width - 1}:0]" if width > 1 else ""


def _statements(text: str) -> Dict[int, str]:
    """line -> the statement starting there, on one line (a compound
    statement: its header), from a function's source."""
    lines = textwrap.dedent(text).splitlines()
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return {}
    out = {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.stmt) or isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            continue
        body = getattr(n, "body", None)
        end = body[0].lineno - 1 if body and body[0].lineno > n.lineno else n.end_lineno
        stmt = " ".join(line.strip() for line in lines[n.lineno - 1:max(end, n.lineno)])
        out[n.lineno] = stmt if len(stmt) <= 96 else stmt[:93] + "..."
    return out


def _integer(node: Any) -> int | None:
    """The value of a 32-bit number n or -n (a selection tree's bound)."""
    neg = node[0] == "un" and node[1] == "-"
    if neg:
        node = node[2]
    if node[0] != "atom" or not node[1].isdigit() or int(node[1]) >= 1 << 31:
        return None
    return -int(node[1]) if neg else int(node[1])


def _columns(rows: Sequence[Sequence[str]]) -> List[str]:
    """rows of fields, each column padded to its widest (empty columns left out)."""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))] if rows else []
    out = []
    for r in rows:
        fields = [f.ljust(w) for f, w in zip(r[:-1], widths[:-1]) if w] + [r[-1]]
        out.append(" ".join(fields))
    return out


class Renderer:
    def __init__(self, template_dir: str):
        self.env = make_env(template_dir)
        self._cache: Dict[str, Any] = {}

    def _tpl(self, name: str):
        if name not in self._cache:
            self._cache[name] = self.env.get_template(name)
        return self._cache[name]

    def render_module(self, ir: ModuleIR, attr: Dict[str, Dict[str, Any]] | None = None,
                      extra_ports: Sequence[Tuple[str, int, str]] = (),
                      source: Tuple[str, str, int, str] | None = None) -> str:
        """
        The module. 'attr' should be the normalized attribute dict from
        FunctionVisitor.attr, so that internal SSA temporaries (e.g. acc_0,
        acc_1, ...) have width/signed info. extra_ports: (direction, width,
        name) of ports the IR doesn't hold (the SDRAM's pins). source: the
        Python function's (name, source text, first line, file), for comments.
        """
        if attr is None:
            attr = {}
        # name -> (width, signed) of everything declared here
        self._decl: Dict[str, Tuple[int, bool]] = {}

        # ----------------- Ports -----------------
        ports = []          # (direction, signed, range, name)
        control = sorted((p for p in ir.inputs if p.kind in _CONTROL),
                         key=lambda p: _CONTROL.index(p.kind))
        for p in control + [p for p in ir.inputs if p.kind not in _CONTROL]:
            name = p.name if p.kind != 'clk' else 'clk'
            ports.append(("input", "signed" if p.signed else "", _range(p.width), name))
            self._decl[name] = (p.width, bool(p.signed))
        for p in ir.outputs:
            kind = "output reg" if p.kind == 'reg' else "output"
            ports.append((kind, "signed" if p.signed else "", _range(p.width), p.name))
            self._decl[p.name] = (p.width, bool(p.signed))
        for d, w, n in extra_ports:
            ports.append((d, "", _range(w), n))
            self._decl[n] = (w, False)
        port_names = {p[3] for p in ports}
        portlist = ",\n".join(IND + line.rstrip() for line in _columns(ports))

        # ----------------- Internal wires/regs -----------------
        # Anything assigned but not a port, in the order it is assigned
        reg_lhs = {a.left for a in ir.assigns if a.is_reg}
        internal = [n for n in dict.fromkeys(a.left for a in ir.assigns) if n not in port_names]

        def _lookup_base_meta(base_name: str) -> Dict[str, Any]:
            if base_name in attr:
                return attr[base_name]
            for p in (ir.inputs + ir.outputs):
                if p.name == base_name:
                    return {'width': p.width, 'signed': int(p.signed), 'type': 'wire'}
            return {'width': 1, 'signed': 0, 'type': 'wire'}

        regs, wires = [], []
        for name in internal:
            meta = attr.get(name)
            if meta is None:
                # <base>_<idx>, where idx is an integer (e.g. acc_0)
                head, _, tail = name.rpartition('_')
                meta = (_lookup_base_meta(head) if head and tail.isdigit()
                        else {'width': 1, 'signed': 0, 'type': 'wire'})
            width, signed = int(meta.get('width', 1)), int(meta.get('signed', 0)) == 1
            self._decl[name] = (width, signed)
            row = ("reg" if name in reg_lhs else "wire", "signed" if signed else "",
                   _range(width), name + ";")
            (regs if name in reg_lhs else wires).append(row)

        # ----------------- Lists read at a variable index -----------------
        self._functions: Dict[Any, List[Any]] = {}  # table -> [name, index width, index signed]
        self._blocks: List[str] = []                          # (a list of signals' case blocks)
        assigns, blocks = [], {}                  # (an assignment's case blocks, by its place)
        for a in ir.assigns:
            tree = vprint.tree_or_none(a.right) if "?" in a.right else None
            if tree is not None:
                self._blocks = []
                new = self._lookups(tree)
                if new != tree:
                    a = dataclasses.replace(a, right=vprint.text(new))
                    blocks[len(assigns)] = self._blocks
            assigns.append(a)
        known = port_names | set(internal)
        for name in [n for n in self._decl if n not in known]:     # (the case blocks' regs)
            width, signed = self._decl[name]
            regs.append(("reg", "signed" if signed else "", _range(width), name + ";"))
        decls = _columns(regs + wires)

        # ----------------- Registers' updates -----------------
        always = ''
        reg_assigns = [a for a in assigns if a.is_reg]
        if ir.has_clk and reg_assigns:
            always = self._registers(reg_assigns)

        # ----------------- Logic -----------------
        # (each Python statement's assignments under it, as a comment)
        stmts, first, header, generated = {}, 1, "", False
        if source is not None:
            fname, text, first, path = source
            stmts = _statements(text)
            stmts.pop(1, None)                   # (the def line: code made by the passes)
            generated = path.startswith("<")       # (numpy code written out as Python)
            where = (f" ({path.strip('<>')})" if generated
                     else f" ({os.path.basename(path)}, line {first})" if path else "")
            header = f"// Generated by py2ver from {fname}(){where}"
        logic, last = [], None
        for k, a in enumerate(assigns):
            if not a.is_reg and a.line != last and a.line in stmts:
                if logic:
                    logic.append("")
                at = "" if generated else f"line {first + a.line - 1}: "
                logic.append(f"// {at}{stmts[a.line]}")
            logic += blocks.get(k, [])
            if a.is_reg:
                continue
            last = a.line
            head = f"assign {a.left} = "
            logic.append(head + vprint.tidy(a.right, len(head)) + ";")

        items = ['\n'.join(decls)] if decls else []
        if ir.items:
            items.append('\n'.join(ir.items))
        items += [self._function(key, *f) for key, f in self._functions.items()]
        if always:
            items.append(always)
        if logic:
            items.append('\n'.join(logic))
        items.append(self._tpl('dumpwaves.txt').render({'top_name': ir.name}))

        return self._tpl('moduledef.txt').render({
            'header': header,
            'modulename': ir.name,
            'paramlist': '',
            'portlist': portlist,
            'items': items,
        }) + "\n"

    # --------------------------------------------------------------------
    # Lists read at a variable index
    # --------------------------------------------------------------------

    def _lookups(self, node: Any) -> Any:
        """node with each selection tree in it that _lookup takes as a case table
        read: a call of the table's function, or its reg."""
        kind = node[0]
        if kind == "tern":
            read = self._lookup(node)
            if read is not None:
                return read
        f = self._lookups
        if kind == "atom":
            return node
        if kind == "sel":
            return ("sel", f(node[1]), [f(p) for p in node[2]], node[3])
        if kind == "call":
            return ("call", node[1], [f(x) for x in node[2]])
        if kind == "cat":
            return ("cat", [f(x) for x in node[1]])
        if kind == "rep":
            return ("rep", f(node[1]), f(node[2]))
        if kind == "un":
            return ("un", node[1], f(node[2]))
        if kind == "bin":
            return ("bin", node[1], f(node[2]), f(node[3]))
        return ("tern", f(node[1]), f(node[2]), f(node[3]))

    def _lookup(self, node: Any) -> Any:
        """The read of a case table that gives the same bits as node, a selection
        tree on one index (`i < 2 ? (i < 1 ? a : b) : (i < 3 ? c : d)`, the
        translator's list read at a variable index); None if node isn't one.

        The table's value, a function's result or a reg, has the selection's
        width (its widest element's) and sign (signed if every element is), so
        it takes part in the expression around it as the selection did. Each
        element is extended to that width by its own sign in the table, by the
        expression's sign in the selection: the same bits when the element is
        that wide, unsigned, or a number whose top bit is clear."""
        c = node[1]
        if not (c[0] == "bin" and c[1] == "<" and c[2][0] == "atom" and c[2][1] in self._decl):
            return None
        idx = c[2]
        pieces: List[Tuple[int | None, int | None, Any]] = []      # (from, to, element)
        if not self._pieces(node, idx, None, None, pieces) or len(pieces) < LOOKUP_MIN:
            return None
        info = []
        for _, _, e in pieces:
            lit = vprint.literal(e)
            if lit is not None and ("'" in e[1] or lit[2] < 1 << 31):
                info.append(lit)
            elif lit is None and e[1] in self._decl:
                info.append((*self._decl[e[1]], None))
            else:
                return None
        width, signed = max(w for w, _, _ in info), all(s for _, s, _ in info)
        if not all(w == width or not s or (v is not None and v < 1 << (w - 1)) for w, s, v in info):
            return None
        # (the comparisons and the case items compare values: the index is signed
        #  or every bound is a number >= 0)
        iw, isigned = self._decl[idx[1]]
        bounds = [b for lo, hi, _ in pieces for b in (lo, hi) if b is not None]
        if (min(bounds) < 0 and not isigned) or max(bounds) - min(bounds) >= LOOKUP_MAX:
            return None
        decl = ("signed " if signed else "") + _range(width)
        if all(i[2] is not None for i in info):
            # a table of numbers: a function, one for every read of the same table,
            # its index wide enough for every read's (a value passed as it is)
            key = (decl, tuple(pieces))
            if key not in self._functions:
                self._functions[key] = [self._fresh("table"), iw, isigned]
            f = self._functions[key]
            if f[2] != isigned:
                f[1] = max(f[1] + (not f[2]), iw + (not isigned))
                f[2] = True
            else:
                f[1] = max(f[1], iw)
            return ("call", f[0], [idx])
        # a list of signals: a reg (a function reading signals isn't evaluated
        # again when they change); named after the list, xs_0, xs_1, ... -> xs
        entries = self._entries(pieces, iw, isigned)[0]
        base = "lookup"
        named = [(v, e[1]) for v, e in entries if v >= 0]
        m = re.fullmatch(r"(\w+)_0", named[0][1]) if named and named[0][0] == 0 else None
        if m and all(e == f"{m.group(1)}_{v}" for v, e in named):
            base = m.group(1)
        name = self._fresh(f"{base}__at")
        self._decl[name] = (width, signed)
        self._blocks.append("always @*\n" + "\n".join(
            IND + line for line in self._case(idx[1], name, pieces, iw, isigned)))
        return ("atom", name)

    def _function(self, key: Tuple[str, Any], name: str, iw: int, isigned: bool) -> str:
        decl, pieces = key
        index = ("signed " if isigned else "") + _range(iw)
        return "\n".join([f"function {decl} {name}(input {index} i);".replace("  ", " ")]
                         + [IND + line for line in self._case("i", name, list(pieces), iw, isigned)]
                         + ["endfunction"])

    @staticmethod
    def _entries(pieces: List[Tuple[int | None, int | None, Any]], iw: int, isigned: bool):
        """([(index, element)] from the element below the lowest bound to the
        highest bound, as far as the index reaches; (the first index, the
        element below it if the index reaches there, the element above the last
        if it reaches there))."""
        bounds = [b for lo, hi, _ in pieces for b in (lo, hi) if b is not None]
        lowest, highest = ((-(1 << (iw - 1)), (1 << (iw - 1)) - 1) if isigned
                           else (0, (1 << iw) - 1))
        start, end = max(min(bounds) - 1, lowest), min(max(bounds), highest)

        def at(v: int) -> Any:
            return next(e for lo, hi, e in pieces
                        if (lo is None or lo <= v) and (hi is None or v < hi))
        return ([(v, at(v)) for v in range(start, end + 1)],
                (start, at(start - 1) if lowest < start else None,
                 at(end + 1) if end < highest else None))

    @staticmethod
    def _pieces(node: Any, idx: Any, lo: int | None, hi: int | None,
                out: List[Tuple[int | None, int | None, Any]]) -> bool:
        """out += (from, to, element) of selection tree node on index idx over
        [lo, hi) (None: unbounded); False if node isn't one."""
        if node[0] == "atom":
            out.append((lo, hi, node))
            return True
        if node[0] != "tern":
            return False
        c = node[1]
        bound = _integer(c[3]) if c[0] == "bin" and c[1] == "<" and c[2] == idx else None
        if bound is None:
            return False
        return (Renderer._pieces(node[2], idx, lo, bound if hi is None else min(bound, hi), out)
                and Renderer._pieces(node[3], idx, bound if lo is None else max(bound, lo), hi,
                                     out))

    @staticmethod
    def _case(index: str, out: str, pieces: List[Tuple[int | None, int | None, Any]],
              iw: int, isigned: bool) -> List[str]:
        """case (index): out = the element at each index value; the values below
        and above the table the default."""
        entries, (start, below, above) = Renderer._entries(pieces, iw, isigned)
        if below is None or above is None or below == above:
            default = above if below is None else below
        else:
            bound = ("atom", str(start)) if start >= 0 else ("un", "-", ("atom", str(-start)))
            default = ("tern", ("bin", "<", ("atom", index), bound), below, above)
        w = max(len(f"{v}:") for v, _ in entries)
        lines = [f"{IND}{f'{v}:'.ljust(w)} {out} = {vprint.text(e)};" for v, e in entries]
        if default is not None:
            lines.append(f"{IND}default: {out} = {vprint.text(default)};")
        return [f"case ({index})"] + lines + ["endcase"]

    def _fresh(self, base: str) -> str:
        """base1, base2, ...: the first not declared and no function's name."""
        taken = set(self._decl) | {f[0] for f in self._functions.values()}
        k = 1
        while f"{base}{k}" in taken:
            k += 1
        return f"{base}{k}"

    # --------------------------------------------------------------------
    # Registers
    # --------------------------------------------------------------------

    def _operand(self, node: Any) -> Tuple[int, bool, int | None] | None:
        """(width, signed, value if a number) of a selection's operand, or None
        if it isn't a declared name, a number or $signed / $unsigned of a name."""
        lit = vprint.literal(node)
        if lit is not None:
            return lit
        if node[0] == "atom" and node[1] in self._decl:
            return (*self._decl[node[1]], None)
        if node[0] == "call" and node[1] in ("$signed", "$unsigned") and len(node[2]) == 1:
            inner = self._operand(node[2][0])
            if inner is not None and inner[2] is None:
                return inner[0], node[1] == "$signed", None
        return None

    def _register_update(self, a: Assignment) -> Tuple[str | None, str | None, str] | None:
        """(reset value, enable, next value) of `r <= rst ? init : (en ? next : r)`
        (or `en ? next : r`), if writing it as `if (rst) r <= init; else if (en)
        r <= next;` gives the same bits; else None.

        In the selection, init and next are extended to its width (32 bits with
        an unsized number), signed only if all its operands are, then cut to r's
        width; on their own, each by its own sign to r's width. The low bits
        agree when the operand is at least as wide as r, or extends the same
        way, or is a number whose top bit is clear."""
        tree = vprint.tree_or_none(a.right)
        if tree is None or tree[0] != "tern":
            return None
        init = None
        if tree[1] == ("atom", "rst"):
            init, tree = tree[2], tree[3]
            if tree[0] != "tern":
                return None
        enable, nxt, hold = tree[1], tree[2], tree[3]
        own = ("atom", a.left)
        if hold != own and not (hold[0] == "call" and hold[1] in ("$signed", "$unsigned")
                                and hold[2] == [own]):
            return None
        if a.left not in self._decl:
            return None
        width = self._decl[a.left][0]
        operands = [x for x in (init, nxt, hold) if x is not None]
        info = [self._operand(x) for x in operands]
        if any(i is None for i in info):
            return None
        signed = all(i[1] for i in info)
        for node, (w, s, value) in zip(operands, info):
            if node is hold:
                continue
            if not (w >= width or s == signed or (value is not None and value < (1 << (w - 1)))):
                return None
        return (vprint.text(init) if init is not None else None, vprint.text(enable),
                vprint.text(nxt))

    def _registers(self, reg_assigns: List[Assignment]) -> str:
        """The always blocks: a block per (reset, enable) the updates share, and
        one for the updates that aren't of that form."""
        groups: Dict[Tuple[bool, str], List[Tuple[str, str | None, str]]] = {}
        other = []
        for a in reg_assigns:
            u = self._register_update(a)
            if u is None:
                head = f"{IND}{a.left} <= "
                other.append(head + vprint.tidy(a.right, len(head)) + ";")
                continue
            init, enable, nxt = u
            groups.setdefault((init is not None, enable), []).append((a.left, init, nxt))
        blocks = []
        for (has_rst, enable), regs in groups.items():
            if len(regs) == 1:
                r, init, nxt = regs[0]
                if has_rst:
                    blocks.append(f"always @(posedge clk)\n{IND}if (rst) {r} <= {init};\n"
                                  f"{IND}else if ({enable}) {r} <= {nxt};")
                else:
                    blocks.append(f"always @(posedge clk) if ({enable}) {r} <= {nxt};")
                continue
            lines = ["always @(posedge clk)"]
            if has_rst:
                lines.append(f"{IND}if (rst) begin")
                lines += [f"{IND * 2}{r} <= {init};" for r, init, _ in regs]
                lines.append(f"{IND}end else if ({enable}) begin")
            else:
                lines.append(f"{IND}if ({enable}) begin")
            lines += [f"{IND * 2}{r} <= {nxt};" for r, _, nxt in regs]
            lines.append(f"{IND}end")
            blocks.append("\n".join(lines))
        if len(other) == 1 and "\n" not in other[0]:
            blocks.append("always @(posedge clk)" + other[0][len(IND) - 1:])
        elif other:
            blocks.append("always @(posedge clk) begin\n" + "\n".join(other) + "\nend")
        return "\n\n".join(blocks)
