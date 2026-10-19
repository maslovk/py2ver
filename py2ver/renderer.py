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
  - above the logic each Python statement became, the statement, with its
    line in the file
"""
import ast
import os
import textwrap
from typing import Any, Dict, List, Sequence, Tuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import vprint
from .ast_ir import Assignment, ModuleIR

IND = "  "
_CONTROL = ("clk", "rst", "ce", "vin")


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
        decls = _columns(regs + wires)

        # ----------------- Registers' updates -----------------
        always = ''
        reg_assigns = [a for a in ir.assigns if a.is_reg]
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
        for a in ir.assigns:
            if a.is_reg:
                continue
            if a.line != last and a.line in stmts:
                if logic:
                    logic.append("")
                at = "" if generated else f"line {first + a.line - 1}: "
                logic.append(f"// {at}{stmts[a.line]}")
            last = a.line
            head = f"assign {a.left} = "
            logic.append(head + vprint.tidy(a.right, len(head)) + ";")

        items = ['\n'.join(decls)] if decls else []
        if ir.items:
            items.append('\n'.join(ir.items))
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
