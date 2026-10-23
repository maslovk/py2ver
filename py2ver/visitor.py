import ast
import functools
import logging
import re
from typing import Dict, Any, List
from .ast_ir import ModuleIR, Port, Assignment, resolve_meta
from .fixed import FIT_LIST
from .semantics import Unsupported, analyze, parse

log = logging.getLogger(__name__)

try:
    import textwrap
    indent = textwrap.indent
except Exception:
    def indent(text, prefix, predicate=None):
        if predicate is None:
            def predicate(x): return x and not x.isspace()
        ret = []
        for line in text.split('\n'):
            if predicate(line):
                ret.append(prefix)
            ret.append(line)
            ret.append('\n')
        return ''.join(ret[:-1])


def _int_literal(node: ast.expr) -> int | None:
    """The integer a constant is (5, or -5 written as minus 5), else None."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _int_literal(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


# ------------------------------------------------------------
# Attribute normalization
# ------------------------------------------------------------
def normalize_attr(attr: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in attr.items():
        if not isinstance(v, dict):
            out[k] = {'signed': 0, 'width': 1, 'type': 'wire'}
            continue
        out[k] = {
            'signed': int(v.get('signed', 0)),
            'width': int(v.get('width', 1)),
            'type': str(v.get('type', 'wire')),
        }
        if 'length' in v:
            out[k]['length'] = int(v['length'])
        if 'memory' in v:
            out[k]['memory'] = str(v['memory'])
    return out


INPUT_ARRAY_MIN = 64      # a list argument this long, read by index: an array of its ports

# ------------------------------------------------------------
# Supported binary operators
# ------------------------------------------------------------
BINOP_TOKENS = {
    ast.Add: " + ",
    ast.Sub: " - ",
    ast.Mult: " * ",
    # Python / is truncated toward zero by int(); // floors and is lowered to
    # Verilog by semantics.exact_ir (" // " is not Verilog on its own)
    ast.Div: " / ",
    ast.FloorDiv: " // ",
    # Python % (sign of the divisor) is lowered by semantics.exact_ir
    ast.Mod: " % ",
    ast.BitAnd: " & ",
    ast.BitOr: " | ",
    ast.BitXor: " ^ ",
    ast.LShift: " << ",
    # Python right shift is arithmetic for negative integers; match with Verilog >>>.
    ast.RShift: " >>> ",
}


# ------------------------------------------------------------
# AST → IR Visitor
# ------------------------------------------------------------
_IDENT = re.compile(r"(?<![\w'$])[A-Za-z_]\w*")      # (names in an expression; not 8'sd5's sd5)

def _at_line(method):
    """Assignments made while visiting the statement come from its line."""
    @functools.wraps(method)
    def visit(self, node, *args, **kwargs):
        outer = self._line
        self._line = getattr(node, "lineno", None) or outer
        try:
            return method(self, node, *args, **kwargs)
        finally:
            self._line = outer
    return visit


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, attr: Dict[str, Any], indentsize: int = 2):
        self.attr = normalize_attr(attr)
        self.indent = functools.partial(indent, prefix=' ' * indentsize)

        self.top_name = ''
        self.input_args_list: List[str] = []
        self.input_args_width_list: List[int] = []
        self.output_args_list: List[str] = []
        self.output_args_width_list: List[int] = []
        self.assigns: List[Assignment] = []
        self.has_clk = False

        self.input_bit_count = 0
        self.output_bit_count = 0

        # stack of name substitutions (for loops / SSA)
        # each entry is a dict {var_name: replacement_str}
        self._name_env_stack: List[Dict[str, str]] = []

        # Versioned variables: reassigning y creates y__v2, y__v3, ...; reads
        # see the latest version. Names declared in attr keep their width for
        # every version; undeclared ones get an exact inferred width.
        self._declared = set(self.attr)
        # Complex variables (by Python name); z is held as z_re / z_im
        self._complex: set[str] = set()
        # Lists (by Python name) -> length; element k of xs is variable xs_k
        self._lists: Dict[str, int] = {}
        # Versions whose value is a compile-time integer
        self._const_of: Dict[str, int] = {}
        # Function parameters as ports: (param, [ports] for a list or None)
        self.arg_groups: List[tuple[str, List[str] | None]] = []
        # Returned arguments get their own output port: variable -> port name
        self.output_renames: Dict[str, str] = {}
        # Block-RAM memories (list arguments with 'memory': 'ram'): name -> info.
        # Reads/writes become <mem>__raddr / __we / __waddr / __wdata signals and
        # the read data is <mem>__rdata; ram_ports holds their final names.
        self._rams: Dict[str, Dict[str, Any]] = {}
        self.ram_ports: Dict[str, Dict[str, str | None]] = {}
        # A process's clock: its states' code side by side, so a read after a
        # write in the text may be in another clock (process.py checks each clock)
        self.reads_after_writes = False
        self._tmp_count = 0
        self._line: int | None = None      # the Python line being translated
        self._ssa: Dict[str, str] = {}
        self._ssa_count: Dict[str, int] = {}
        # Tables of at least rom_min constants become ROM reads: the read's signal ->
        # {'index', 'entries', ...}. A pipelined design reads them from block RAM
        # (a register stage of its own), others as arrays (see core._rom_arrays)
        self.rom_min = 0
        self.rom_reads: Dict[str, Dict[str, Any]] = {}
        self._defs: Dict[str, ast.expr] = {}

    # ---------- helpers ----------
    def _attrs(self, name: str) -> Dict[str, Any]:
        return self.attr.get(name, {'signed': 0, 'width': 1, 'type': 'wire'})

    def _clone_attr(self, src: str, dst: str):
        meta = dict(self._attrs(src))
        self.attr[dst] = meta

    def _any_reg_outputs(self) -> bool:
        """Detect if any output port is a reg (requires clk)."""
        for name in self.output_args_list:
            meta = self._attrs(name)
            if isinstance(meta, dict) and meta.get('type') == 'reg':
                return True
        return False

    def _resolve_name(self, name: str) -> str:
        """
        Resolve a variable name through the substitution stack.
        The most recent binding wins.
        """
        for env in reversed(self._name_env_stack):
            if name in env:
                return env[name]
        return self._ssa.get(name, name)

    def _is_defined(self, name: str) -> bool:
        return name in self._ssa or any(name in env for env in self._name_env_stack)

    def _infer_meta(self, rhs: str) -> Dict[str, int]:
        """Exact width/sign of an expression, as semantics.exact_ir computes it."""
        try:
            info = analyze(parse(rhs), lambda n: resolve_meta(n, self.attr, {}))
        except Unsupported as e:
            log.warning("Can't size %s (%s): 1 bit; declare it in attr", rhs, e)
            return {"width": 1, "signed": 0}
        if info.nonneg:
            return {"width": max(1, info.bits - 1), "signed": 0}
        return {"width": info.bits, "signed": 1}

    def _new_version(self, var: str) -> str:
        n = self._ssa_count.get(var, 0) + 1
        self._ssa_count[var] = n
        name = var if n == 1 else f"{var}__v{n}"
        self._ssa[var] = name
        return name

    def _assign_var(self, var: str, rhs: str, is_reg: bool | None = None) -> str:
        """Assign rhs to a new version of var; returns the version's name."""
        name = self._new_version(var)
        base = var[:-3] if var[-3:] in ("_re", "_im") else None
        elem = re.match(r"(.+)_(\d+)$", var)
        list_base = elem.group(1) if elem and elem.group(1) in self._lists else None
        if var in self._declared:
            self.attr[name] = {k: v for k, v in self.attr[var].items() if k != "length"}
        elif base in self._complex and base in self._declared:
            # Declaring z covers z_re and z_im
            self.attr[name] = dict(self.attr[base])
        elif list_base in self._declared:
            # Declaring xs covers its elements xs_0, xs_1, ...
            self.attr[name] = {k: v for k, v in self.attr[list_base].items() if k != "length"}
        else:
            meta = self._infer_meta(rhs)
            self.attr[name] = {"signed": meta["signed"], "width": meta["width"], "type": "wire"}
        if is_reg is None:
            is_reg = self.attr[name].get("type") == "reg"
        value = self._fold(rhs)
        if value is not None and not is_reg:
            self._const_of[name] = value
        self.assigns.append(Assignment(left=name, right=rhs, is_reg=is_reg, line=self._line))
        return name

    def _temp(self, rhs: str) -> str:
        """A fresh internal variable holding rhs (used once-computed index/value)."""
        self._tmp_count += 1
        return self._assign_var(f"__t{self._tmp_count}", rhs)

    @staticmethod
    def _target_name(tgt: ast.expr) -> str:
        if not isinstance(tgt, ast.Name):
            raise NotImplementedError(f"Assignment target must be a name, got {ast.unparse(tgt)}")
        return tgt.id

    # ---------- complex values: each complex variable z is z_re / z_im ----------
    def _is_complex(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self._complex
        if isinstance(node, ast.Constant):
            return isinstance(node.value, complex)
        if isinstance(node, ast.Call):
            f = node.func
            return ((isinstance(f, ast.Name) and f.id == "complex")
                    or (isinstance(f, ast.Attribute) and f.attr == "conjugate"))
        if isinstance(node, ast.BinOp):
            return self._is_complex(node.left) or self._is_complex(node.right)
        if isinstance(node, ast.UnaryOp):
            return self._is_complex(node.operand)
        if isinstance(node, ast.IfExp):
            return self._is_complex(node.body) or self._is_complex(node.orelse)
        return False

    def _visit_cplx(self, node: ast.AST) -> tuple[str, str]:
        """(real, imaginary) expressions of a complex-valued (or real) expression."""
        if not self._is_complex(node):
            return self.visit(node), "0"
        if isinstance(node, ast.Name):
            return self._resolve_name(f"{node.id}_re"), self._resolve_name(f"{node.id}_im")
        if isinstance(node, ast.Constant):
            v = node.value
            if v.real != int(v.real) or v.imag != int(v.imag):
                raise NotImplementedError(f"Only integer complex constants are supported: {v}")
            return str(int(v.real)), str(int(v.imag))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):          # z.conjugate()
                re, im = self._visit_cplx(node.func.value)
                return re, _neg(im)
            if node.keywords or not 1 <= len(node.args) <= 2:
                raise NotImplementedError("complex() takes one or two positional arguments")
            if any(self._is_complex(a) for a in node.args):
                raise NotImplementedError("complex() arguments must be real")
            re = self.visit(node.args[0])
            im = self.visit(node.args[1]) if len(node.args) == 2 else "0"
            return re, im
        if isinstance(node, ast.UnaryOp):
            re, im = self._visit_cplx(node.operand)
            if isinstance(node.op, ast.USub):
                return _neg(re), _neg(im)
            if isinstance(node.op, ast.UAdd):
                return re, im
            raise NotImplementedError(f"{type(node.op).__name__} on a complex value is not supported")
        if isinstance(node, ast.IfExp):
            cond = self.visit(node.test)
            (tr, ti), (er, ei) = self._visit_cplx(node.body), self._visit_cplx(node.orelse)
            return f"(({cond}) ? ({tr}) : ({er}))", f"(({cond}) ? ({ti}) : ({ei}))"
        if isinstance(node, ast.BinOp):
            (ar, ai), (br, bi) = self._visit_cplx(node.left), self._visit_cplx(node.right)
            if isinstance(node.op, ast.Add):
                return _add(ar, br), _add(ai, bi)
            if isinstance(node.op, ast.Sub):
                return _sub(ar, br), _sub(ai, bi)
            if isinstance(node.op, ast.Mult):
                # (ar + j ai)(br + j bi) = (ar br - ai bi) + j (ar bi + ai br)
                return _sub(_mul(ar, br), _mul(ai, bi)), _add(_mul(ar, bi), _mul(ai, br))
            raise NotImplementedError(
                f"{type(node.op).__name__} on complex values is not supported (only + - *)"
            )
        raise NotImplementedError(f"Unsupported complex expression: {ast.unparse(node)}")

    def _assign_value(self, var: str, value: ast.AST) -> None:
        """Assign a real or complex expression to var."""
        if self._is_complex(value):
            re, im = self._visit_cplx(value)
            self._complex.add(var)
            self._assign_var(f"{var}_re", re)
            self._assign_var(f"{var}_im", im)
        else:
            self._complex.discard(var)
            self._assign_var(var, self.visit(value))

    def visit_Attribute(self, node: ast.Attribute):
        """z.real and z.imag."""
        if node.attr in ("real", "imag"):
            re, im = self._visit_cplx(node.value)
            return re if node.attr == "real" else im
        raise NotImplementedError(f"Unsupported attribute: {ast.unparse(node)}")

    # ---------- lists: list xs of length n is held as variables xs_0 .. xs_{n-1} ----------
    def _is_list(self, node: ast.AST) -> bool:
        if isinstance(node, (ast.List, ast.Tuple)):
            return True
        if isinstance(node, ast.Name):
            if node.id in self._rams:
                raise NotImplementedError(
                    f"RAM '{node.id}' can only be used as {node.id}[i] (read or write)"
                )
            return node.id in self._lists
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self._is_list(node.left) and self._is_list(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return self._is_list(node.left) or self._is_list(node.right)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            return self._is_list(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id in ("list", "tuple", FIT_LIST)
        return False

    def _visit_list(self, node: ast.AST) -> List[str]:
        """Element expressions of a list-valued expression."""
        if isinstance(node, (ast.List, ast.Tuple)):
            for e in node.elts:
                if self._is_list(e) or self._is_complex(e):
                    raise NotImplementedError(
                        "List elements must be integers (no nested lists or complex values)"
                    )
            return [self.visit(e) for e in node.elts]
        if isinstance(node, ast.Name):
            return [self._resolve_name(f"{node.id}_{k}") for k in range(self._lists[node.id])]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self._visit_list(node.left) + self._visit_list(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            lst, n = (node.left, node.right) if self._is_list(node.left) else (node.right, node.left)
            return self._visit_list(lst) * max(self._eval_int(n), 0)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            sl = node.slice
            bound = [None if b is None else self._eval_int(b) for b in (sl.lower, sl.upper, sl.step)]
            return self._visit_list(node.value)[slice(*bound)]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == FIT_LIST:
            # Fixed point: each element shifted, rounded, clamped (see fixed.fit_list)
            shift, rnd, lo, hi = (ast.literal_eval(a) for a in node.args[1:])
            out = []
            for e in self._visit_list(node.args[0]):
                x: ast.expr = ast.Name(id=self._temp(e), ctx=ast.Load())
                if shift > 0:
                    if rnd:
                        x = ast.BinOp(left=x, op=ast.Add(), right=ast.Constant(1 << (shift - 1)))
                    x = ast.BinOp(left=x, op=ast.RShift(), right=ast.Constant(shift))
                elif shift < 0:
                    x = ast.BinOp(left=x, op=ast.LShift(), right=ast.Constant(-shift))
                if lo is not None:
                    lo_node = (ast.UnaryOp(op=ast.USub(), operand=ast.Constant(-lo)) if lo < 0
                               else ast.Constant(lo))
                    x = ast.Call(func=ast.Name(id="max", ctx=ast.Load()), keywords=[], args=[
                        lo_node, ast.Call(func=ast.Name(id="min", ctx=ast.Load()), keywords=[],
                                          args=[ast.Constant(hi), x])])
                out.append(self.visit(x))
            return out
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if len(node.args) != 1 or node.keywords:
                raise NotImplementedError(f"{node.func.id}() takes one argument here")
            arg = node.args[0]
            if self._is_list(arg):
                return self._visit_list(arg)
            if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                    and arg.func.id == "range"):
                return [str(v) for v in range(*self._parse_range(arg))]
        raise NotImplementedError(f"Unsupported list expression: {ast.unparse(node)}")

    def _assign_list(self, var: str, value: ast.AST) -> None:
        # (bindings made by inlining a helper, __<helper><k>_..., copy by design)
        if isinstance(value, ast.Name) and value.id in self._lists and value.id != var \
                and not var.startswith("__"):
            raise NotImplementedError(
                f"'{var} = {value.id}' would make both names the same list in Python; "
                f"copy it with '{var} = list({value.id})'"
            )
        elems = self._visit_list(value)      # read everything before assigning
        self._complex.discard(var)
        self._lists[var] = len(elems)
        for k, e in enumerate(elems):
            self._assign_var(f"{var}_{k}", e)

    def _index_keys(self, idx: str, n: int) -> List[tuple[int, int]]:
        """(index value, element) pairs, including Python's negative indices if idx can be < 0."""
        keys = [(k, k) for k in range(n)]
        if self._infer_meta(idx)["signed"]:
            keys = [(k - n, k) for k in range(n)] + keys
        return keys

    def _select(self, elems: List[str], idx_node: ast.AST) -> str:
        """elems[idx] for a variable idx: a balanced mux tree (a ROM if elems are constants)."""
        idx = self._temp(self.visit(idx_node))
        keys = self._index_keys(idx, len(elems))
        array = self._input_array(elems, idx)
        if array is not None:
            return array

        def tree(part):
            if len(part) == 1:
                return elems[part[0][1]]
            mid = len(part) // 2
            return (f"(({idx}) < ({part[mid][0]})) ? ({tree(part[:mid])}) "
                    f": ({tree(part[mid:])})")
        # Out-of-range indices (an IndexError in Python) read an arbitrary element
        return f"({tree(keys)})"

    def _input_array(self, elems: List[str], idx: str) -> str | None:
        """A long list argument read at an index that can't be negative: an array of
        its ports read by the index (a mux of thousands of inputs as one expression
        is more than Verilator reads), as a ROM read whose entries are signals."""
        if len(elems) < INPUT_ARRAY_MIN or self._infer_meta(idx)["signed"] \
                or not all(e in self.input_args_list for e in elems):
            return None
        metas = {(int(m["width"]), int(m["signed"])) for m in map(self._infer_meta, elems)}
        if len(metas) != 1:
            return None
        w, sgn = metas.pop()
        var = f"__rom{len(self.rom_reads) + 1}"
        self.attr[var] = {"width": w, "signed": sgn, "type": "wire"}
        self._declared.add(var)
        name = self._assign_var(var, idx)
        self.rom_reads[name] = dict(width=w, signed=sgn, index=idx, signals=list(elems),
                                    address_bits=max(1, (len(elems) - 1).bit_length()))
        return name

    def _store_index(self, target: ast.Subscript, rhs: str, env: Dict[str, str] | None = None):
        """xs[i] = rhs, at top level (env is None) or into a branch environment."""
        if isinstance(target.value, ast.Name) and target.value.id in self._rams:
            return self._ram_write(target.value.id, target, rhs, env)
        if not (isinstance(target.value, ast.Name) and target.value.id in self._lists):
            raise NotImplementedError(f"Can only assign into a list variable: {ast.unparse(target)}")
        var = target.value.id
        n = self._lists[var]
        if isinstance(target.slice, ast.Slice):
            raise NotImplementedError("Slice assignment is not supported")

        def put(k, value):
            if env is None:
                self._assign_var(f"{var}_{k}", value)
            else:
                env[f"{var}_{k}"] = f"({value})"

        try:
            k = self._eval_int(target.slice)
        except NotImplementedError:
            k = None
        if k is not None:
            if not -n <= k < n:
                raise NotImplementedError(f"Index {k} is out of range for '{var}' (length {n})")
            put(k % n, rhs)
            return
        idx = self._temp(self.visit(target.slice))
        value = self._temp(rhs)
        hits: Dict[int, List[int]] = {}
        for key, k in self._index_keys(idx, n):
            hits.setdefault(k, []).append(key)
        for k, keys in hits.items():
            cond = " || ".join(f"(({idx}) == ({key}))" for key in keys)
            put(k, f"({cond}) ? ({value}) : ({self._resolve_name(f'{var}_{k}')})")

    def _list_method(self, call: ast.Call, env: Dict[str, str] | None = None) -> None:
        """xs.append(v) / xs.extend(list) as a statement."""
        f = call.func
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id in self._lists and f.attr in ("append", "extend")):
            raise NotImplementedError(f"Unsupported statement: {ast.unparse(call)}")
        if env is not None:
            raise NotImplementedError(
                f"'{ast.unparse(call)}' inside an if changes the list's length on one path only"
            )
        var = f.value.id
        new = [self.visit(call.args[0])] if f.attr == "append" else self._visit_list(call.args[0])
        n = self._lists[var]
        self._lists[var] = n + len(new)
        for k, e in enumerate(new, start=n):
            self._assign_var(f"{var}_{k}", e)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in self._rams:
            if isinstance(node.slice, ast.Slice):
                raise NotImplementedError("A RAM can only be read one element at a time")
            return self._ram_read(node.value.id, node.slice)
        if isinstance(node.slice, ast.Slice):
            raise NotImplementedError("A slice is a list; index it or assign it to a variable")
        if self.rom_min and isinstance(node.value, ast.Tuple) and len(node.value.elts) >= self.rom_min:
            values = [_int_literal(e) for e in node.value.elts]
            rom = self._rom_read(values, node.slice) if None not in values else None
            if rom is not None:
                return rom
        elems = self._visit_list(node.value)
        n = len(elems)
        try:
            k = self._eval_int(node.slice)
        except NotImplementedError:
            k = None
        if k is not None:
            if not -n <= k < n:
                raise NotImplementedError(f"Index {k} is out of range (length {n})")
            return elems[k]
        return self._select(elems, node.slice)

    def _rom_read(self, values: List[int], index: ast.expr) -> str | None:
        """values[index] as a ROM read (an array indexed by the low address bits).
        Every index Python allows reads its entry: one that can't be negative,
        one clamped as fixed.py writes a table index (max(0, min(n - 1, ...))),
        or any for a power-of-two table (the low bits of -k are n - k), as they
        are; any other index plus n when it is negative. Out-of-range indices,
        an IndexError in Python, read some entry (the array is padded to a power
        of two)."""
        try:
            self._eval_int(index)
            return None                                    # a constant index
        except NotImplementedError:
            pass
        n = len(values)
        node = self._defs.get(index.id) if isinstance(index, ast.Name) else index
        clamped = (self._ssa_count.get(getattr(index, "id", ""), 1) == 1
                   and isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "max" and len(node.args) == 2
                   and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0
                   and isinstance(node.args[1], ast.Call)
                   and isinstance(node.args[1].func, ast.Name)
                   and node.args[1].func.id == "min" and len(node.args[1].args) == 2
                   and isinstance(node.args[1].args[0], ast.Constant)
                   and node.args[1].args[0].value == n - 1)
        idx = self._temp(self.visit(index))
        if not (clamped or n & (n - 1) == 0 or not self._attrs(idx).get("signed")):
            # (an index that may be negative counts from the end, as in Python)
            idx = self._temp(f"((({idx}) < 0) ? (({idx}) + {n}) : ({idx}))")
        lo, hi = min(values), max(values)
        if lo >= 0:
            fmt = {"width": max(1, hi.bit_length()), "signed": 0}
        else:
            fmt = {"width": max((-lo - 1).bit_length(), hi.bit_length()) + 1, "signed": 1}
        var = f"__rom{len(self.rom_reads) + 1}"
        self.attr[var] = dict(fmt, type="wire")
        self._declared.add(var)
        name = self._assign_var(var, idx)
        self.rom_reads[name] = dict(fmt, index=idx, entries=list(values),
                                    address_bits=max(1, (n - 1).bit_length()))
        return name

    @_at_line
    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Constant):
            return None                      # docstring / bare constant
        if isinstance(node.value, ast.Call):
            return self._list_method(node.value)
        raise NotImplementedError(f"Unsupported statement: {ast.unparse(node)}")

    def _finalize_output(self, var: str) -> str:
        """Expose the final version of var on an output port named var."""
        current = self._resolve_name(var)
        if var in self.input_args_list:
            # An output can't share the input port's name: expose it as <var>_out
            out = f"{var}_out"
            while self._is_defined(out) or out in self.attr or out in self.input_args_list:
                out += "_"
            self.attr[out] = dict(self.attr.get(current) or self._attrs(var), type="wire")
            self.assigns.append(Assignment(left=out, right=current, is_reg=False))
            self.output_renames[var] = out
            return out
        if current == var:
            return var
        # var's first version becomes var__v1, and the port carries the last one
        first = f"{var}__v1"
        ident = re.compile(rf"\b{re.escape(var)}\b")
        for i, a in enumerate(self.assigns):
            self.assigns[i] = Assignment(
                left=first if a.left == var else a.left,
                right=ident.sub(first, a.right),
                is_reg=a.is_reg,
                line=a.line,
            )
        if var in self.attr:
            self.attr[first] = dict(self.attr[var])
        if var not in self._declared:
            self.attr[var] = dict(self.attr[current], type="wire")
        self.assigns.append(Assignment(left=var, right=current, is_reg=False))
        return var

    def _eval_int(self, node: ast.expr) -> int:
        """
        Evaluate an expression node as a compile-time integer.

        Supported:
          * integer Constant / Num
          * unary +/- on those
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Num):  # <3.8
            return int(node.n)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            val = self._eval_int(node.operand)
            return -val if isinstance(node.op, ast.USub) else val
        value = self._fold(self.visit(node))
        if value is None:
            raise NotImplementedError(
                f"Expected a value known at translation time: {ast.unparse(node)}"
            )
        return value

    def _fold(self, expr: str):
        """Value of an expression made only of constants (and constant versions), else None."""
        # (a name that isn't a constant: nothing to fold, without parsing it)
        if any(name not in self._const_of for name in _IDENT.findall(expr)):
            return None
        try:
            tree = parse(expr)
        except Unsupported:
            return None
        ops = {
            "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
            "//": lambda a, b: a // b, "%": lambda a, b: a % b,
            "/": lambda a, b: int(a / b),
            "<<": lambda a, b: a << b, ">>": lambda a, b: a >> b, ">>>": lambda a, b: a >> b,
            "&": lambda a, b: a & b, "|": lambda a, b: a | b, "^": lambda a, b: a ^ b,
            "<": lambda a, b: int(a < b), "<=": lambda a, b: int(a <= b),
            ">": lambda a, b: int(a > b), ">=": lambda a, b: int(a >= b),
            "==": lambda a, b: int(a == b), "!=": lambda a, b: int(a != b),
        }

        def ev(n):
            if n.kind == "num":
                return n.value
            if n.kind == "id":
                return self._const_of.get(n.name)
            vals = [ev(k) for k in n.kids]
            if any(v is None for v in vals):
                return None
            try:
                if n.kind == "un":
                    return {"-": lambda a: -a, "+": lambda a: a, "~": lambda a: ~a,
                            "!": lambda a: int(not a)}[n.op](vals[0])
                if n.kind == "tern":
                    return vals[1] if vals[0] else vals[2]
                return ops[n.op](*vals) if n.op in ops else None
            except (ZeroDivisionError, ValueError):
                return None

        return ev(tree)

    def _parse_range(self, iter_node: ast.expr) -> tuple[int, int, int]:
        """
        Parse a range(...) call into (start, stop, step) integers.

        Only supports range with 1–3 integer-literal arguments.
        """
        if not isinstance(iter_node, ast.Call):
            raise NotImplementedError("for-loops must iterate over range(...) or a tuple of values.")
        func = iter_node.func
        if not (isinstance(func, ast.Name) and func.id == "range"):
            raise NotImplementedError("for-loops must iterate over range(...) or a tuple of values.")

        args = iter_node.args
        if len(args) == 1:
            start, stop, step = 0, self._eval_int(args[0]), 1
        elif len(args) == 2:
            start, stop, step = self._eval_int(args[0]), self._eval_int(args[1]), 1
        elif len(args) == 3:
            start, stop, step = (
                self._eval_int(args[0]),
                self._eval_int(args[1]),
                self._eval_int(args[2]),
            )
        else:
            raise NotImplementedError("range() with 1–3 integer arguments is supported.")

        if step == 0:
            raise NotImplementedError("range() step cannot be 0.")
        return start, stop, step

    def _expr_uses_name(self, node: ast.AST, name: str) -> bool:
        """Return True if the expression AST uses the given variable name."""
        found = False

        class _Visitor(ast.NodeVisitor):
            def visit_Name(self, n: ast.Name):
                nonlocal found
                if n.id == name:
                    found = True
                self.generic_visit(n)

        _Visitor().visit(node)
        return found

    @staticmethod
    def _int_meta(v: int) -> Dict[str, int]:
        if v < 0:
            width = max(1, abs(v).bit_length() + 1)
            return {"width": width, "signed": 1}
        width = max(1, v.bit_length())
        return {"width": width, "signed": 0}

    def _expr_meta(self, node: ast.AST) -> Dict[str, int]:
        """
        Best-effort width/signed inference for undeclared temporaries.
        Conservative widths are preferred over underestimation.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return self._int_meta(int(node.value))
        if isinstance(node, ast.Num):
            return self._int_meta(int(node.n))
        if isinstance(node, ast.Name):
            name = self._resolve_name(node.id)
            # Loop unroll may resolve to literal text.
            if isinstance(name, str) and name.lstrip("-").isdigit():
                return self._int_meta(int(name))
            m = self._attrs(name)
            return {"width": int(m.get("width", 1)), "signed": int(m.get("signed", 0))}
        if isinstance(node, ast.UnaryOp):
            em = self._expr_meta(node.operand)
            if isinstance(node.op, ast.USub):
                return {"width": int(em["width"]) + 1, "signed": 1}
            if isinstance(node.op, ast.UAdd):
                return em
            if isinstance(node.op, ast.Invert):
                return em
            if isinstance(node.op, ast.Not):          # (not x: a bit, x == 0)
                return {"width": 1, "signed": 0}
            return em
        if isinstance(node, ast.BoolOp):
            return {"width": 1, "signed": 0}
        if isinstance(node, ast.Compare):
            return {"width": 1, "signed": 0}
        if isinstance(node, ast.BinOp):
            l = self._expr_meta(node.left)
            r = self._expr_meta(node.right)
            lw, rw = int(l["width"]), int(r["width"])
            ls, rs = int(l["signed"]), int(r["signed"])

            if isinstance(node.op, (ast.Add, ast.Sub)):
                return {"width": max(lw, rw) + 1, "signed": 1 if (ls or rs) else 0}
            if isinstance(node.op, ast.Mult):
                return {"width": lw + rw, "signed": 1 if (ls or rs) else 0}
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                return {"width": lw, "signed": 1 if (ls or rs) else 0}
            if isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
                return {"width": max(lw, rw), "signed": 1 if (ls or rs) else 0}
            if isinstance(node.op, ast.LShift):
                sh = 0
                try:
                    sh = max(0, int(self._eval_int(node.right)))
                except Exception:
                    # (by a value: as far as its format reaches, 2^rw - 1 at most)
                    sh = (1 << (rw - 1 if rs else rw)) - 1
                return {"width": lw + sh, "signed": ls}
            if isinstance(node.op, ast.RShift):
                return {"width": lw, "signed": ls}

        # Unknown expressions fallback to minimal safe default.
        return {"width": 1, "signed": 0}

    def _ensure_target_attr(self, left: str, value_node: ast.AST):
        if left in self.attr:
            return
        meta = self._expr_meta(value_node)
        self.attr[left] = {
            "signed": int(meta.get("signed", 0)),
            "width": max(1, int(meta.get("width", 1))),
            "type": "wire",
        }

    # ---------- visitors ----------
    def visit_Module(self, node: ast.Module):
        if not node.body:
            raise NotImplementedError("Empty module.")
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ModuleIR:
        node = _Desugar().visit(node)
        self.top_name = node.name

        # parse arguments (inputs)
        self.visit(node.args)

        seen_return = False

        # body
        for item in node.body:
            self._line = getattr(item, "lineno", None)       # (a return's outputs: its line)
            # skip docstrings or bare expression statements
            if isinstance(item, ast.Expr):
                self.visit_Expr(item)
                continue

            if isinstance(item, ast.Assign):
                self._visit_assign(item)
            elif isinstance(item, ast.If):
                self._visit_if(item)
            elif isinstance(item, ast.For):
                self.visit_For(item)
            elif isinstance(item, ast.While):
                self.visit_While(item)
            elif isinstance(item, ast.Return):
                if seen_return:
                    raise NotImplementedError("Multiple return statements are not supported.")
                seen_return = True
                vals = item.value.elts if isinstance(item.value, ast.Tuple) else [item.value]
                outs = []
                for i, v in enumerate(vals):
                    if isinstance(v, ast.Name) and v.id in self._rams:
                        raise NotImplementedError(
                            f"RAM '{v.id}' is updated in place; don't return it"
                        )
                    if isinstance(v, ast.Name) and v.id in self._lists:
                        # A list is returned as one output per element, <v>_0, <v>_1, ...
                        for k in range(self._lists[v.id]):
                            outs.append(self._finalize_output(f"{v.id}_{k}"))
                        continue
                    if not isinstance(v, ast.Name) and self._is_list(v):
                        name = f"ret{i}" if len(vals) > 1 else "ret"
                        self._assign_list(name, v)
                        outs += [f"{name}_{k}" for k in range(self._lists[name])]
                        continue
                    if isinstance(v, ast.Name) and v.id in self._complex:
                        # A complex value is returned as two outputs, <v>_re and <v>_im
                        outs.append(self._finalize_output(f"{v.id}_re"))
                        outs.append(self._finalize_output(f"{v.id}_im"))
                    elif isinstance(v, ast.Name):
                        outs.append(self._finalize_output(v.id))
                    else:
                        # Returned expression: give it an output named ret<i>
                        name = f"ret{i}" if len(vals) > 1 else "ret"
                        if self._is_defined(name) or self._is_defined(f"{name}_re"):
                            raise NotImplementedError(f"Output name '{name}' is already used")
                        self._assign_value(name, v)
                        if name in self._complex:
                            outs += [f"{name}_re", f"{name}_im"]
                        else:
                            outs.append(name)
                self.output_args_list = outs
                for name in outs:
                    meta = self._attrs(name)
                    self.output_args_width_list.append(meta['width'])
                    self.output_bit_count += meta['width']
            else:
                self.generic_visit(item)

        for mem, ram in self._rams.items():
            self.ram_ports[mem] = {
                "reads": [(self._resolve_name(f"{mem}__raddr{j}"), f"{mem}__rdata{j}")
                          for j in range(len(ram["reads"]))],
                "writes": [tuple(self._resolve_name(f"{mem}__{k}{i}")
                                 for k in ("we", "waddr", "wdata"))
                           for i in range(ram["writes"])],
            }

        # determine if clk needed
        self.has_clk = self._any_reg_outputs()

        # Build IR Ports
        inputs: List[Port] = []
        for name, width in zip(self.input_args_list, self.input_args_width_list):
            inputs.append(
                Port(
                    name=name,
                    width=width,
                    signed=bool(self._attrs(name)['signed']),
                    kind='input'
                )
            )
        if self.has_clk:
            inputs.append(Port(name='clk', width=1, signed=False, kind='clk'))

        outputs: List[Port] = []
        for name, width in zip(self.output_args_list, self.output_args_width_list):
            kind = 'reg' if self._attrs(name)['type'] == 'reg' else 'output'
            outputs.append(
                Port(
                    name=name,
                    width=width,
                    signed=bool(self._attrs(name)['signed']),
                    kind=kind
                )
            )

        return ModuleIR(
            name=self.top_name,
            inputs=inputs,
            outputs=outputs,
            assigns=self.assigns,
            has_clk=self.has_clk
        )

    def visit_arguments(self, node: ast.arguments):
        values = [self.visit(a) for a in node.args]
        for param in values:
            length = self.attr.get(param, {}).get("length")
            if self.attr.get(param, {}).get("memory") == "ram":
                self._declare_ram(param)
                self.arg_groups.append((param, None))
                continue
            if length is None:
                self.arg_groups.append((param, None))
                self._register_arg(param)
                continue
            ports = [f"{param}_{k}" for k in range(length)]
            self.arg_groups.append((param, ports))
            self._lists[param] = length
            for port in ports:
                self.attr.setdefault(port, {k: v for k, v in self.attr[param].items()
                                            if k != "length"})
                self._register_arg(port)

    def _declare_ram(self, mem: str):
        meta = self.attr[mem]
        if "length" not in meta:
            raise NotImplementedError(f"RAM '{mem}' needs a 'length'")
        elem = {"width": meta["width"], "signed": meta["signed"], "type": "wire"}
        # reads: one address per read port; writes: number of write statements
        self._rams[mem] = {"length": int(meta["length"]), "elem": elem,
                           "reads": [], "writes": 0, "written": False}

    def _ram_address(self, mem: str, index: ast.AST) -> str:
        """Address for mem[index], with Python's negative indices mapped."""
        n = self._rams[mem]["length"]
        idx = self.visit(index)
        if self._infer_meta(idx)["signed"]:
            idx = f"((({idx}) < 0) ? (({idx}) + {n}) : ({idx}))"
        return idx

    def _ram_read(self, mem: str, index: ast.AST) -> str:
        """mem[index]: a read port per distinct address (<mem>__rdata<j>)."""
        ram = self._rams[mem]
        if ram["written"] and not self.reads_after_writes:
            raise NotImplementedError(
                f"RAM '{mem}' is read after being written in the same sample; "
                "read it first (the RAM reads before it writes)"
            )
        addr = self._ram_address(mem, index)
        if addr in ram["reads"]:
            return f"{mem}__rdata{ram['reads'].index(addr)}"
        j = len(ram["reads"])
        ram["reads"].append(addr)
        self.attr[f"{mem}__rdata{j}"] = dict(ram["elem"])
        self._assign_var(f"{mem}__raddr{j}", addr)
        return f"{mem}__rdata{j}"

    def _ram_write(self, mem: str, target: ast.Subscript, rhs: str,
                   env: Dict[str, str] | None) -> None:
        """mem[index] = rhs: write slot i (<mem>__we<i>, __waddr<i>, __wdata<i>)."""
        ram = self._rams[mem]
        ram["written"] = True
        i = ram["writes"]
        ram["writes"] += 1
        we, waddr, wdata = f"{mem}__we{i}", f"{mem}__waddr{i}", f"{mem}__wdata{i}"
        addr = self._ram_address(mem, target.slice)
        if env is None:
            self._assign_var(we, "1")
            self._assign_var(waddr, addr)
            self._assign_var(wdata, rhs)
        else:
            # Inside an if: no write unless this branch is taken
            for var in (we, waddr, wdata):
                self._assign_var(var, "0")
            env[we] = "(1)"
            env[waddr] = f"({addr})"
            env[wdata] = f"({rhs})"

    def _register_arg(self, name: str):
        self._ssa_count[name] = 1
        self._ssa[name] = name
        meta = self._attrs(name)
        self.input_args_list.append(name)
        self.input_args_width_list.append(meta['width'])
        self.input_bit_count += meta['width']

    def visit_arg(self, node: ast.arg):
        return node.arg

    def visit_Return(self, node: ast.Return):
        val = self.visit(node.value)
        return val if isinstance(val, list) else [val]

    def visit_Tuple(self, node: ast.Tuple):
        return [self.visit(e) for e in node.elts]

    # ---- expressions ----
    def visit_Name(self, node: ast.Name):
        return self._resolve_name(node.id)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool):
            return "1'b1" if node.value else "1'b0"
        if isinstance(node.value, int):
            return str(node.value)
        if isinstance(node.value, str):
            return f'"{node.value}"'
        return str(node.value)

    def visit_Num(self, node: ast.Num):  # for Python <3.8
        return str(node.n)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        op = self.visit(node.op)
        return f"{op}({self.visit(node.operand)})"

    def visit_USub(self, node): return "-"
    def visit_UAdd(self, node): return "+"
    def visit_Invert(self, node): return "~"
    def visit_Not(self, node): return "!"

    def visit_BinOp(self, node: ast.BinOp):
        op_type = type(node.op)
        if op_type not in BINOP_TOKENS:
            raise NotImplementedError(f"Unsupported binary op: {op_type.__name__}")
        op = BINOP_TOKENS[op_type]
        return f"({self.visit(node.left)}){op}({self.visit(node.right)})"

    def visit_Compare(self, node: ast.Compare):
        """Support simple binary comparisons: a < b, a == b, etc."""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Only simple binary comparisons are supported.")
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            token = " == "
        elif isinstance(op, ast.NotEq):
            token = " != "
        elif isinstance(op, ast.Lt):
            token = " < "
        elif isinstance(op, ast.LtE):
            token = " <= "
        elif isinstance(op, ast.Gt):
            token = " > "
        elif isinstance(op, ast.GtE):
            token = " >= "
        else:
            raise NotImplementedError(f"Unsupported comparison operator: {type(op).__name__}")
        return f"(({left}){token}({right}))"

    def visit_BoolOp(self, node: ast.BoolOp):
        """Support 'and' / 'or' conditions."""
        if isinstance(node.op, ast.And):
            op_str = " && "
        elif isinstance(node.op, ast.Or):
            op_str = " || "
        else:
            raise NotImplementedError(f"Unsupported boolean operator: {type(node.op).__name__}")
        parts = [f"({self.visit(v)})" for v in node.values]
        return "(" + op_str.join(parts) + ")"

    def visit_IfExp(self, node: ast.IfExp):
        return (f"(({self.visit(node.test)}) ? ({self.visit(node.body)}) "
                f": ({self.visit(node.orelse)}))")

    def visit_Call(self, node: ast.Call):
        """Integer builtins: abs(x), min(a, b, ...), max(a, b, ...)."""
        if (isinstance(node.func, ast.Name) and len(node.args) == 1 and not node.keywords
                and node.func.id in ("len", "sum", "min", "max")
                and self._is_list(node.args[0])):
            elems = self._visit_list(node.args[0])
            fn = node.func.id
            if fn == "len":
                return str(len(elems))
            if not elems:
                if fn == "sum":
                    return "0"
                raise NotImplementedError(f"{fn}() of an empty list")
            if fn == "sum":
                acc = elems[0]
                for e in elems[1:]:
                    acc = f"({acc}) + ({e})"
                return acc
            op = "<" if fn == "min" else ">"
            acc = elems[0]
            for e in elems[1:]:
                acc = f"((({e}) {op} ({acc})) ? ({e}) : ({acc}))"
            return acc
        if isinstance(node.func, ast.Name) and not node.keywords:
            fn = node.func.id
            args = [] if any(self._is_complex(a) for a in node.args) else [self.visit(a) for a in node.args]
            if fn == "abs" and len(node.args) == 1 and self._is_complex(node.args[0]):
                re, im = self._visit_cplx(node.args[0])
                return f"$sqrt(({re})*({re}) + ({im})*({im}))"
            if fn == "abs" and len(args) == 1:
                x = args[0]
                return f"((({x}) < 0) ? (-({x})) : ({x}))"
            if fn in ("int", "int_") and len(args) == 1:
                return args[0]                 # (of an integer, here: itself)
            if fn in ("min", "max") and len(args) >= 2:
                op = "<" if fn == "min" else ">"
                acc = args[0]
                for a in args[1:]:
                    # Ties keep the earlier argument, like Python
                    acc = f"((({a}) {op} ({acc})) ? ({a}) : ({acc}))"
                return acc
        raise NotImplementedError(f"Unsupported call: {ast.unparse(node)}")

    def visit_While(self, node: ast.While):
        raise NotImplementedError("while-loops are not supported")

    def visit_List(self, node: ast.List):
        if len(node.elts) > 0:
            raise NotImplementedError("Non-empty list literals are not supported")
        raise NotImplementedError("List literals are not supported")

    # ---- for-loops ----
    @_at_line
    def visit_For(self, node: ast.For):
        """
        Unroll `for <var> in range(...)`. Assignments in the body create new
        versions of their variables, so accumulators (acc = acc + ...) chain
        naturally across iterations.
        """
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("Only 'for <var> in range(...)' loops are supported.")
        loop_var = node.target.id

        if self._is_list(node.iter) and not isinstance(node.iter, (ast.Tuple, ast.List)):
            for e in self._visit_list(node.iter):
                self._complex.discard(loop_var)
                self._unroll_body(node.body, {loop_var: e})
            return

        if isinstance(node.iter, (ast.Tuple, ast.List)):
            # for v in (a, b, c): Python evaluates the tuple once, up front
            bindings = []
            for elt in node.iter.elts:
                if self._is_complex(elt):
                    re, im = self._visit_cplx(elt)
                    bindings.append((True, {f"{loop_var}_re": re, f"{loop_var}_im": im}))
                else:
                    bindings.append((False, {loop_var: self.visit(elt)}))
            for is_cplx, env in bindings:
                if is_cplx:
                    self._complex.add(loop_var)
                else:
                    self._complex.discard(loop_var)
                self._unroll_body(node.body, env)
            self._complex.discard(loop_var)
            return

        start, stop, step = self._parse_range(node.iter)

        for val in range(start, stop, step):
            self._unroll_body(node.body, {loop_var: str(val)})

    def _unroll_body(self, body: List[ast.stmt], env: Dict[str, str]):
        """Visit one unrolled iteration with the loop variable bound by env."""
        self._name_env_stack.append(env)
        try:
            for stmt in body:
                if isinstance(stmt, ast.Assign):
                    self._visit_assign(stmt)
                elif isinstance(stmt, ast.If):
                    self._visit_if(stmt)
                elif isinstance(stmt, ast.For):
                    self.visit_For(stmt)
                elif isinstance(stmt, ast.Expr):
                    self.visit_Expr(stmt)
                else:
                    self.generic_visit(stmt)
        finally:
            self._name_env_stack.pop()

    # ---- assignments ----
    @_at_line
    def _visit_assign(self, node: ast.Assign):
        for tgt in node.targets:
            if isinstance(tgt, ast.Tuple):
                vals = node.value.elts if isinstance(node.value, ast.Tuple) else None
                if vals is None or len(vals) != len(tgt.elts):
                    raise NotImplementedError(
                        "Tuple assignment needs a tuple of the same length on the right"
                    )
                # All right-hand sides are read before anything is assigned
                names = [self._target_name(t) for t in tgt.elts]
                rhss = []
                for name, v in zip(names, vals):
                    if self._is_list(v):
                        if (isinstance(v, ast.Name) and v.id in self._lists and v.id != name
                                and not name.startswith("__")):
                            raise NotImplementedError(
                                f"'{name} = {v.id}' would make both names the same list in "
                                f"Python; copy it with '{name} = list({v.id})'")
                        rhss.append(("list", self._visit_list(v)))
                    else:
                        rhss.append(self._visit_cplx(v) if self._is_complex(v) else self.visit(v))
                for name, rhs in zip(names, rhss):
                    if isinstance(rhs, tuple) and rhs[0] == "list":
                        self._complex.discard(name)
                        self._lists[name] = len(rhs[1])
                        for k, e in enumerate(rhs[1]):
                            self._assign_var(f"{name}_{k}", e)
                    elif isinstance(rhs, tuple):
                        self._complex.add(name)
                        self._assign_var(f"{name}_re", rhs[0])
                        self._assign_var(f"{name}_im", rhs[1])
                    else:
                        self._complex.discard(name)
                        self._assign_var(name, rhs)
            elif isinstance(tgt, ast.Subscript):
                self._store_index(tgt, self.visit(node.value))
            elif self._is_list(node.value):
                self._assign_list(self._target_name(tgt), node.value)
            else:
                self._defs[self._target_name(tgt)] = node.value
                self._assign_value(self._target_name(tgt), node.value)

    # ---------- nested if/elif/else lowering ----------
    def _eval_stmt_block(self, stmts: List[ast.stmt]) -> Dict[str, str]:
        """
        Evaluate a *purely combinational* statement block into a mapping
        var -> expression string.

        Supported statements:
          * Assign (single target)
          * If / elif / else (recursively lowered to nested ternaries)

        Statement order is respected: later assignments override earlier ones.
        """
        env: Dict[str, str] = {}
        # Later statements in the block read values assigned earlier in it
        self._name_env_stack.append(env)
        try:
            for s in stmts:
                if isinstance(s, ast.Assign):
                    if len(s.targets) != 1:
                        raise NotImplementedError(
                            "Multiple assignment targets are not supported."
                        )
                    tgt = s.targets[0]
                    if isinstance(tgt, ast.Subscript):
                        self._store_index(tgt, self.visit(s.value), env)
                        continue
                    if isinstance(tgt, ast.Name) and self._is_list(s.value):
                        elems = self._visit_list(s.value)
                        self._lists[tgt.id] = len(elems)
                        for k, e in enumerate(elems):
                            env[f"{tgt.id}_{k}"] = f"({e})"
                        continue
                    if isinstance(tgt, ast.Tuple):
                        vals = s.value.elts if isinstance(s.value, ast.Tuple) else None
                        if vals is None or len(vals) != len(tgt.elts):
                            raise NotImplementedError(
                                "Tuple assignment needs a tuple of the same length on the right"
                            )
                        pairs = list(zip([self._target_name(t) for t in tgt.elts], vals))
                    else:
                        pairs = [(self._target_name(tgt), s.value)]
                    computed = [(name, self._is_complex(v),
                                 self._visit_cplx(v) if self._is_complex(v) else self.visit(v))
                                for name, v in pairs]
                    for name, is_cplx, rhs in computed:
                        if is_cplx:
                            self._complex.add(name)
                            env[f"{name}_re"] = f"({rhs[0]})"
                            env[f"{name}_im"] = f"({rhs[1]})"
                        else:
                            env[name] = f"({rhs})"
                elif isinstance(s, ast.If):
                    # Recursively lower nested if/elif/else into expressions
                    for k, v in self._eval_if_expr(s).items():
                        env[k] = f"({v})"
                elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
                    self._list_method(s.value, env)
                elif isinstance(s, ast.Pass):
                    pass
                else:
                    raise NotImplementedError(
                        f"Unsupported statement type in combinational block: {type(s).__name__}"
                    )
        finally:
            self._name_env_stack.pop()
        return env

    def _eval_if_expr(self, node: ast.If) -> Dict[str, str]:
        """
        Turn an if/elif/else *expression tree* into a mapping
        var -> nested ternary expression (as a string).

        Each branch body is evaluated via _eval_stmt_block, so nested if's are
        fully supported.

        Requirements:
          * final 'else' branch is present (no latches)
          * all branches assign to the same set of variables
        """
        branches: List[tuple[str, Dict[str, str]]] = []

        cur = node
        while True:
            cond = self.visit(cur.test)
            then_env = self._eval_stmt_block(cur.body)
            branches.append((cond, then_env))

            # Chained elif?
            if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
                cur = cur.orelse[0]
                continue
            else_env = self._eval_stmt_block(cur.orelse) if cur.orelse else {}
            break

        # Variables in first-assigned order across all branches
        var_order: List[str] = []
        for env in [m for _, m in branches] + [else_env]:
            for k in env:
                if k not in var_order:
                    var_order.append(k)

        result_env: Dict[str, str] = {}
        for lhs in var_order:
            def value_in(env: Dict[str, str]) -> str:
                if lhs in env:
                    return env[lhs]
                # Not assigned in this branch: keeps its value from before the if.
                # With none, Python can only read it after a branch that assigns
                # it (else NameError), so that branch's value serves (no latch)
                if not self._is_defined(lhs):
                    return next(m[lhs] for m in [e for _, e in branches] + [else_env]
                                if lhs in m)
                return self._resolve_name(lhs)

            # Build nested ternary from bottom (else) up
            rhs = value_in(else_env)
            for cond, then_map in reversed(branches):
                rhs = f"({cond}) ? ({value_in(then_map)}) : ({rhs})"
            result_env[lhs] = rhs

        return result_env

    @_at_line
    def _visit_if(self, node: ast.If):
        """
        Lower a restricted if/elif/else (with possible nested ifs) into
        per-variable nested ternary expressions.
        """
        for lhs, rhs in self._eval_if_expr(node).items():
            self._assign_var(lhs, rhs)

    # ---- default ----
    def generic_visit(self, node):
        raise NotImplementedError(f"Unsupported AST node: {type(node).__name__}")


# ------------------------------------------------------------
# Helper: emit internal wire/reg declarations
# ------------------------------------------------------------
def emit_internal_declarations(module_ir: ModuleIR,
                               attr: Dict[str, Dict[str, Any]]) -> str:
    """
    Emit Verilog declarations for internal nets:
      - Anything that appears as an assignment LHS
      - BUT is not a port name

    Uses 'attr' for width/signed/type (wire/reg).
    """
    # 1) All port names (inputs + outputs)
    port_names = {p.name for p in (module_ir.inputs + module_ir.outputs)}

    # 2) All LHS names from assignments
    lhs_names = {a.left for a in module_ir.assigns}

    # 3) Internal = assigned but not a port
    internal_names = sorted(lhs_names - port_names)

    lines: List[str] = []

    for name in internal_names:
        meta = attr.get(name, {'width': 1, 'signed': 0, 'type': 'wire'})
        width = int(meta.get('width', 1))
        signed = bool(meta.get('signed', 0))
        vtype = str(meta.get('type', 'wire'))  # 'wire' or 'reg'

        if width <= 1:
            range_str = ""
        else:
            range_str = f"[{width-1}:0] "

        signed_str = "signed " if signed else ""

        if vtype == 'reg':
            decl_kw = 'reg'
        else:
            decl_kw = 'wire'

        lines.append(f"{decl_kw} {signed_str}{range_str}{name};")

    return "\n".join(lines)


class _Desugar(ast.NodeTransformer):
    """Rewrite `x op= e` as `x = x op e`."""

    def visit_AugAssign(self, node: ast.AugAssign):
        tgt = node.target
        if isinstance(tgt, ast.Name):
            load = ast.Name(id=tgt.id, ctx=ast.Load())
            store = ast.Name(id=tgt.id, ctx=ast.Store())
        elif isinstance(tgt, ast.Subscript):
            # xs[i] op= v -> xs[i] = xs[i] op v (index expressions have no side effects)
            load = ast.Subscript(value=tgt.value, slice=tgt.slice, ctx=ast.Load())
            store = ast.Subscript(value=tgt.value, slice=tgt.slice, ctx=ast.Store())
        else:
            raise NotImplementedError("Augmented assignment target must be a name or xs[i]")
        value = ast.BinOp(left=load, op=node.op, right=node.value)
        return ast.copy_location(ast.Assign(targets=[store], value=value), node)


# Small builders for complex parts that keep obvious zeros out of the Verilog
def _neg(x: str) -> str:
    return "0" if x == "0" else f"-({x})"


def _add(x: str, y: str) -> str:
    if x == "0":
        return y
    if y == "0":
        return x
    return f"({x}) + ({y})"


def _sub(x: str, y: str) -> str:
    if y == "0":
        return x
    if x == "0":
        return _neg(y)
    return f"({x}) - ({y})"


def _mul(x: str, y: str) -> str:
    if x == "0" or y == "0":
        return "0"
    return f"({x}) * ({y})"
