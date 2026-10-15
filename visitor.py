import ast
import functools
import re
from typing import Dict, Any, List
from ast_ir import ModuleIR, Port, Assignment, resolve_meta
from semantics import Unsupported, analyze, parse

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
    return out


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
        self._ssa: Dict[str, str] = {}
        self._ssa_count: Dict[str, int] = {}

    # ---------- helpers ----------
    def _attrs(self, name: str) -> Dict[str, Any]:
        return self.attr.get(name, {'signed': 0, 'width': 1, 'type': 'wire'})

    def _clone_attr(self, src: str, dst: str):
        meta = dict(self._attrs(src))
        self.attr[dst] = meta

    def _complex_parts(self, name: str) -> tuple[str, str]:
        return f"{name}_re", f"{name}_im"

    def _ensure_complex_part_attrs(self, name: str):
        re_name, im_name = self._complex_parts(name)
        if re_name not in self.attr:
            self._clone_attr(name, re_name)
        if im_name not in self.attr:
            self._clone_attr(name, im_name)

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
        except Unsupported:
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
        if var in self._declared:
            self.attr[name] = dict(self.attr[var])
        else:
            meta = self._infer_meta(rhs)
            self.attr[name] = {"signed": meta["signed"], "width": meta["width"], "type": "wire"}
        if is_reg is None:
            is_reg = self.attr[name].get("type") == "reg"
        self.assigns.append(Assignment(left=name, right=rhs, is_reg=is_reg))
        return name

    @staticmethod
    def _target_name(tgt: ast.expr) -> str:
        if not isinstance(tgt, ast.Name):
            raise NotImplementedError(f"Assignment target must be a name, got {ast.unparse(tgt)}")
        return tgt.id

    def _finalize_output(self, var: str) -> str:
        """Expose the final version of var on an output port named var."""
        current = self._resolve_name(var)
        if var in self.input_args_list:
            raise NotImplementedError(
                f"Returning argument '{var}' is not supported; assign it to a new variable first."
            )
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
        raise NotImplementedError(f"Only integer literals are allowed in range(): {ast.dump(node)}")

    def _parse_range(self, iter_node: ast.expr) -> tuple[int, int, int]:
        """
        Parse a range(...) call into (start, stop, step) integers.

        Only supports range with 1–3 integer-literal arguments.
        """
        if not isinstance(iter_node, ast.Call):
            raise NotImplementedError("for-loops must iterate over range(...).")
        func = iter_node.func
        if not (isinstance(func, ast.Name) and func.id == "range"):
            raise NotImplementedError("for-loops must iterate over range(...).")

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
                    sh = rw
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
            # skip docstrings or bare expression statements
            if isinstance(item, ast.Expr):
                if (
                    isinstance(getattr(item, "value", None), ast.Constant)
                    and isinstance(item.value.value, str)
                ):
                    # docstring
                    continue
                # ignore other bare expressions for now
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
                    if isinstance(v, ast.Name):
                        outs.append(self._finalize_output(v.id))
                    else:
                        # Returned expression: give it an output named ret<i>
                        name = f"ret{i}" if len(vals) > 1 else "ret"
                        if self._is_defined(name):
                            raise NotImplementedError(f"Output name '{name}' is already used")
                        outs.append(self._assign_var(name, self.visit(v)))
                self.output_args_list = outs
                for name in outs:
                    meta = self._attrs(name)
                    self.output_args_width_list.append(meta['width'])
                    self.output_bit_count += meta['width']
            else:
                self.generic_visit(item)

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
        for name in values:
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
        if isinstance(node.func, ast.Name) and not node.keywords:
            fn = node.func.id
            args = [self.visit(a) for a in node.args]
            if fn == "abs" and len(args) == 1:
                x = args[0]
                return f"((({x}) < 0) ? (-({x})) : ({x}))"
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
    def visit_For(self, node: ast.For):
        """
        Unroll `for <var> in range(...)`. Assignments in the body create new
        versions of their variables, so accumulators (acc = acc + ...) chain
        naturally across iterations.
        """
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("Only 'for <var> in range(...)' loops are supported.")
        loop_var = node.target.id

        start, stop, step = self._parse_range(node.iter)

        for val in range(start, stop, step):
            env = {loop_var: str(val)}
            self._name_env_stack.append(env)
            try:
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        self._visit_assign(stmt)
                    elif isinstance(stmt, ast.If):
                        self._visit_if(stmt)
                    elif isinstance(stmt, ast.For):
                        self.visit_For(stmt)
                    else:
                        self.generic_visit(stmt)
            finally:
                self._name_env_stack.pop()

    # ---- assignments ----
    def _visit_assign(self, node: ast.Assign):
        # Special-case complex constructor:
        #   z = complex(a, b)  ->  z_re = a; z_im = b
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "complex"
        ):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise NotImplementedError("complex(...) assignment supports a single name target.")
            if len(node.value.args) != 2:
                raise NotImplementedError("complex(...) requires exactly 2 arguments.")

            base = node.targets[0].id
            self._ensure_complex_part_attrs(base)
            re_name, im_name = self._complex_parts(base)
            re_rhs = self.visit(node.value.args[0])
            im_rhs = self.visit(node.value.args[1])
            re_meta = self._attrs(re_name)
            im_meta = self._attrs(im_name)

            self.assigns.append(
                Assignment(
                    left=re_name,
                    right=re_rhs,
                    is_reg=(re_meta.get('type') == 'reg')
                )
            )
            self.assigns.append(
                Assignment(
                    left=im_name,
                    right=im_rhs,
                    is_reg=(im_meta.get('type') == 'reg')
                )
            )
            return

        # Special-case absolute value of a complex temporary:
        #   mag = abs(z)  ->  mag = $sqrt((z_re*z_re) + (z_im*z_im))
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "abs"
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Name)
            and f"{node.value.args[0].id}_re" in self.attr
        ):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise NotImplementedError("abs(...) assignment supports a single name target.")
            if len(node.value.args) != 1 or not isinstance(node.value.args[0], ast.Name):
                raise NotImplementedError("abs(...) currently supports a single complex variable argument.")

            z_name = node.value.args[0].id
            re_name, im_name = self._complex_parts(z_name)
            rhs = f"$sqrt(({re_name})*({re_name}) + ({im_name})*({im_name}))"
            left = node.targets[0].id
            meta = self._attrs(left)
            self.assigns.append(
                Assignment(
                    left=left,
                    right=rhs,
                    is_reg=(meta.get('type') == 'reg')
                )
            )
            return

        for tgt in node.targets:
            if isinstance(tgt, ast.Tuple):
                vals = node.value.elts if isinstance(node.value, ast.Tuple) else None
                if vals is None or len(vals) != len(tgt.elts):
                    raise NotImplementedError(
                        "Tuple assignment needs a tuple of the same length on the right"
                    )
                # All right-hand sides are read before anything is assigned
                rhss = [self.visit(v) for v in vals]
                for t, rhs in zip(tgt.elts, rhss):
                    self._assign_var(self._target_name(t), rhs)
            else:
                self._assign_var(self._target_name(tgt), self.visit(node.value))

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
                    if isinstance(tgt, ast.Tuple):
                        vals = s.value.elts if isinstance(s.value, ast.Tuple) else None
                        if vals is None or len(vals) != len(tgt.elts):
                            raise NotImplementedError(
                                "Tuple assignment needs a tuple of the same length on the right"
                            )
                        rhss = [self.visit(v) for v in vals]
                        for t, rhs in zip(tgt.elts, rhss):
                            env[self._target_name(t)] = f"({rhs})"
                    else:
                        env[self._target_name(tgt)] = f"({self.visit(s.value)})"
                elif isinstance(s, ast.If):
                    # Recursively lower nested if/elif/else into expressions
                    for k, v in self._eval_if_expr(s).items():
                        env[k] = f"({v})"
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
                # Not assigned in this branch: keeps its value from before the if
                if not self._is_defined(lhs):
                    raise NotImplementedError(
                        f"'{lhs}' is assigned in only some branches of an if and has no "
                        "earlier value (that would infer a latch)."
                    )
                return self._resolve_name(lhs)

            # Build nested ternary from bottom (else) up
            rhs = value_in(else_env)
            for cond, then_map in reversed(branches):
                rhs = f"({cond}) ? ({value_in(then_map)}) : ({rhs})"
            result_env[lhs] = rhs

        return result_env

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
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("Augmented assignment target must be a name")
        value = ast.BinOp(
            left=ast.Name(id=node.target.id, ctx=ast.Load()), op=node.op, right=node.value
        )
        new = ast.Assign(targets=[ast.Name(id=node.target.id, ctx=ast.Store())], value=value)
        return ast.copy_location(new, node)
