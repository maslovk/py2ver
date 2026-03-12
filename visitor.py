import ast
import functools
from typing import Dict, Any, List
from ast_ir import ModuleIR, Port, Assignment

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
    # Support both / and // in Python, both mapped to Verilog integer division
    ast.Div: " / ",
    ast.FloorDiv: " / ",
    ast.BitAnd: " & ",
    ast.BitOr: " | ",
    ast.BitXor: " ^ ",
    ast.LShift: " << ",
    ast.RShift: " >> ",
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
        return name

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

    # ---------- visitors ----------
    def visit_Module(self, node: ast.Module):
        if not node.body:
            raise NotImplementedError("Empty module.")
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ModuleIR:
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
            elif isinstance(item, ast.Return):
                if seen_return:
                    raise NotImplementedError("Multiple return statements are not supported.")
                seen_return = True
                outs = self.visit(item)
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
        return f"{op}{self.visit(node.operand)}"

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
        return f"({left}{token}{right})"

    def visit_BoolOp(self, node: ast.BoolOp):
        """Support 'and' / 'or' conditions."""
        if isinstance(node.op, ast.And):
            op_str = " && "
        elif isinstance(node.op, ast.Or):
            op_str = " || "
        else:
            raise NotImplementedError(f"Unsupported boolean operator: {type(node.op).__name__}")
        parts = [self.visit(v) for v in node.values]
        return "(" + op_str.join(parts) + ")"

    # ---- for-loops ----
    def visit_For(self, node: ast.For):
        """
        Handle two cases:

        (A) Accumulator pattern (SSA-style):

            acc = <init>      # already seen earlier in self.assigns
            for i in range(...):
                acc = acc + f(i, ...)
            ...

        -> emit acc_0, acc_1, ..., acc_N and final acc = acc_N.

        (B) Fallback: generic unrolling (no self-updates),
            as before, for "simple" loops.
        """
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("Only 'for <var> in range(...)' loops are supported.")
        loop_var = node.target.id

        start, stop, step = self._parse_range(node.iter)

        # --- Try to detect accumulator pattern (A) ---

        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Assign)
            and len(node.body[0].targets) == 1
            and isinstance(node.body[0].targets[0], ast.Name)
        ):
            assign_node = node.body[0]
            acc_name = assign_node.targets[0].id

            # RHS must use acc_name (acc = acc + ...)
            if self._expr_uses_name(assign_node.value, acc_name):
                self._lower_accumulating_for(node, loop_var, acc_name, start, stop, step)
                return

        # --- Otherwise, fallback to plain unrolling (B) ---

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

    def _lower_accumulating_for(
        self,
        node: ast.For,
        loop_var: str,
        acc_name: str,
        start: int,
        stop: int,
        step: int
    ):
        """
        Lower accumulator-style loop into a chain:

            acc_0 = <init>
            acc_1 = f(acc_0, ...)
            ...
            acc_N = f(acc_{N-1}, ...)
            acc   = acc_N
        """
        assign_node = node.body[0]

        # Find initial assignment to acc_name in self.assigns (last one wins)
        init_idx = None
        for idx in range(len(self.assigns) - 1, -1, -1):
            if self.assigns[idx].left == acc_name:
                init_idx = idx
                break

        if init_idx is not None:
            init_assign = self.assigns.pop(init_idx)
            init_rhs = init_assign.right
            is_reg = init_assign.is_reg
        else:
            # No explicit init seen; use acc itself as initial value.
            init_rhs = acc_name
            is_reg = (self._attrs(acc_name).get('type') == 'reg')

        # Declare acc_0
        prev_name = f"{acc_name}_0"
        self._clone_attr(acc_name, prev_name)
        self.assigns.append(
            Assignment(
                left=prev_name,
                right=init_rhs,
                is_reg=is_reg
            )
        )

        curr_name = prev_name
        iter_values = list(range(start, stop, step))

        for idx, val in enumerate(iter_values):
            next_name = f"{acc_name}_{idx + 1}"
            self._clone_attr(acc_name, next_name)

            # For this iteration, acc -> curr_name, loop_var -> val
            env = {
                loop_var: str(val),
                acc_name: curr_name,
            }
            self._name_env_stack.append(env)
            try:
                rhs_expr = self.visit(assign_node.value)
            finally:
                self._name_env_stack.pop()

            self.assigns.append(
                Assignment(
                    left=next_name,
                    right=rhs_expr,
                    is_reg=is_reg
                )
            )

            curr_name = next_name

        # Final acc assignment: acc = acc_N (or acc_0 if loop empty)
        final_rhs = curr_name
        self.assigns.append(
            Assignment(
                left=acc_name,
                right=final_rhs,
                is_reg=is_reg
            )
        )

    # ---- assignments ----
    def _visit_assign(self, node: ast.Assign):
        rhs = self.visit(node.value)
        for tgt in node.targets:
            left = self.visit(tgt)
            meta = self._attrs(left)
            self.assigns.append(
                Assignment(
                    left=left,
                    right=rhs,
                    is_reg=(meta.get('type') == 'reg')
                )
            )

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
        for s in stmts:
            if isinstance(s, ast.Assign):
                if len(s.targets) != 1:
                    raise NotImplementedError(
                        "Multiple assignment targets are not supported."
                    )
                lhs = self.visit(s.targets[0])
                rhs = self.visit(s.value)
                env[lhs] = rhs
            elif isinstance(s, ast.If):
                # Recursively lower nested if/elif/else into expressions
                nested_env = self._eval_if_expr(s)
                # Later statements override earlier ones
                env.update(nested_env)
            else:
                raise NotImplementedError(
                    f"Unsupported statement type in combinational block: {type(s).__name__}"
                )
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
        else_env: Dict[str, str] | None = None

        cur = node
        while True:
            if not cur.orelse:
                raise NotImplementedError(
                    "if/elif without final else is not supported (would infer a latch)."
                )

            cond = self.visit(cur.test)
            then_env = self._eval_stmt_block(cur.body)
            branches.append((cond, then_env))

            # Chained elif?
            if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
                cur = cur.orelse[0]
                continue
            else:
                # Final else
                else_env = self._eval_stmt_block(cur.orelse)
                break

        if else_env is None:
            raise NotImplementedError("if/elif chain ended without a final else branch.")

        # All branches must assign to the same set of variables
        key_sets = [set(env.keys()) for _, env in branches] + [set(else_env.keys())]
        first_keys = key_sets[0]
        for ks in key_sets[1:]:
            if ks != first_keys:
                raise NotImplementedError(
                    "All branches of if/elif/else must assign to the same set of variables "
                    f"(got {[sorted(ks) for ks in key_sets]})."
                )

        # Deterministic variable order: that of the first branch
        var_order = list(branches[0][1].keys())

        result_env: Dict[str, str] = {}
        for lhs in var_order:
            # Build nested ternary from bottom (else) up
            rhs = else_env[lhs]
            for cond, then_map in reversed(branches):
                rhs = f"({cond}) ? ({then_map[lhs]}) : ({rhs})"
            result_env[lhs] = rhs

        return result_env

    def _visit_if(self, node: ast.If):
        """
        Lower a restricted if/elif/else (with possible nested ifs) into
        per-variable nested ternary expressions.
        """
        env = self._eval_if_expr(node)
        for lhs, rhs in env.items():
            meta = self._attrs(lhs)
            self.assigns.append(
                Assignment(
                    left=lhs,
                    right=rhs,
                    is_reg=(meta.get('type') == 'reg')
                )
            )

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
