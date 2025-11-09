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

    # ---------- helpers ----------
    def _attrs(self, name: str) -> Dict[str, Any]:
        return self.attr.get(name, {'signed': 0, 'width': 1, 'type': 'wire'})

    def _any_reg_outputs(self) -> bool:
        """Detect if any output port is a reg (requires clk)."""
        for name in self.output_args_list:
            meta = self._attrs(name)
            if isinstance(meta, dict) and meta.get('type') == 'reg':
                return True
        return False

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
        return node.id

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
        Turn an if/elif/else *expression tree* into a mapping var -> nested
        ternary expression (as a string).

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

        High-level pattern:

            if cond0:
                ...
            elif cond1:
                ...
            else:
                ...

        where each branch body is a *combinational block* of Assign / nested If.
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
