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

    def _visit_if(self, node: ast.If):
        """
        Lower a restricted if/else with multiple assignments per branch into
        per-variable ternary assignments.

        Supported pattern:

            if cond:
                a = expr_a_then
                b = expr_b_then
                ...
            else:
                a = expr_a_else
                b = expr_b_else
                ...

        Requirements:
          * else branch is present (no latches)
          * all statements in both branches are simple Assign
          * each Assign has exactly one target
          * both branches assign to the same set of variables
        """
        if not node.orelse:
            raise NotImplementedError("if without else is not supported (latch).")

        then_stmts = node.body
        else_stmts = node.orelse

        # Enforce simple assignment-only branches
        for stmt_list in (then_stmts, else_stmts):
            for s in stmt_list:
                if not isinstance(s, ast.Assign):
                    raise NotImplementedError(
                        "if/else branches must contain only simple assignments."
                    )
                if len(s.targets) != 1:
                    raise NotImplementedError(
                        "Multiple assignment targets in if/else are not supported."
                    )

        # Build mapping: var -> RHS for each branch
        then_map: Dict[str, str] = {}
        else_map: Dict[str, str] = {}

        for s in then_stmts:
            lhs = self.visit(s.targets[0])
            rhs = self.visit(s.value)
            if lhs in then_map:
                raise NotImplementedError(
                    f"Variable '{lhs}' assigned multiple times in 'then' branch."
                )
            then_map[lhs] = rhs

        for s in else_stmts:
            lhs = self.visit(s.targets[0])
            rhs = self.visit(s.value)
            if lhs in else_map:
                raise NotImplementedError(
                    f"Variable '{lhs}' assigned multiple times in 'else' branch."
                )
            else_map[lhs] = rhs

        if set(then_map.keys()) != set(else_map.keys()):
            raise NotImplementedError(
                "If/else branches must assign to the same set of variables "
                f"(then: {sorted(then_map.keys())}, else: {sorted(else_map.keys())})."
            )

        cond = self.visit(node.test)

        # Preserve some determinism by iterating in the 'then' branch key order
        for lhs in then_map.keys():
            rhs_then = then_map[lhs]
            rhs_else = else_map[lhs]
            rhs = f"({cond}) ? ({rhs_then}) : ({rhs_else})"

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
