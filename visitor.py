# visitor.py
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

def normalize_attr(attr: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in attr.items():
        if not isinstance(v, dict):
            out[k] = {'signed': 0, 'width': 1, 'type': 'wire'}
            continue
        out[k] = {
            'signed': int(v.get('signed', 0)),
            'width': int(v.get('width', 1)),
            'type':  str(v.get('type', 'wire')),
        }
    return out

BINOP_TOKENS = {
    ast.Add: " + ", ast.Sub: " - ", ast.Mult: " * ", ast.Div: " / ",
    ast.BitAnd: " & ", ast.BitOr: " | ", ast.BitXor: " ^ ",
    ast.LShift: " << ", ast.RShift: " >> ",
}

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

    def _any_reg(self) -> bool:
        for meta in self.attr.values():
            if isinstance(meta, dict) and meta.get('type') == 'reg':
                return True
        return False

    # ---------- node visitors ----------
    def visit_Module(self, node: ast.Module):
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ModuleIR:
        self.top_name = node.name

        # inputs (from args)
        self.visit(node.args)

        # body: collect assigns and discover outputs via Return
        for item in node.body:
            if isinstance(item, ast.Assign):
                self._visit_assign(item)
            elif isinstance(item, ast.Return):
                outs = self.visit(item)  # list[str]
                self.output_args_list = outs
                for name in outs:
                    meta = self._attrs(name)
                    self.output_args_width_list.append(meta['width'])
                    self.output_bit_count += meta['width']

        # clk if any reg present
        self.has_clk = self._any_reg()

        # Build IR ports
        inputs: List[Port] = []
        for name, width in zip(self.input_args_list, self.input_args_width_list):
            inputs.append(Port(name=name, width=width, signed=bool(self._attrs(name)['signed']), kind='input'))
        if self.has_clk:
            # add explicit clk input
            inputs.append(Port(name='clk', width=1, signed=False, kind='clk'))

        outputs: List[Port] = []
        for name, width in zip(self.output_args_list, self.output_args_width_list):
            kind = 'reg' if self._attrs(name)['type'] == 'reg' else 'output'
            outputs.append(Port(name=name, width=width, signed=bool(self._attrs(name)['signed']), kind=kind))

        return ModuleIR(
            name=self.top_name,
            inputs=inputs,
            outputs=outputs,
            assigns=self.assigns,
            has_clk=self.has_clk
        )

    def visit_arguments(self, node: ast.arguments):
        values = [self.visit(a) for a in node.args]  # returns name strings
        # Each arg corresponds to a port meta
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

    # ---- expressions to strings (for RHS rendering) ----
    def visit_Name(self, node: ast.Name):
        return node.id

    # literals
    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool):
            return "1'b1" if node.value else "1'b0"
        if isinstance(node.value, int):
            return str(node.value)
        if isinstance(node.value, str):
            return f'"{node.value}"'
        return str(node.value)

    def visit_Num(self, node: ast.Num):  # py<3.8
        return str(node.n)

    # unary
    def visit_UnaryOp(self, node: ast.UnaryOp):
        op = self.visit(node.op)
        return f"{op}{self.visit(node.operand)}"
    def visit_USub(self, node): return "-"
    def visit_UAdd(self, node): return "+"
    def visit_Invert(self, node): return "~"

    # binary
    def visit_BinOp(self, node: ast.BinOp):
        op = BINOP_TOKENS[type(node.op)]
        return f"({self.visit(node.left)}){op}({self.visit(node.right)})"

    # assignments
    def _visit_assign(self, node: ast.Assign):
        rhs = self.visit(node.value)
        for tgt in node.targets:
            left = self.visit(tgt)
            meta = self._attrs(left)
            self.assigns.append(Assignment(left=left, right=rhs, is_reg=(meta.get('type') == 'reg')))

    # default
    def generic_visit(self, node):
        raise NotImplementedError(f"Unsupported AST node: {type(node).__name__}")
