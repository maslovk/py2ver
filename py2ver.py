import inspect
import ast
from jinja2 import Environment, FileSystemLoader
import os
import functools
import time

DEFAULT_TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/templates/'

# -------------------------------------------------------------------------------
try:
    import textwrap
    indent = textwrap.indent
except:
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


def indent_multiline_assign(text):
    ret = []
    texts = text.split('\n')
    if len(texts) <= 1:
        return text
    try:
        p = texts[0].index('=')
    except:
        return text
    ret.append(texts[0])
    ret.append('\n')
    ret.append(indent('\n'.join(texts[1:]), ' ' * (p + 2)))
    return ''.join(ret)

class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, attr, indentsize=2):
        self.attr =attr
        self.env = Environment(loader=FileSystemLoader(DEFAULT_TEMPLATE_DIR))
        self.indent = functools.partial(indent, prefix=' ' * indentsize)
        self.template_cache = {}

    def check_attributes(self, name):
            if name in self.attr.keys():
                result = self.attr[name]
            else:
                result = [{'signed': 0, 'width': 1}]

            return result

    def get_template(self, filename):
        if filename in self.template_cache:
            return self.template_cache[filename]

        template = self.env.get_template(filename)
        self.template_cache[filename] = template
        return template

    def visit_arg(self, node):
        template = self.get_template('port.txt')
        input_template = self.get_template('input.txt')
        attr = self.check_attributes(node.arg)
        input_template_dict = {
            'name': node.arg,
            'width': attr['width'],
            'signed': attr['signed'],
            'dimensions': '',
        }
        template_dict = {
            'name': node.arg,
        }
        rslt_port = template.render(template_dict)
        rslt_input = input_template.render(input_template_dict)
        return {"port" : rslt_port,"input" : rslt_input}

    def visit_arguments(self, node):
        template = self.get_template('portlist.txt')
        values  = [self.visit(arg) for arg in node.args]
        ports = [val["port"] for val in values]
        inputs = [val["input"] for val in values]
        template_dict = {
            'ports': ports,
            'len_ports': len(ports),
        }
        rslt_portlist = template.render(template_dict)
        rslt_inputlist = ''.join(inputs)
        return {"port_list" : rslt_portlist,"input_list" : rslt_inputlist}

    def visit_Return(self, node):
        output_template = self.get_template('output.txt')
        attr = self.check_attributes(node.value.id)
        output_template_dict = {
            'name': node.value.id,
            'width': attr['width'],
            'signed': attr['signed'],
            'dimensions': '',
        }
        rslt_output = output_template.render(output_template_dict)
        return rslt_output


    def visit_Name(self,node):
        return node.id

    def visit_Add(self, node):
        return " + "

    def visit_Sub(self, node):
        return " - "

    def visit_Mult(self, node):
        return " * "

    def visit_Div(self, node):
        return " / "

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = self.visit(node.op)
        template = self.get_template("operator.txt")
        template_dict = {
            'left': left,
            'right': right,
            'op': op,
        }

        rslt = template.render(template_dict)
        return rslt

    def visit_Assign(self, node):
        op = self.visit(node.value)
        target = ''
        for item in node.targets:
            target = self.visit(item)

        template = self.get_template("assign.txt")
        template_dict = {
            'left': target,
            'right': op,
        }
        rslt = template.render(template_dict)
        rslt = indent_multiline_assign(rslt)
        return rslt

    def visit_Module(self, node):
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node):
        print(f"Found function: {node.name}")
        template = self.get_template("moduledef.txt")
        paramlist = ''
        output_list = ''
        portlist = ''
        input_list = ''
        assign_list = ''
        for item in node.body:
            if isinstance(item, ast.Return):
                output_list = self.indent(self.visit(item))
            else :
                if isinstance(item, ast.Assign):
                    assign_list += (self.indent(self.visit(item)) + "\n")
        if node.args is not None:
            values = self.visit(node.args)
            portlist = self.indent(values['port_list'])
            input_list = self.indent(values['input_list'])

        items = [input_list, output_list, assign_list]

        template_dict = {
            'modulename': node.name,
            'paramlist': paramlist,
            'portlist': portlist + ','+output_list,
            'items': [self.indent(item) for item in items] if items else (),
        }
        rslt = template.render(template_dict)
        #self.generic_visit(node)
        return rslt

class Py2ver:
    def __init__(self, func, attr):
        source_foo = inspect.getsource(func)
        tree = ast.parse(source_foo)
        result = FunctionVisitor(attr).visit(tree)

        with open("main.v", "w") as text_file:
            text_file.write(result)