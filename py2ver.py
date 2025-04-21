import inspect
import ast
from jinja2 import Environment, FileSystemLoader
import os
import functools
import time
import  tb_runner
import pickle

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
        self.top_name = ''
        self.input_args_list =[]
        self.output_args_list =[]

    def check_attributes(self, name):
            if name in self.attr.keys():
                result = self.attr[name]
            else:
                result = {'signed': 0, 'width': 1}

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
        return {"port_list" : ports,"port_list_str" : rslt_portlist,"input_list" : rslt_inputlist}

    def visit_Return(self, node):
        out = self.visit(node.value)
        return out

    def visit_Tuple(self, node):
        tup = [self.visit(elts) for elts in node.elts]
        return tup


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
        self.top_name = node.name
        template = self.get_template("moduledef.txt")
        paramlist = ''
        output_list = ''
        portlist = ''
        out_portlist_str = ''
        input_list = ''
        assign_list = ''
        output_args = ''
        for item in node.body:
            if isinstance(item, ast.Return):
                items = self.visit(item)
                self.output_args_list = items
                for it in items:
                    output_template = self.get_template('output.txt')
                    attr = self.check_attributes(it)
                    output_template_dict = {
                        'name': it,
                        'width': attr['width'],
                        'signed': attr['signed'],
                        'dimensions': '',
                    }
                    output_list += output_template.render(output_template_dict) + "\n"

                template_pl = self.get_template('portlist.txt')
                template_pl_dict = {
                       'ports': items,
                       'len_ports': len(items),
                }
                out_portlist_str = template_pl.render(template_pl_dict)
            else :
                if isinstance(item, ast.Assign):
                    assign_list += (self.indent(self.visit(item)) + "\n")
        if node.args is not None:
            values = self.visit(node.args)
            portlist = self.indent(values['port_list_str'])
            self.input_args_list = values['port_list']
            input_list = self.indent(values['input_list'])


        #Dump waves
        template_dw = self.get_template('dumpwaves.txt')
        template_dw_dict = {
            'top_name': self.top_name
        }

        output_dw = template_dw.render(template_dw_dict)


        items = [input_list, output_list, assign_list, output_dw]

        template_dict = {
            'modulename': node.name,
            'paramlist': paramlist,
            'portlist': portlist + ',\n' +out_portlist_str + ',' +output_args,
            'items': [self.indent(item) for item in items] if items else (),
        }
        rslt = template.render(template_dict)
        #self.generic_visit(node)
        return rslt

    def get_top_name(self):
        return self.top_name

    def get_input_args_list(self):
        return self.input_args_list

    def get_output_args_list(self):
        return self.output_args_list

class Py2ver:
    t_name = ''
    def __init__(self, func, attr):
        source_foo = inspect.getsource(func)
        tree = ast.parse(source_foo)
        f_visitor = FunctionVisitor(attr)
        result = f_visitor.visit(tree)
        self.t_name = f_visitor.get_top_name()
        self.input_args_list = f_visitor.get_input_args_list()
        self.output_args_list = f_visitor.get_output_args_list()
        self.env = Environment(loader=FileSystemLoader(DEFAULT_TEMPLATE_DIR))
        self.template_cache = {}

        if not os.path.isdir("hdl"):
            os.mkdir("hdl")

        f = open("hdl/main.v", "w+")
        f.write(result)
        f.close()

    def get_template(self, filename):
        if filename in self.template_cache:
            return self.template_cache[filename]

        template = self.env.get_template(filename)
        self.template_cache[filename] = template
        return template

    def gen_tb(self, *in_arg):
        #Check if we have values for all arguments
        key_val = []
        for i in range(len(self.input_args_list)):
            if in_arg[i] is not None:
                key_val.append((self.input_args_list[i],in_arg[i]))
            else:
                key_val.append((self.input_args_list[i],0))
        template = self.get_template("tb.txt")
        template_dict = {
            'in_args': key_val,
            'out_args': self.output_args_list
        }
        tb_out = template.render(template_dict)
        with open("tb.py", "w+") as f:
            f.write(tb_out)



    def tb(self):
        return self.tb_fun

    def tb_fun(self, *in_arg):
        print("I was called with", len(in_arg), "arguments:", in_arg)
        if os.path.isfile('results/results.pickle'):
            os.remove('results/results.pickle')
        self.gen_tb(*in_arg)
        tb_runner.test_runner(self.t_name)
        if os.path.isfile('results/results.pickle'):
            with open('results/results.pickle', 'rb') as handle:
                p = pickle.load(handle)
                b = ast.literal_eval(p)
        else:
            b = (0,) * len(self.output_args_list)
        return b


