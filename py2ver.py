import inspect
import ast
from jinja2 import Environment, FileSystemLoader
import os
import functools
import tb_runner
import pickle
import subprocess
import shutil

DEFAULT_TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/templates/'

# -------------------------------------------------------------------------------
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


def indent_multiline_assign(text):
    lines = text.split('\n')
    if len(lines) <= 1:
        return text
    first = lines[0]
    if '=' not in first:
        return text
    p = first.index('=')
    return first + '\n' + indent('\n'.join(lines[1:]), ' ' * (p + 2))


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, attr, indentsize=2):
        self.attr = attr
        self.env = Environment(loader=FileSystemLoader(DEFAULT_TEMPLATE_DIR))
        self.indent = functools.partial(indent, prefix=' ' * indentsize)
        self.template_cache = {}
        self.top_name = ''
        self.input_args_list = []
        self.output_args_list = []
        self.input_bit_count = 0
        self.output_bit_count = 0
        self.input_args_width_list = []
        self.output_args_width_list = []

    # ---------- attribute helpers ----------
    def check_regs_present(self):
        for name, meta in self.attr.items():
            if isinstance(meta, dict) and meta.get('type') == 'reg':
                return True
        return False

    def check_attributes(self, name):
        meta = self.attr.get(name, None)
        if not isinstance(meta, dict):
            return {'signed': 0, 'width': 1, 'type': 'wire'}
        return {
            'signed': int(meta.get('signed', 0)),
            'width': int(meta.get('width', 1)),
            'type':  str(meta.get('type', 'wire')),
        }

    # ---------- template caching ----------
    def get_template(self, filename):
        if filename in self.template_cache:
            return self.template_cache[filename]
        template = self.env.get_template(filename)
        self.template_cache[filename] = template
        return template

    # ---------- arguments / ports ----------
    def visit_arg(self, node):
        template = self.get_template('port.txt')
        input_template = self.get_template('input.txt')
        attr = self.check_attributes(node.arg)
        self.input_bit_count += attr['width']
        input_template_dict = {
            'name': node.arg,
            'width': attr['width'],
            'signed': attr['signed'],
            'dimensions': '',
        }
        template_dict = {'name': node.arg}
        rslt_port = template.render(template_dict)
        rslt_input = input_template.render(input_template_dict)
        return {"port": rslt_port, "input": rslt_input, "width": attr['width']}

    def visit_arguments(self, node):
        template = self.get_template('portlist.txt')
        values = [self.visit(arg) for arg in node.args]
        ports = [val["port"] for val in values]
        widths = [val["width"] for val in values]
        inputs = [val["input"] for val in values]

        if self.check_regs_present():
            # If at least one reg is present, generate clk port
            ports.append('clk')
            inputs.append('input clk;')

        rslt_portlist = template.render({'ports': ports, 'len_ports': len(ports)})
        rslt_inputlist = ''.join(inputs)
        return {
            "port_list": ports,
            "port_list_str": rslt_portlist,
            "input_list": rslt_inputlist,
            "width_list": widths
        }

    # ---------- expressions / literals ----------
    def visit_Return(self, node):
        val = self.visit(node.value)
        return val if isinstance(val, list) else [val]

    def visit_Tuple(self, node):
        return [self.visit(elts) for elts in node.elts]

    def visit_Name(self, node):
        return node.id

    # arithmetic
    def visit_Add(self, node):   return " + "
    def visit_Sub(self, node):   return " - "
    def visit_Mult(self, node):  return " * "
    def visit_Div(self, node):   return " / "

    # bitwise
    def visit_BitAnd(self, node): return " & "
    def visit_BitOr(self, node):  return " | "
    def visit_BitXor(self, node): return " ^ "
    def visit_LShift(self, node): return " << "
    def visit_RShift(self, node): return " >> "

    # unary
    def visit_UnaryOp(self, node):
        op = self.visit(node.op)
        operand = self.visit(node.operand)
        return f"{op}{operand}"

    def visit_USub(self, node):    return "-"
    def visit_UAdd(self, node):    return "+"
    def visit_Invert(self, node):  return "~"

    # literals
    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            return "1'b1" if node.value else "1'b0"
        if isinstance(node.value, int):
            return str(node.value)
        if isinstance(node.value, str):
            return f'"{node.value}"'
        return str(node.value)

    # Python <3.8 (optional back-compat)
    def visit_Num(self, node):  # pragma: no cover
        return str(node.n)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = self.visit(node.op)
        template = self.get_template("operator.txt")
        template_dict = {'left': left, 'right': right, 'op': op}
        rslt = template.render(template_dict)
        return rslt

    def visit_Assign(self, node):
        op = self.visit(node.value)
        rslt_regs = ''
        rslt_wires = ''
        for item in node.targets:
            target = self.visit(item)
            attr = self.check_attributes(target)
            if attr.get('type') == 'reg':
                template = self.get_template("regassign.txt")
                template_dict = {'left': target, 'right': op}
                rslt_regs += template.render(template_dict) + "\n"
            else:
                template = self.get_template("assign.txt")
                template_dict = {'left': target, 'right': op}
                rslt_wires += template.render(template_dict) + "\n"
        rslt_regs = indent_multiline_assign(rslt_regs)
        rslt_wires = indent_multiline_assign(rslt_wires)
        return self.indent(rslt_regs), self.indent(rslt_wires)

    def visit_Module(self, node):
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node):
        print(f"Found function: {node.name}")
        self.top_name = node.name
        moduledef_t = self.get_template("moduledef.txt")

        # Collect outputs and assignments
        output_decls = ''
        assign_list_regs = ''
        assign_list_wires = ''
        out_names = []

        for item in node.body:
            if isinstance(item, ast.Return):
                items = self.visit(item)  # list of output names
                self.output_args_list = items
                out_names = items
                for it in items:
                    attr = self.check_attributes(it)
                    self.output_bit_count += attr['width']
                    self.output_args_width_list.append(attr['width'])
                    out_t = self.get_template(
                        'outputreg.txt' if attr['type'] == 'reg' else 'output.txt'
                    )
                    output_decls += out_t.render({
                        'name': it,
                        'width': attr['width'],
                        'signed': attr['signed'],
                        'dimensions': '',
                    }) + "\n"
            elif isinstance(item, ast.Assign):
                regs, wires = self.visit(item)
                assign_list_regs += regs
                assign_list_wires += wires

        # inputs
        values = self.visit(node.args)
        portlist_str = self.indent(values['port_list_str'])
        self.input_args_list = values['port_list']
        self.input_args_width_list = values['width_list']
        input_decls = self.indent(values['input_list'])

        # outputs port list (for the module header)
        out_portlist_str = self.get_template('portlist.txt').render({
            'ports': out_names, 'len_ports': len(out_names)
        })

        # dump waves
        dump_t = self.get_template('dumpwaves.txt')
        dump_waves = dump_t.render({'top_name': self.top_name})

        # optional always block for reg assigns
        output_always = ''
        if self.check_regs_present():
            always_t = self.get_template('always.txt')
            output_always = always_t.render({'sens_list': 'clk', 'statement': assign_list_regs})

        items = [input_decls, output_decls, output_always, assign_list_wires, dump_waves]

        rslt = moduledef_t.render({
            'modulename': node.name,
            'paramlist': '',
            'portlist': portlist_str + ',\n' + out_portlist_str,
            'items': [self.indent(item) for item in items] if items else (),
        })
        return rslt

    # ---------- getters ----------
    def get_in_args_bit_count(self):
        return self.input_bit_count

    def get_out_bit_count(self):
        return self.output_bit_count

    def get_top_name(self):
        return self.top_name

    def get_input_args_width_list(self):
        return self.input_args_width_list

    def get_output_args_width_list(self):
        return self.output_args_width_list

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
        self.input_args_bits = f_visitor.get_in_args_bit_count()
        self.output_bits = f_visitor.get_out_bit_count()
        self.t_name = f_visitor.get_top_name()
        self.input_args_list = f_visitor.get_input_args_list()
        self.output_args_list = f_visitor.get_output_args_list()
        self.input_args_width_list = f_visitor.get_input_args_width_list()
        self.output_args_width_list = f_visitor.get_output_args_width_list()
        self.env = Environment(loader=FileSystemLoader(DEFAULT_TEMPLATE_DIR))
        self.template_cache = {}
        self.attr = attr
        self.syn_data = {
            'top_name': self.t_name,
            'inputs': [{'name': arg, 'width': width, 'delay': 0}
                       for arg, width in zip(f_visitor.get_input_args_list(),
                                             f_visitor.get_input_args_width_list())],
            'inputs_size': self.input_args_bits,
            'outputs': [{'name': arg, 'width': width, 'delay': 0}
                        for arg, width in zip(f_visitor.get_output_args_list(),
                                              f_visitor.get_output_args_width_list())],
            'outputs_size': self.output_bits
        }

        if not os.path.isdir("hdl"):
            os.mkdir("hdl")

        with open("hdl/main.v", "w+") as f:
            f.write(result)

    def HW(self, syn_attr):
        self.syn_attr = syn_attr
        self.createHWproject()
        return self.hw_run

    def hw_run(self):
        return 1

    def createHWproject(self):
        syn_tool_dir = self.syn_attr.get('QUARTUS_DIR')
        if syn_tool_dir is None:
            print("No syntesis tool specified")
            return

        # Delete previous results
        if os.path.isdir(os.path.join(os.getcwd(), "hw")):
            shutil.rmtree(os.path.join(os.getcwd(), "hw"))

        # Project settings
        project_name = self.t_name
        project_dir = os.path.join(os.getcwd(), "hw", project_name)
        top_level_entity = self.t_name

        # Create project directory
        os.makedirs(project_dir, exist_ok=True)

        # Write .qpf file
        with open(os.path.join(project_dir, f"{project_name}.qpf"), "w") as f:
            f.write(f"PROJECT_REVISION = {project_name}\n")

        # Write .qsf file
        template_qsf = self.get_template("hw/de0_nano_qsf.txt")
        rslt_qsf = template_qsf.render({})
        with open(os.path.join(project_dir, f"{project_name}.qsf"), "w") as f:
            f.write(rslt_qsf)

        # UART transceiver
        template_rxtx = self.get_template("hw/uart_transceiver.txt")
        template_rxtx_dict = {
            'in_clk_freq': 50000000,
            'baud_rate': 115200,
            'reg_width_tx': self.output_bits,
            'reg_width_rx': self.input_args_bits
        }
        rslt_rxtx = template_rxtx.render(template_rxtx_dict)
        with open(os.path.join(project_dir, "uart_transceiver.sv"), "w") as f:
            f.write(rslt_rxtx)

        # SDC
        template_sdc = self.get_template("hw/de0_nano_sdc.txt")
        rslt_sdc = template_sdc.render({})
        with open(os.path.join(project_dir, "DE0_Nano.SDC"), "w") as f:
            f.write(rslt_sdc)

        # Delay block
        template_delay = self.get_template("hw/delayed_registers.txt")
        rslt_delay = template_delay.render(self.syn_data)
        with open(os.path.join(project_dir, "delayed_registers.sv"), "w") as f:
            f.write(rslt_delay)

        # Board specific top file (DE0-Nano)
        template_board = self.get_template("hw/de0_nano_top.txt")
        rslt_board = template_board.render(self.syn_data)
        with open(os.path.join(project_dir, "DE0_Nano.v"), "w") as f:
            f.write(rslt_board)

        shutil.copy(os.path.join(os.getcwd(), "hdl", "main.v"),
                    os.path.join(project_dir, top_level_entity + ".v"))

        print(f"Quartus project '{project_name}' created at {project_dir}")

        # === Compile using Quartus command-line ===
        print("Starting compilation...")
        result = subprocess.run(
            [os.path.join(syn_tool_dir, "quartus_sh"), "--flow", "compile", project_name],
            cwd=project_dir, capture_output=True, text=True
        )
        print("Output:", result.stdout)

        # # === Program the FPGA (optional) ===
        # sof_file = os.path.join(project_dir, f"{project_name}.sof")
        # if os.path.exists(sof_file):
        #     print("Programming FPGA...")
        #     result = subprocess.run([
        #         os.path.join(syn_tool_dir,"quartus_pgm"), "-m", "jtag", "-o", f"p;{sof_file}"
        #     ], capture_output=True, text=True)
        #     print("Output:", result.stdout)
        # else:
        #     print("Error: .sof file not found. Compilation may have failed.")

    def TB(self):
        return self.tb_fun

    def get_template(self, filename):
        if filename in self.template_cache:
            return self.template_cache[filename]
        template = self.env.get_template(filename)
        self.template_cache[filename] = template
        return template

    def gen_tb(self, *in_arg):
        # Check if we have values for all arguments
        key_val = []
        for i in range(len(self.input_args_list)):
            # Skip reserved ports (clk)
            if self.input_args_list[i] != 'clk':
                if i < len(in_arg) and in_arg[i] is not None:
                    key_val.append((self.input_args_list[i], in_arg[i]))
                else:
                    key_val.append((self.input_args_list[i], 0))
        template = self.get_template("tb.txt")
        template_dict = {
            'period': 2000 if 'clk' in self.input_args_list else 0,
            'in_args': key_val,
            'out_args': self.output_args_list
        }
        tb_out = template.render(template_dict)
        with open("tb.py", "w+") as f:
            f.write(tb_out)

    def tosigned(self, n, nbits):
        mask = (1 << nbits) - 1
        n &= mask
        sign_bit = 1 << (nbits - 1)
        return (n ^ sign_bit) - sign_bit

    def tb_fun(self, *in_arg):
        print("I was called with", len(in_arg), "arguments:", in_arg)
        if os.path.isfile('results/results.pickle'):
            os.remove('results/results.pickle')
        self.gen_tb(*in_arg)
        tb_runner.test_runner(self.t_name)
        if os.path.isfile('results/results.pickle'):
            with open('results/results.pickle', 'rb') as handle:
                p = pickle.load(handle)
                d = ast.literal_eval(p)

                # Build outputs in declared order and widths
                out_vals = []
                for name, width in zip(self.output_args_list, self.output_args_width_list):
                    val = d.get(name, 0)
                    if name in self.attr and self.attr[name].get('signed', 0) == 1:
                        val = self.tosigned(val, width)
                    out_vals.append(val)
                b = tuple(out_vals)
        else:
            b = (0,) * len(self.output_args_list)
        return b
