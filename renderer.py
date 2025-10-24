# renderer.py
from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from ast_ir import ModuleIR, Port, Assignment

def make_env(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

class Renderer:
    def __init__(self, template_dir: str):
        self.env = make_env(template_dir)
        self._cache: Dict[str, Any] = {}

    def _tpl(self, name: str):
        if name in self._cache:
            return self._cache[name]
        t = self.env.get_template(name)
        self._cache[name] = t
        return t

    def _render_portlist(self, names: List[str]) -> str:
        return self._tpl('portlist.txt').render({'ports': names, 'len_ports': len(names)})

    def render_module(self, ir: ModuleIR) -> str:
        """
        Produce the full module (what used to be 'moduledef.txt' as the outer template).
        We keep sub-templates identical to your current ones:
          - port.txt, input.txt, output.txt, outputreg.txt, operator.txt, always.txt, dumpwaves.txt
        """
        # Inputs
        input_decls = []
        input_port_names = []
        for p in ir.inputs:
            input_port_names.append(p.name if p.kind != 'clk' else 'clk')
            if p.kind == 'input' or p.kind == 'clk':
                input_decls.append(
                    self._tpl('input.txt').render({
                        'name': p.name if p.kind != 'clk' else 'clk',
                        'width': p.width,
                        'signed': int(p.signed),
                        'dimensions': '',
                    })
                )

        # Outputs
        output_decls = []
        output_port_names = []
        for p in ir.outputs:
            output_port_names.append(p.name)
            tpl = 'outputreg.txt' if p.kind == 'reg' else 'output.txt'
            output_decls.append(
                self._tpl(tpl).render({
                    'name': p.name,
                    'width': p.width,
                    'signed': int(p.signed),
                    'dimensions': '',
                })
            )

        # Assignments (wire vs reg in always)
        reg_assigns = []
        wire_assigns = []
        for a in ir.assigns:
            if a.is_reg:
                reg_assigns.append(self._tpl('regassign.txt').render({'left': a.left, 'right': a.right}))
            else:
                wire_assigns.append(self._tpl('assign.txt').render({'left': a.left, 'right': a.right}))

        # Optional always block
        output_always = ''
        if ir.has_clk and reg_assigns:
            output_always = self._tpl('always.txt').render({
                'sens_list': 'clk',
                'statement': '\n'.join(reg_assigns)
            })

        # Dumpwaves
        dumpwaves = self._tpl('dumpwaves.txt').render({'top_name': ir.name})

        # Portlist in the module header: inputs first then outputs (like before)
        # Add clk to port list if present
        header_input_ports = input_port_names
        header_output_ports = output_port_names
        portlist_header = self._render_portlist(header_input_ports) + ',\n' + self._render_portlist(header_output_ports)

        # Items block body
        items = []
        if input_decls:
            items.append('\n'.join(input_decls))
        if output_decls:
            items.append('\n'.join(output_decls))
        if output_always:
            items.append(output_always)
        if wire_assigns:
            items.append('\n'.join(wire_assigns))
        if dumpwaves:
            items.append(dumpwaves)

        # Render module wrapper
        return self._tpl('moduledef.txt').render({
            'modulename': ir.name,
            'paramlist': '',
            'portlist': portlist_header,
            'items': items,
        })
