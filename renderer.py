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

    def render_module(self, ir: ModuleIR, attr: Dict[str, Dict[str, Any]] | None = None) -> str:
        """
        Produce the full module (what used to be 'moduledef.txt' as the outer template).
        We keep sub-templates identical to your current ones:
          - port.txt, input.txt, output.txt, outputreg.txt, operator.txt,
            always.txt, dumpwaves.txt, assign.txt, regassign.txt,
          plus a new 'wire.txt' for internal nets.

        'attr' should be the normalized attribute dict from FunctionVisitor.attr,
        so that internal SSA temporaries (e.g. acc_0, acc_1, ...) have width/signed info.
        """
        if attr is None:
            attr = {}

        # ----------------- Inputs -----------------
        input_decls = []
        input_port_names = []
        for p in ir.inputs:
            port_name = p.name if p.kind != 'clk' else 'clk'
            input_port_names.append(port_name)
            if p.kind == 'input' or p.kind == 'clk':
                input_decls.append(
                    self._tpl('input.txt').render({
                        'name': port_name,
                        'width': p.width,
                        'signed': int(p.signed),
                        'dimensions': '',
                    })
                )

        # ----------------- Outputs -----------------
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

        # ----------------- Internal wires/regs -----------------
        # Anything assigned but not a port is "internal"
        port_name_set = set(input_port_names + output_port_names)
        lhs_names = {a.left for a in ir.assigns}
        internal_names = sorted(lhs_names - port_name_set)

        internal_decls: List[str] = []

        # Helper: look up a "base" signal's meta (from attr or ports)
        def _lookup_base_meta(base_name: str) -> Dict[str, Any]:
            # Prefer attr if present
            if base_name in attr:
                return attr[base_name]
            # Otherwise infer from ports
            for p in (ir.inputs + ir.outputs):
                if p.name == base_name:
                    return {
                        'width': p.width,
                        'signed': int(p.signed),
                        'type': 'wire',
                    }
            # Fallback default
            return {'width': 1, 'signed': 0, 'type': 'wire'}

        for name in internal_names:
            meta = attr.get(name)

            if meta is None:
                # Try pattern: <base>_<idx>, where idx is an integer (e.g. acc_0)
                base_name = None
                if '_' in name:
                    head, tail = name.rsplit('_', 1)
                    if tail.isdigit():
                        base_name = head

                if base_name is not None:
                    meta = _lookup_base_meta(base_name)
                else:
                    meta = {'width': 1, 'signed': 0, 'type': 'wire'}

            width = int(meta.get('width', 1))
            signed = int(meta.get('signed', 0))

            internal_decls.append(
                self._tpl('wire.txt').render({
                    'name': name,
                    'width': width,
                    'signed': signed,
                    'dimensions': '',
                })
            )

        # ----------------- Assignments (reg vs wire) -----------------
        reg_assigns = []
        wire_assigns = []
        for a in ir.assigns:
            if a.is_reg:
                reg_assigns.append(self._tpl('regassign.txt').render({'left': a.left, 'right': a.right}))
            else:
                wire_assigns.append(self._tpl('assign.txt').render({'left': a.left, 'right': a.right}))

        # ----------------- Optional always block -----------------
        output_always = ''
        if ir.has_clk and reg_assigns:
            output_always = self._tpl('always.txt').render({
                'sens_list': 'clk',
                'statement': '\n'.join(reg_assigns)
            })

        # ----------------- Dumpwaves -----------------
        dumpwaves = self._tpl('dumpwaves.txt').render({'top_name': ir.name})

        # ----------------- Portlist in the header -----------------
        header_input_ports = input_port_names
        header_output_ports = output_port_names
        portlist_header = (
            self._render_portlist(header_input_ports)
            + ',\n'
            + self._render_portlist(header_output_ports)
        )

        # ----------------- Items block body -----------------
        items = []
        if input_decls:
            items.append('\n'.join(input_decls))
        if output_decls:
            items.append('\n'.join(output_decls))
        if internal_decls:
            items.append('\n'.join(internal_decls))
        if output_always:
            items.append(output_always)
        if wire_assigns:
            items.append('\n'.join(wire_assigns))
        if dumpwaves:
            items.append(dumpwaves)

        # ----------------- Render module wrapper -----------------
        return self._tpl('moduledef.txt').render({
            'modulename': ir.name,
            'paramlist': '',
            'portlist': portlist_header,
            'items': items,
        })
