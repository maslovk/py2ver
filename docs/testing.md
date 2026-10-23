# Running py2ver's tests


Tests run in parallel (`-n auto`, pytest-xdist from the `test` extra) and are
marked by what they need, so a quick check after a change doesn't wait for
everything:

```
.venv/bin/python -m pytest -q -n auto -m "not sim and not board and not slow"   # no simulator: ~30 s
.venv/bin/python -m pytest -q -n auto -m "not board and not slow"  # + simulation: ~2.5 min
.venv/bin/python -m pytest -q -n auto                              # all but the long ones: ~5.5 min
.venv/bin/python -m pytest -q -n auto --long                       # everything: ~10 min
.venv/bin/python -m pytest -q -n auto tests/test_share.py          # one feature's tests
.venv/bin/python -m pytest -q --lf                                 # what failed last time
.venv/bin/python -m pytest -q -n auto -m "not quartus"             # everything but Quartus
```

- `sim`: compares the Python function against a simulation (the native
  testbench on Icarus or Verilator, or cocotb); skipped if Icarus is missing.
  A test that isn't marked `sim` (or `board`) and starts a simulator fails and
  says so, which keeps the quick run quick.
- `board`: the virtual DE0-Nano (`HW_sim()`): the whole project, UART
  included, simulated.
- `slow`: a large design simulated (a torch training step: minutes, most of
  them compiling its Verilog), or a Quartus compile.
- `quartus`: a real Quartus compile (area, timing at 50 MHz: an SDRAM design
  of the mini Llama's must meet it), with Quartus from `QUARTUS_DIR`,
  `quartus_sh` on `PATH`, or its usual install
  (`/opt/intelFPGA/22.1std/quartus/bin`); skipped if none of them has it.
- `long`: minutes each (the mini Llama's training step simulated, its SDRAM
  design and the stages in Quartus): skipped unless pytest runs with `--long`.
- `hardware`: on a DE0-Nano plugged in (`tests/test_hardware.py`: a design
  compiled, programmed into the board and answering over the JTAG UART as
  its generator does): skipped unless pytest runs with `--hardware`.

The SDRAM controller is tested against py2ver's model of the chip, and also
against Micron's MT48LC16M16A2 model (organized as the DE0-Nano's chip; not in
the repository) when `PY2VER_MICRON_MODEL` gives the path of its
`mt48lc16m16a2.v`.

Every test builds in its own temporary directory, so they can run in parallel.
`tests/test_combinations.py` builds a set of designs (state, lists, complex
values, tables, block RAM, helpers, keyword names) with each option
(pipelining, folding, automatic formats, serial ports, stages, annotations)
and checks every result against the references.

A change to how the Verilog is written (the same hardware, other text) is
checked with `tests/icarus.py`: `assert_same_bits` runs pairs of expressions
on random inputs in Icarus and says which differ, and `assert_same_modules`
runs two modules with the same ports side by side, from reset if they are
clocked:

```python
p = Py2ver(f, attr, write=False)
assert_same_modules(tmp_path, before, p.verilog, "f", inputs, outputs)   # mark it sim
```
