"""py2ver's defaults: files, the DE0-Nano's clock and UART, the testbench's clock and
standalone synthesis's device."""

from pathlib import Path

RESULTS_PATH = Path("results/results.pickle")
DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
DEFAULT_CLK_FREQ = 50_000_000
DEFAULT_BAUD_RATE = 115_200
DEFAULT_TB_PERIOD = 2000

# Standalone synthesis defaults (DE0-Nano device, 50 MHz)
DEFAULT_SYN_FAMILY = "Cyclone IV E"
DEFAULT_SYN_DEVICE = "EP4CE22F17C6"
DEFAULT_SYN_PERIOD_NS = 20.0
