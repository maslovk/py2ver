# A terminal on the ILI9341 display: what you type on the PC appears on the
# screen, 30 columns by 24 rows of text, and scrolls up when it reaches the
# bottom.
#
# The bytes come over the JTAG UART (the USB-Blaster cable that programs the
# board: run juart-terminal, in Quartus's bin directory, and type). Enter
# starts a new line, backspace rubs out the character before the cursor.
#
#   terminal --color/valid/ready--> ili9341_display
#
# The screen's characters are a list in a memory block, and the font (X11's
# 8 x 13 fixed font, py2ver/devices/font.py) a table in another. Each frame
# draws every row of pixels of every row of text: for each character, its row
# of the font, a pixel a bit. Scrolling moves which row of the list is shown
# at the top (nothing is copied), and clears the row that comes in at the bottom.
#
#   python examples/terminal.py          # type a little, test it, save terminal.png
#   python examples/terminal.py --sim    # also simulate the generated Verilog
#   python examples/terminal.py --pins   # build it for the board (wiring: mandelbrot_lcd.py)
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mandelbrot_lcd import PINS as LCD_PINS  # noqa: E402
from py2ver import connect  # noqa: E402
from py2ver.devices import font as F  # noqa: E402
from py2ver.devices.ili9341 import HEIGHT as SCREEN_HEIGHT  # noqa: E402
from py2ver.devices.ili9341 import ILI9341, ili9341_display, rgb565  # noqa: E402
from py2ver.protocol import SPI, stream_send  # noqa: E402

COLUMNS = 30                             # 240 pixels across: 30 characters
ROWS = 24                                # 320 down: 24 rows of 13 (and 8 pixels to spare)
INK = rgb565(80, 255, 120)               # green on black, as on an old terminal
PAPER = rgb565(0, 16, 8)
GREETING = b"py2ver on the DE0-Nano"
ENTER, NEW_LINE, BACKSPACE, DELETE = 13, 10, 8, 127


def take_byte(port):
    """A byte from the PC, if one is waiting: (1, it), else (0, the input as it
    is). Takes a clock: rx_ready is high in it, so a byte there is taken then."""
    port.rx_ready = 1
    yield
    got = port.rx_valid
    byte = port.rx
    port.rx_ready = 0
    return got, byte


def terminal(port, columns=COLUMNS, rows=ROWS, height=SCREEN_HEIGHT, ink=INK, paper=PAPER):
    """The terminal, frame after frame, sending each pixel's color.

    columns, rows: the text's size, each character 8 x 13 pixels (font.py);
    height: the screen's, in pixels (the rows below the text are paper).
    Input: the PC's bytes (rx, rx_valid; rx_ready: taken); ready.
    Output stream: color (RGB565), valid (see protocol.stream_send).
    """
    port.valid = 0
    port.rx_ready = 0
    text = [32] * (columns * rows)               # the screen's characters, row by row: spaces
    top = 0                                      # the row of the list shown at the top
    row = 0                                      # the cursor: where the next character goes,
    col = 0                                      # counted from the top of the screen
    last = 0                                     # (the byte before: Enter is often \r\n)
    frame = 0
    for k in range(min(len(GREETING), columns)):     # a greeting on the first row
        text[k] = GREETING[k]
        yield
    row = 2
    while True:
        blink = (frame >> 3) & 1                 # the cursor: on 8 frames, off 8
        for line in range(rows):                 # each row of text on the screen,
            shown = top + line                   # (its row in the list: round from the top)
            if shown >= rows:
                shown = shown - rows
            for y in range(F.HEIGHT):            # each row of its pixels,
                # A byte from the PC, if one has come
                got, byte = yield from take_byte(port)
                here = top + row                 # (the cursor's row in the list)
                if here >= rows:
                    here = here - rows
                if got and (byte == ENTER or (byte == NEW_LINE and last != ENTER)):
                    col = 0                      # Enter: the start of the next line
                    row = row + 1
                elif got and (byte == BACKSPACE or byte == DELETE) and col > 0:
                    col = col - 1                # back a character, rubbed out
                    text[here * columns + col] = 32
                    yield
                elif got and byte >= F.FIRST and byte <= F.LAST:
                    text[here * columns + col] = byte
                    yield
                    col = col + 1
                    if col == columns:           # (past the end of the line: the next)
                        col = 0
                        row = row + 1
                if got:
                    last = byte
                if row == rows:                  # past the bottom: up a row, a clear one
                    row = rows - 1               # coming in at the bottom
                    top = 0 if top == rows - 1 else top + 1
                    bottom = top + row
                    if bottom >= rows:
                        bottom = bottom - rows
                    for x in range(columns):
                        text[bottom * columns + x] = 32
                        yield
                # The row of pixels: each character's row of the font, a pixel a bit
                for x in range(columns):
                    c = text[shown * columns + x]
                    yield
                    bits = F.FONT[(c - F.FIRST) * F.HEIGHT + y]
                    yield
                    if blink and line == row and x == col and y >= F.HEIGHT - 2:
                        bits = 0xFF              # (the cursor: a bar under its place)
                    for b in range(F.WIDTH):
                        lit = (bits >> (F.WIDTH - 1 - b)) & 1
                        yield from stream_send(port, ink if lit else paper, data="color")
        for k in range(columns * F.WIDTH * (height - rows * F.HEIGHT)):
            yield from stream_send(port, paper, data="color")    # (below the text)
        frame = 0 if frame == 15 else frame + 1


ATTR = {"ready": 1, "rx": 8, "rx_valid": 1}

# SCK at 25 MHz, as in mandelbrot_zoom.py (if the picture is noisy: SPI(2, ...))
LCD = ILI9341(spi=SPI(1, mode=0, miso=""))

# On the board: the display's pins, and the JTAG UART's for the PC's bytes
PINS = dict(LCD_PINS, rx="JTAG_RX_DATA", rx_valid="JTAG_RX_VALID", rx_ready="JTAG_RX_READY")


def show_on_screen(lcd=LCD, columns=COLUMNS, rows=ROWS, height=SCREEN_HEIGHT):
    """The terminal feeding the display."""
    source = functools.partial(terminal, columns=columns, rows=rows, height=height)
    width = columns * F.WIDTH
    display = functools.partial(ili9341_display, lcd=lcd, width=width, height=height)
    return connect(terminal=(source, ATTR), display=(display, {"color": 16, "valid": 1}),
                   name="text_terminal",
                   wires={"display.color": "terminal.color", "display.valid": "terminal.valid",
                          "terminal.ready": "display.ready"})


def picture(lines, columns, rows, height, cursor=None):
    """The screen showing `lines` (a row of text each, from the top), worked out
    directly; cursor: (row, column) of a cursor drawn there, or None."""
    rows_of_pixels = []
    for line in range(rows):
        text = (lines[line] if line < len(lines) else "").ljust(columns)
        for y in range(F.HEIGHT):
            pixels = []
            for x, ch in enumerate(text):
                bits = F.glyph(ord(ch))[y]
                if cursor == (line, x) and y >= F.HEIGHT - 2:
                    bits = 0xFF
                pixels += [INK if (bits >> (7 - b)) & 1 else PAPER for b in range(F.WIDTH)]
            rows_of_pixels.append(pixels)
    rows_of_pixels += [[PAPER] * (columns * F.WIDTH)] * (height - rows * F.HEIGHT)
    return rows_of_pixels


def typing(screen, text, count, width, height):
    """A testbench that plays the PC: it types `text`, a byte whenever the
    terminal takes one, until the display has received `count` frames; a
    copy of the picture after each."""
    from py2ver.protocol import alongside
    shots = []

    def pc(dut):
        typed = 0
        while len(shots) < count:
            dut.rx_valid = 0
            if typed < len(text) and dut.rx_ready:
                dut.rx, dut.rx_valid = text[typed], 1
                typed += 1
            if screen.written >= (len(shots) + 1) * width * height:
                shots.append([list(r) for r in screen.pixels])
            yield

    def tb(dut):
        yield from alongside(pc(dut), screen.run(dut))
        return shots
    return tb


if __name__ == "__main__":
    from mandelbrot_zoom import film_strip
    from py2ver import Py2ver
    from py2ver.devices.ili9341 import ILI9341Model, rgb888
    from py2ver.devices.spi_target import save_png

    args = sys.argv[1:]
    # A small screen for the simulation: 8 columns, 8 rows (64 x 104 pixels);
    # "hi", Enter, "ok" typed
    columns, rows, count = 8, 8, 3
    width, height = columns * F.WIDTH, rows * F.HEIGHT
    quick = ILI9341(spi=SPI(1, mode=0, miso=""), wake_clocks=50)
    screen = ILI9341Model(quick, width=width, height=height)
    p = Py2ver(show_on_screen(quick, columns, rows, height))
    run = p.check(typing(screen, b"hi\rok", count, width, height), verilog="--sim" in args)
    want = picture(["py2ver o", "", "hi", "ok"], columns, rows, height)
    assert run.value[-1] == want, "the screen doesn't show what was typed"
    save_png("terminal.png", film_strip(run.value, scale=3), rgb888)
    print(f"{count} frames of {width} x {height} in {len(run.inputs):,} clock cycles: terminal.png")
    if "--pins" in args:
        board = Py2ver(show_on_screen(), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
