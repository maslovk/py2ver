# A marble maze on the ILI9341 display: tilt the board to roll the marble
# through the maze to the green exit, against the clock. KEY1 then starts the
# next maze (there are four, then the first again).
#
# Four processes, joined:
#
#   tilt_sensor --tilt_x, tilt_y--> marble_maze --color/valid/ready--> ili9341_display
#                     stopwatch --the time's digits--> marble_maze --running--> stopwatch
#
# The mazes are made before building (each a random maze, a path between
# every two places): 8 x 8 blocks of wall or floor, a table. Each frame the
# tilt speeds the marble up, friction slows it, and it moves one way, then the
# other: if the blocks its leading edge would move into are wall, it bounces
# back off them at half the speed instead. Then the frame is drawn: the time
# and the maze's number in the top row (py2ver/devices/font.py), the maze,
# its exit, the marble.
#
#   python examples/marble_maze.py          # a small maze, tilted one way: test it, save marble_maze.png
#   python examples/marble_maze.py --sim    # also simulate the generated Verilog
#   python examples/marble_maze.py --pins   # build it for the board (wiring: mandelbrot_lcd.py)
import functools
import os
import random
import sys
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mandelbrot_lcd import PINS as LCD_PINS  # noqa: E402
from py2ver import connect  # noqa: E402
from py2ver.devices import font as F  # noqa: E402
from py2ver.devices.adxl345 import adxl345_read_xyz, adxl345_start  # noqa: E402
from py2ver.devices.ili9341 import HEIGHT, ILI9341, WIDTH, ili9341_display, rgb565  # noqa: E402
from py2ver.protocol import I2C, SPI, stream_send  # noqa: E402

# ----------------------------- the tilt sensor -----------------------------

BUS = I2C(q=50_000_000 // 400_000 // 4)  # 400 kHz at 50 MHz, on the pins scl and sda


def tilt_sensor(port, bus=BUS, period=100_000):
    """The ADXL345's X and Y, read every period (2 ms), kept on port.tilt_x and
    port.tilt_y: signed, 1 g (the board on its edge) about 256."""
    port.scl = 1
    port.sda = 1
    port.accel_cs_n = 1                  # (the ADXL345 talks I2C while its CS is high)
    port.tilt_x = 0
    port.tilt_y = 0
    yield 50_000                         # (the bus settles: 1 ms)
    yield from adxl345_start(port, bus)
    while True:
        x, y, z, nack = yield from adxl345_read_xyz(port, bus)
        if not nack:
            port.tilt_x = x - 65536 if x >= 32768 else x      # (16-bit two's complement)
            port.tilt_y = y - 65536 if y >= 32768 else y
        yield period


SENSOR_ATTR = {"scl": {"open_drain": True}, "sda": {"open_drain": True},
               "tilt_x": {"width": 16, "signed": 1}, "tilt_y": {"width": 16, "signed": 1}}

# ----------------------------- the stopwatch -----------------------------


def stopwatch(port, tick=5_000_000):
    """The time while port.running is 1, in decimal digits (hundreds, tens and
    ones of seconds, tenths): from 0 each time running goes from 0 to 1, and
    standing still while it is 0. tick: a tenth of a second, in clocks."""
    port.hundreds = 0
    port.tens = 0
    port.ones = 0
    port.tenths = 0
    was = 0
    while True:
        if port.running and not was:             # (a new run: from 0)
            port.hundreds = 0
            port.tens = 0
            port.ones = 0
            port.tenths = 0
        elif port.running:                       # (a tenth more, carried: 9.9 to 10.0)
            if port.tenths < 9:
                port.tenths = port.tenths + 1
            else:
                port.tenths = 0
                if port.ones < 9:
                    port.ones = port.ones + 1
                else:
                    port.ones = 0
                    if port.tens < 9:
                        port.tens = port.tens + 1
                    else:
                        port.tens = 0
                        port.hundreds = port.hundreds + 1 if port.hundreds < 9 else 9
        was = port.running
        yield tick


STOPWATCH_ATTR = {"running": 1}

# ----------------------------- the mazes -----------------------------

BLOCK = 8                                # a block of the maze: 8 x 8 pixels
TOP = 2 * BLOCK                          # the maze below the row of text
MARBLE = (0b011110, 0b111111, 0b111111, 0b111111, 0b111111, 0b011110)   # its rows, 6 x 6
SIDE = len(MARBLE)


def random_maze(cells_across, cells_down, seed):
    """A random maze of cells_across x cells_down places, each 2 x 2 blocks of
    floor, with a block of wall (or of floor: a way through) between them:
    its blocks row by row, 1 a wall. Every two places are joined by one path
    (a depth-first walk, its turns from `seed`)."""
    across, down = 3 * cells_across + 1, 3 * cells_down + 1
    blocks = [1] * (across * down)
    rng = random.Random(seed)

    def floor(c, r, w=2, h=2):
        for y in range(3 * r + 1, 3 * r + 1 + h):
            for x in range(3 * c + 1, 3 * c + 1 + w):
                blocks[y * across + x] = 0
    seen = {(0, 0)}
    path = [(0, 0)]
    floor(0, 0)
    while path:
        c, r = path[-1]
        ways = [(c + dc, r + dr) for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= c + dc < cells_across and 0 <= r + dr < cells_down
                and (c + dc, r + dr) not in seen]
        if not ways:
            path.pop()
            continue
        nc, nr = rng.choice(ways)
        floor(nc, nr)
        # (the way through: the wall between the two places)
        floor(min(c, nc), min(r, nr), 3 if nc != c else 2, 3 if nr != r else 2)
        seen.add((nc, nr))
        path.append((nc, nr))
    return blocks, across, down


class Maze(NamedTuple):
    """The mazes for a screen: `levels` of them, each `across` x `down` blocks,
    their blocks one after another (blocks[(level * down + row) * across + col],
    1 a wall); where the marble starts (pixels: its top left), the exit (the
    block at its top left)."""
    width: int
    height: int
    across: int
    down: int
    levels: int
    blocks: tuple
    start_x: int
    start_y: int
    exit_col: int
    exit_row: int


def mazes(width=WIDTH, height=HEIGHT, levels=4, seed=1):
    """`levels` random mazes, as many places across and down as fit the screen."""
    cells_across = (width // BLOCK - 1) // 3
    cells_down = ((height - TOP) // BLOCK - 1) // 3
    blocks = []
    for level in range(levels):
        b, across, down = random_maze(cells_across, cells_down, seed + level)
        blocks += b
    middle = BLOCK + (2 * BLOCK - SIDE) // 2  # (the first place's middle)
    return Maze(width, height, across, down, levels, tuple(blocks), start_x=middle,
                start_y=TOP + middle, exit_col=3 * (cells_across - 1) + 1,
                exit_row=3 * (cells_down - 1) + 1)


MAZES = mazes()

# ----------------------------- the game -----------------------------

WALL = rgb565(70, 90, 150)
FLOOR = rgb565(15, 15, 25)
EXIT = rgb565(40, 170, 70)
MARBLE_COLOR = rgb565(250, 250, 250)
INK = rgb565(230, 230, 230)
TIME_TEXT = b"TIME "
MAZE_TEXT = b"MAZE "
MADE_IT = b"YOU MADE IT! KEY1"
FAST = 40                                # the marble's top speed: 40 / 16 = 2.5 pixels a frame


def marble_maze(port, maze=MAZES, grip=5):
    """The game, frame after frame, sending each pixel's color.

    grip: the tilt speeds the marble up by tilt >> grip sixteenths of a pixel a
    frame, each frame (1 g: 8 with 5). Inputs: tilt_x, tilt_y; the time's
    digits (hundreds, tens, ones, tenths); key (0 while pressed); ready.
    Outputs: running (to the stopwatch); the stream color, valid.
    """
    port.valid = 0
    port.running = 1
    level = 0
    x = maze.start_x << 4                        # the marble's top left, in sixteenths
    y = maze.start_y << 4                        # of a pixel, and its speed
    vx = 0
    vy = 0
    made_it = 0                                  # (at the exit: waiting for KEY1)
    was = 1
    while True:
        key = port.key
        pressed = was == 1 and key == 0          # (pressed just now)
        was = key
        base = level * maze.across * maze.down   # (this maze's first block)
        yield
        if made_it:
            if pressed:                          # the next maze
                level = 0 if level == maze.levels - 1 else level + 1
                x = maze.start_x << 4
                y = maze.start_y << 4
                vx = 0
                vy = 0
                made_it = 0
                port.running = 1
        else:
            # The tilt speeds it up, friction slows it down (the accelerometer's
            # Y points up the screen, so its sign turned for rows going down)
            vx = max(-FAST, min(FAST, vx + (port.tilt_x >> grip) - (vx >> 4)))
            vy = max(-FAST, min(FAST, vy - (port.tilt_y >> grip) - (vy >> 4)))
            yield
            # Across: the blocks its leading edge would move into, top and bottom
            to = max(0, min((maze.width - SIDE) << 4, x + vx))
            edge = (to >> 4) + SIDE - 1 if vx > 0 else to >> 4
            wall = maze.blocks[base + (((y >> 4) - TOP) >> 3) * maze.across + (edge >> 3)]
            yield
            hit = wall
            wall = maze.blocks[base + (((y >> 4) + SIDE - 1 - TOP) >> 3) * maze.across + (edge >> 3)]
            yield
            if hit or wall:
                vx = -(vx >> 1)                  # (off the wall, at half the speed)
            else:
                x = to
            # Down: the same
            to = max(TOP << 4, min((maze.height - SIDE) << 4, y + vy))
            edge = (to >> 4) + SIDE - 1 if vy > 0 else to >> 4
            wall = maze.blocks[base + ((edge - TOP) >> 3) * maze.across + ((x >> 4) >> 3)]
            yield
            hit = wall
            wall = maze.blocks[base + ((edge - TOP) >> 3) * maze.across + (((x >> 4) + SIDE - 1) >> 3)]
            yield
            if hit or wall:
                vy = -(vy >> 1)
            else:
                y = to
            # At the exit (its middle in the exit's 2 x 2 blocks): the clock stops
            col = ((x >> 4) + SIDE // 2) >> 3
            row = ((y >> 4) + SIDE // 2 - TOP) >> 3
            if col >= maze.exit_col and col <= maze.exit_col + 1 \
                    and row >= maze.exit_row and row <= maze.exit_row + 1:
                made_it = 1
                port.running = 0
        # ---- the picture ----
        mx = x >> 4
        my = y >> 4
        columns = maze.width // F.WIDTH
        message_y = maze.height // 2 - F.HEIGHT // 2
        for r in range(maze.height):
            message = made_it and r >= message_y and r < message_y + F.HEIGHT
            if r < F.HEIGHT or message:
                # A row of text: the time and the maze's number, or YOU MADE IT
                line = r if r < F.HEIGHT else r - message_y
                start = (columns - len(MADE_IT)) // 2
                for k in range(columns):
                    c = 32
                    if message:
                        if k >= start and k < start + len(MADE_IT):
                            c = MADE_IT[k - start]
                    elif k < len(TIME_TEXT):
                        c = TIME_TEXT[k]
                    elif k == len(TIME_TEXT):
                        c = 48 + port.hundreds
                    elif k == len(TIME_TEXT) + 1:
                        c = 48 + port.tens
                    elif k == len(TIME_TEXT) + 2:
                        c = 48 + port.ones
                    elif k == len(TIME_TEXT) + 3:
                        c = 46                   # '.'
                    elif k == len(TIME_TEXT) + 4:
                        c = 48 + port.tenths
                    elif k >= columns - len(MAZE_TEXT) - 1 and k < columns - 1:
                        c = MAZE_TEXT[k - (columns - len(MAZE_TEXT) - 1)]
                    elif k == columns - 1:
                        c = 49 + level           # (the maze's number, from 1)
                    bits = F.FONT[(c - F.FIRST) * F.HEIGHT + line]
                    yield
                    for b in range(F.WIDTH):
                        lit = (bits >> (F.WIDTH - 1 - b)) & 1
                        yield from stream_send(port, INK if lit else FLOOR, data="color")
            elif r < TOP or r >= TOP + maze.down * BLOCK:
                for k in range(maze.width):      # (between the text and the maze, below it)
                    yield from stream_send(port, FLOOR, data="color")
            else:
                # A row of the maze: each block's 8 pixels, the marble over them
                block_row = (r - TOP) >> 3
                in_exit = block_row >= maze.exit_row and block_row <= maze.exit_row + 1
                on_marble = r >= my and r < my + SIDE
                for col in range(maze.across):
                    solid = maze.blocks[base + block_row * maze.across + col]
                    yield
                    ground = WALL if solid else FLOOR
                    if in_exit and col >= maze.exit_col and col <= maze.exit_col + 1:
                        ground = EXIT
                    for b in range(BLOCK):
                        px = col * BLOCK + b
                        color = ground
                        if on_marble and px >= mx and px < mx + SIDE \
                                and (MARBLE[r - my] >> (SIDE - 1 - (px - mx))) & 1:
                            color = MARBLE_COLOR
                        yield from stream_send(port, color, data="color")
                for k in range(maze.width - maze.across * BLOCK):
                    yield from stream_send(port, FLOOR, data="color")


GAME_ATTR = {"ready": 1, "key": 1, "tilt_x": {"width": 16, "signed": 1},
             "tilt_y": {"width": 16, "signed": 1},
             "hundreds": 4, "tens": 4, "ones": 4, "tenths": 4}

# SCK at 25 MHz, as in mandelbrot_zoom.py (if the picture is noisy: SPI(2, ...))
LCD = ILI9341(spi=SPI(1, mode=0, miso=""))

# On the board: the display's pins, KEY1, and the accelerometer's I2C bus and CS
PINS = dict(LCD_PINS, key="KEY[1]", scl="I2C_SCLK", sda="I2C_SDAT", accel_cs_n="G_SENSOR_CS_N")


def show_on_screen(lcd=LCD, maze=MAZES, bus=BUS, period=100_000, tick=5_000_000, grip=5):
    """The sensor tilting the maze, the stopwatch timing it, the game feeding the display."""
    sensor = functools.partial(tilt_sensor, bus=bus, period=period)
    clock = functools.partial(stopwatch, tick=tick)
    game = functools.partial(marble_maze, maze=maze, grip=grip)
    display = functools.partial(ili9341_display, lcd=lcd, width=maze.width, height=maze.height)
    wires = {"game.tilt_x": "sensor.tilt_x", "game.tilt_y": "sensor.tilt_y",
             "clock.running": "game.running", "display.color": "game.color",
             "display.valid": "game.valid", "game.ready": "display.ready"}
    wires.update({f"game.{d}": f"clock.{d}" for d in ("hundreds", "tens", "ones", "tenths")})
    return connect(sensor=(sensor, SENSOR_ATTR), clock=(clock, STOPWATCH_ATTR),
                   game=(game, GAME_ATTR), display=(display, {"color": 16, "valid": 1}),
                   name="marble_maze", wires=wires)


def frames(screen, count, maze, tilt=(0, 0), press_at=()):
    """A testbench: the board tilted by `tilt` (x, y; 1 g is 256), KEY1 pressed
    in the frames press_at; run until the display has received `count`
    frames, a copy of the picture after each."""
    from py2ver.devices.adxl345 import ADDRESS, registers
    from py2ver.devices.i2c_target import i2c_memory
    from py2ver.protocol import alongside
    shots = []

    def watch(dut):
        while len(shots) < count:
            dut.key = 0 if len(shots) in press_at else 1     # (the board's keys: 0 pressed)
            if screen.written >= (len(shots) + 1) * maze.width * maze.height:
                shots.append([list(r) for r in screen.pixels])
            yield

    def tb(dut):
        yield from alongside(watch(dut), screen.run(dut),
                             i2c_memory(dut, registers(*tilt), 1_000_000, ADDRESS))
        return shots
    return tb


def marble_at(shot):
    """Where the marble is in a picture (its top left), or None."""
    for r, row in enumerate(shot):
        if MARBLE_COLOR in row:
            return row.index(MARBLE_COLOR) - 1, r     # (its top row is 4 wide: one in)
    return None


if __name__ == "__main__":
    from mandelbrot_zoom import film_strip
    from py2ver import Py2ver
    from py2ver.devices.ili9341 import ILI9341Model, rgb888
    from py2ver.devices.spi_target import save_png

    args = sys.argv[1:]
    # A small maze for the simulation: 80 x 88 pixels (3 x 2 places), the
    # board tilted to the right
    small = mazes(80, 88, levels=1)
    count = 5
    quick = ILI9341(spi=SPI(1, mode=0, miso=""), wake_clocks=50)
    screen = ILI9341Model(quick, width=small.width, height=small.height)
    p = Py2ver(show_on_screen(quick, small, bus=I2C(q=4), period=2_000, tick=20_000, grip=3))
    run = p.check(frames(screen, count, small, tilt=(128, 0)), verilog="--sim" in args)
    save_png("marble_maze.png", film_strip(run.value, scale=3), rgb888)
    print(f"{count} frames of {small.width} x {small.height} in {len(run.inputs):,} clock "
          f"cycles, the marble at {[marble_at(s) for s in run.value]}: marble_maze.png")
    if "--pins" in args:
        board = Py2ver(show_on_screen(), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
