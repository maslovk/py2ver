# Breakout on the ILI9341 display, steered by tilting the board: the board's
# accelerometer moves the paddle. KEY1 serves the ball, and starts a new game
# once the balls have run out.
#
# Three processes, joined:
#
#   tilt_sensor --tilt--> breakout --color/valid/ready--> ili9341_display
#
# tilt_sensor reads the board's ADXL345 over I2C, 500 times a second, and
# keeps its latest X on `tilt` (1 g is 256); it runs alongside the game, so
# the game never waits for the sensor. breakout moves the paddle by the tilt,
# then the ball (off the walls, the paddle and the bricks), each frame; then it
# draws the frame, a pixel at a time: the score and the balls left in the top
# row (py2ver/devices/font.py's font), the bricks, the paddle and the ball.
#
#   python examples/breakout.py          # a small game, tilted one way: test it, save breakout.png
#   python examples/breakout.py --sim    # also simulate the generated Verilog
#   python examples/breakout.py --pins   # build it for the board (wiring: mandelbrot_lcd.py)
import functools
import os
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
    """The ADXL345's X, read every period (2 ms), kept on port.tilt: a signed
    number, 1 g (the board on its edge) about 256."""
    port.scl = 1
    port.sda = 1
    port.accel_cs_n = 1                  # (the ADXL345 talks I2C while its CS is high)
    port.tilt = 0
    yield 50_000                         # (the bus settles: 1 ms)
    yield from adxl345_start(port, bus)
    while True:
        x, y, z, nack = yield from adxl345_read_xyz(port, bus)
        if not nack:
            port.tilt = x - 65536 if x >= 32768 else x       # (16-bit two's complement)
        yield period


SENSOR_ATTR = {"scl": {"open_drain": True}, "sda": {"open_drain": True},
               "tilt": {"width": 16, "signed": 1}}

# ----------------------------- the game -----------------------------

BRICK_W, BRICK_H = 32, 16                # a brick's place: 30 x 14 of brick, then a gap
PADDLE_W, PADDLE_H = 48, 6
BALL = 6                                 # the ball's side
BALLS = 3                                # balls a game


class Field(NamedTuple):
    """Where things are on a screen of width x height pixels."""
    width: int
    height: int
    across: int                          # bricks across, and rows of them
    rows: int
    left: int                            # the bricks' left and top edges
    top: int
    paddle_y: int                        # the paddle's top
    message_y: int                       # the top of the GAME OVER line


def field(width=WIDTH, height=HEIGHT, rows=6):
    """The Field for a screen: as many bricks across as fit, `rows` of them
    below the score; the paddle near the bottom."""
    across = (width - 16) // BRICK_W
    return Field(width, height, across, rows, left=(width - across * BRICK_W) // 2,
                 top=F.HEIGHT + 2 * BRICK_H, paddle_y=height - 2 * BRICK_H,
                 message_y=height // 2)


FIELD = field()
PAPER = rgb565(10, 10, 30)               # the background, and the colors of things
INK = rgb565(230, 230, 230)
PADDLE = rgb565(240, 240, 255)
BALL_COLOR = rgb565(255, 220, 60)
ROW_COLORS = tuple(rgb565(*c) for c in ((230, 60, 60), (240, 140, 40), (240, 220, 60),
                                        (80, 200, 80), (60, 140, 240), (150, 80, 220)))
SCORE_TEXT = b"SCORE "
BALLS_TEXT = b"BALLS "
GAME_OVER = b"GAME OVER - KEY1"


def breakout(port, place=FIELD, speed=4, steer=4):
    """The game, frame after frame, sending each pixel's color.

    speed: how far the ball goes up or down each frame, in pixels.
    steer: the paddle moves tilt >> steer pixels a frame (1 g: 16 with 4).
    Input: tilt (from tilt_sensor), key (0 while pressed: serve, or a new
    game), ready. Output stream: color (RGB565), valid (protocol.stream_send).
    """
    port.valid = 0
    bricks = [1] * (place.across * place.rows)   # each brick's place: 1 while it's there
    left = place.across * place.rows             # the bricks left
    score = [0] * 4                              # the score's digits, the ones last
    balls = BALLS
    over = 0                                     # the balls run out: GAME OVER
    serving = 1                                  # the ball on the paddle, waiting for KEY1
    paddle = (place.width - PADDLE_W) // 2
    x = 0                                        # the ball's top left corner, and its
    y = 0                                        # move each frame
    dx = 0
    dy = 0
    was = 1
    while True:
        # ---- the frame's move ----
        key = port.key
        pressed = was == 1 and key == 0          # (pressed just now)
        was = key
        paddle = min(max(paddle + (port.tilt >> steer), 0), place.width - PADDLE_W)
        yield
        if over:
            if pressed:                          # a new game
                for k in range(len(bricks)):
                    bricks[k] = 1
                    yield
                for k in range(len(score)):
                    score[k] = 0
                    yield
                left = len(bricks)
                balls = BALLS
                over = 0
                serving = 1
        elif serving:                            # the ball rides on the paddle
            x = paddle + (PADDLE_W - BALL) // 2
            y = place.paddle_y - BALL
            if pressed:
                serving = 0
                dx = speed // 2
                dy = -speed
        else:
            x = x + dx
            y = y + dy
            yield
            # Off the walls and the ceiling (the row of the score)
            if x < 0:
                x = -x
                dx = -dx
            if x > place.width - BALL:
                x = 2 * (place.width - BALL) - x
                dx = -dx
            if y < F.HEIGHT:
                y = 2 * F.HEIGHT - y
                dy = -dy
            yield
            # Off the paddle: further from its middle, more to the side
            if dy > 0 and y + BALL >= place.paddle_y and y + BALL < place.paddle_y + PADDLE_H + speed \
                    and x + BALL > paddle and x < paddle + PADDLE_W:
                y = place.paddle_y - BALL
                dy = -dy
                dx = max(-speed, min(speed, ((x + BALL // 2) - (paddle + PADDLE_W // 2)) >> 3))
            yield
            # Into a brick (the one the ball's middle is in): gone, and back the ball goes
            cx = x + BALL // 2
            cy = y + BALL // 2
            if cy >= place.top and cy < place.top + place.rows * BRICK_H \
                    and cx >= place.left and cx < place.left + place.across * BRICK_W:
                k = ((cy - place.top) >> 4) * place.across + ((cx - place.left) >> 5)
                if bricks[k]:
                    bricks[k] = 0
                    dy = -dy
                    left = left - 1
                    # (the score up one, a 9 carried to the next digit: 0019, 0020)
                    if score[3] < 9:
                        score[3] = score[3] + 1
                    else:
                        score[3] = 0
                        if score[2] < 9:
                            score[2] = score[2] + 1
                        else:
                            score[2] = 0
                            if score[1] < 9:
                                score[1] = score[1] + 1
                            else:
                                score[1] = 0
                                score[0] = score[0] + 1 if score[0] < 9 else 0
            yield
            if y > place.height:                 # missed: the next ball, or the end
                y = place.height                 # (out of sight, until it is served again)
                if balls > 1:
                    balls = balls - 1
                    serving = 1
                else:
                    balls = 0
                    over = 1
            if left == 0:                        # all gone: a new wall of them
                for k in range(len(bricks)):
                    bricks[k] = 1
                    yield
                left = len(bricks)
                serving = 1
        # ---- the picture ----
        columns = place.width // F.WIDTH
        for row in range(place.height):
            message = over and row >= place.message_y and row < place.message_y + F.HEIGHT
            if row < F.HEIGHT or message:
                # A row of text: the score and the balls left, or GAME OVER
                line = row if row < F.HEIGHT else row - place.message_y
                start = (columns - len(GAME_OVER)) // 2
                for k in range(columns):
                    c = 32
                    if message:
                        if k >= start and k < start + len(GAME_OVER):
                            c = GAME_OVER[k - start]
                    elif k < len(SCORE_TEXT):
                        c = SCORE_TEXT[k]
                    elif k < len(SCORE_TEXT) + 4:
                        c = 48 + score[k - len(SCORE_TEXT)]
                    elif k >= columns - len(BALLS_TEXT) - 1 and k < columns - 1:
                        c = BALLS_TEXT[k - (columns - len(BALLS_TEXT) - 1)]
                    elif k == columns - 1:
                        c = 48 + balls
                    bits = F.FONT[(c - F.FIRST) * F.HEIGHT + line]
                    yield
                    for b in range(F.WIDTH):
                        lit = (bits >> (F.WIDTH - 1 - b)) & 1
                        yield from stream_send(port, INK if lit else PAPER, data="color")
            else:
                # A row of the field: bricks, paddle, ball
                in_wall = row >= place.top and row < place.top + place.rows * BRICK_H \
                    and ((row - place.top) & (BRICK_H - 1)) < BRICK_H - 2
                brick_row = (row - place.top) >> 4
                on_paddle = row >= place.paddle_y and row < place.paddle_y + PADDLE_H
                on_ball = not over and row >= y and row < y + BALL
                for col in range(place.width):
                    color = PAPER
                    if in_wall and col >= place.left and col < place.left + place.across * BRICK_W \
                            and ((col - place.left) & (BRICK_W - 1)) < BRICK_W - 2 \
                            and bricks[brick_row * place.across + ((col - place.left) >> 5)]:
                        color = ROW_COLORS[brick_row]
                    if on_paddle and col >= paddle and col < paddle + PADDLE_W:
                        color = PADDLE
                    if on_ball and col >= x and col < x + BALL:
                        color = BALL_COLOR
                    yield from stream_send(port, color, data="color")


GAME_ATTR = {"ready": 1, "key": 1, "tilt": {"width": 16, "signed": 1}}

# SCK at 25 MHz, as in mandelbrot_zoom.py (if the picture is noisy: SPI(2, ...))
LCD = ILI9341(spi=SPI(1, mode=0, miso=""))

# On the board: the display's pins, KEY1, and the accelerometer's I2C bus and CS
PINS = dict(LCD_PINS, key="KEY[1]", scl="I2C_SCLK", sda="I2C_SDAT", accel_cs_n="G_SENSOR_CS_N")


def show_on_screen(lcd=LCD, place=FIELD, bus=BUS, period=100_000, speed=4):
    """The sensor steering the game, the game feeding the display."""
    sensor = functools.partial(tilt_sensor, bus=bus, period=period)
    game = functools.partial(breakout, place=place, speed=speed)
    display = functools.partial(ili9341_display, lcd=lcd, width=place.width, height=place.height)
    return connect(sensor=(sensor, SENSOR_ATTR), game=(game, GAME_ATTR),
                   display=(display, {"color": 16, "valid": 1}),
                   name="tilt_breakout",
                   wires={"game.tilt": "sensor.tilt", "display.color": "game.color",
                          "display.valid": "game.valid", "game.ready": "display.ready"})


def frames(screen, count, place, tilt=0, press_at=(1,)):
    """A testbench: the board tilted by `tilt` (1 g is 256), KEY1 pressed in
    the frames press_at; run until the display has received `count` frames,
    a copy of the picture after each."""
    from py2ver.devices.adxl345 import ADDRESS, registers
    from py2ver.devices.i2c_target import i2c_memory
    from py2ver.protocol import alongside
    shots = []

    def watch(dut):
        while len(shots) < count:
            dut.key = 0 if len(shots) in press_at else 1     # (the board's keys: 0 pressed)
            if screen.written >= (len(shots) + 1) * place.width * place.height:
                shots.append([list(r) for r in screen.pixels])
            yield

    def tb(dut):
        yield from alongside(watch(dut), screen.run(dut),
                             i2c_memory(dut, registers(x=tilt), 1_000_000, ADDRESS))
        return shots
    return tb


if __name__ == "__main__":
    from mandelbrot_zoom import film_strip
    from py2ver import Py2ver
    from py2ver.devices.ili9341 import ILI9341Model, rgb888
    from py2ver.devices.spi_target import save_png

    args = sys.argv[1:]
    # A small game for the simulation: 144 x 120 pixels (4 bricks across, 2
    # rows), the ball fast; the board tilted to the right; KEY1 serves in frame 2
    small = field(144, 120, rows=2)
    count = 6
    quick = ILI9341(spi=SPI(1, mode=0, miso=""), wake_clocks=50)
    screen = ILI9341Model(quick, width=small.width, height=small.height)
    p = Py2ver(show_on_screen(quick, small, bus=I2C(q=4), period=2_000, speed=12))
    run = p.check(frames(screen, count, small, tilt=96), verilog="--sim" in args)
    save_png("breakout.png", film_strip(run.value, scale=3), rgb888)
    print(f"{count} frames of {small.width} x {small.height} in {len(run.inputs):,} clock "
          f"cycles: breakout.png")
    if "--pins" in args:
        board = Py2ver(show_on_screen(), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
