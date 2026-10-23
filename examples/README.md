# Examples

Each example builds its design and checks the hardware against the Python.
Run one from the repository's root:

```
python examples/basic.py
```

The function examples share a few options (see `_run.py`): `--syn` builds
the design in Quartus and reports its size and speed, `--board-sim` runs it
on a simulated DE0-Nano, and `--board` on a real one. The protocol examples
have their own (`--sim` simulates the Verilog too, `--pins` builds for the
board's pins); each file's first lines list them.

## Start here: functions

| example | what it shows |
|---|---|
| [basic.py](basic.py) | the smallest design: a function of two 8-bit numbers, its widths in attr |
| [branches.py](branches.py) | `if` / `elif` / `else`: a tiny ALU |
| [loop.py](loop.py) | a `for` loop, written out in hardware |
| [registers.py](registers.py) | `'type': 'reg'`: registered values, and the latency they add |
| [regression.py](regression.py) | many random inputs in one simulation |
| [lists.py](lists.py) | building and returning a list |
| [complex_numbers.py](complex_numbers.py) | complex arithmetic |
| [aes_sbox.py](aes_sbox.py) | a table lookup (the AES S-box) |

## Protocols and chips (generators, a `yield` a clock)

| example | what it shows |
|---|---|
| [uart.py](uart.py) | a UART transmitter and receiver, connected as an echo |
| [spi.py](spi.py) | an SPI master and slave on one bus |
| [i2c.py](i2c.py) | an I2C controller, tested against a model of a memory chip |
| [ws2812.py](ws2812.py) | a WS2812 (NeoPixel) LED driver |
| [jtag_echo.py](jtag_echo.py) | an echo over the JTAG cable: type on the PC, the bytes come back |
| [jtag_console.py](jtag_console.py) | a debug console: the design prints what it does |
| [accelerometer.py](accelerometer.py) | the DE0-Nano's ADXL345 accelerometer, read over I2C |
| [adc.py](adc.py) | the DE0-Nano's ADC, read over SPI |
| [lcd.py](lcd.py) | a test picture on an ILI9341 TFT display; the model saves it as a PNG |
| [mandelbrot.py](mandelbrot.py) | a Mandelbrot set generator, one iteration a clock, streaming RGB565 pixels for a display |
| [mandelbrot_lcd.py](mandelbrot_lcd.py) | the generator connected to the ILI9341 display: two blocks joined by a stream |
| [mandelbrot_zoom.py](mandelbrot_zoom.py) | a fractal show on the display, its colors cycling: the Mandelbrot set zooming in and out, or (KEY1) Julia sets morphing; 10 to 17 frames a second |
| [plasma.py](plasma.py) | a plasma on the display: four sine waves added at each pixel pick a rainbow color, flowing at 20 frames a second; its tables in memory blocks |
| [dreaming_digits.py](dreaming_digits.py) | a neural network on the FPGA drawing handwritten digits that melt into one another: a small VAE's decoder (trained on MNIST on first run, 8-bit weights in memory blocks) gliding from one clear digit to another (80 picked from where the network puts MNIST's own) and pausing on each, about 17 frames a second |
| [terminal.py](terminal.py) | a terminal on the display: what you type in juart-terminal on the PC (over the JTAG UART) appears in 30 x 24 characters, and scrolls; the font (py2ver/devices/font.py, X11's 8 x 13) a table, the characters a list in a memory block |
| [fire.py](fire.py) | Doom's fire on the display: flames rising and flickering, their 4,800 heats a list in a memory block; KEY1 puts the fire out and lights it again |
| [tunnel.py](tunnel.py) | flying down an endless checkered tube that twists and fades into the dark: each pixel's point on the wall from a table worked out before building (a quarter of the screen, mirrored) |

## Signal processing

| example | what it shows |
|---|---|
| [lowpass_stream.py](lowpass_stream.py) | a filter with its state in a register |
| [fir_delay_line.py](fir_delay_line.py) | a FIR filter with its delay line kept in registers |
| [fir_fixed_point.py](fir_fixed_point.py) | a float FIR model, built in fixed point |
| [auto_fixed_point.py](auto_fixed_point.py) | fixed-point formats chosen automatically |
| [math_tables.py](math_tables.py) | `math.tanh` as a table |
| [fir_folded.py](fir_folded.py) | a 32-tap FIR on 4 multipliers instead of 32 |
| [decimator.py](decimator.py) | a decimate-by-4 filter: four samples in, one out |
| [downconverter.py](downconverter.py) | a complex downconverter |
| [receiver.py](receiver.py) | a receiver chain built from plain functions |
| [histogram_ram.py](histogram_ram.py) | a histogram in block RAM |
| [sha256_pipeline.py](sha256_pipeline.py) | SHA-256 rounds, pipelined to meet 50 MHz |

## Neural networks

| example | what it shows |
|---|---|
| [xor_network.py](xor_network.py), [xor_hardware.py](xor_hardware.py) | a tiny numpy network, its training step in hardware |
| [digits_network.py](digits_network.py), [digits_hardware.py](digits_hardware.py), [digits_training.py](digits_training.py) | a digit-recognising numpy network, in hardware and trained in hardware |
| [cnn_network.py](cnn_network.py), [cnn_hardware.py](cnn_hardware.py), [cnn_training.py](cnn_training.py) | a convolutional network from scratch in numpy |
| [pytorch_mnist_cnn.py](pytorch_mnist_cnn.py), [pytorch_mnist_hardware.py](pytorch_mnist_hardware.py), [pytorch_mnist_training.py](pytorch_mnist_training.py) | a PyTorch MNIST network, as it is, in hardware |
| [tiny_gpt.py](tiny_gpt.py), [tiny_gpt_hardware.py](tiny_gpt_hardware.py) | a character-level Transformer |
| [mini_llama.py](mini_llama.py), [mini_llama_hardware.py](mini_llama_hardware.py) | a small Llama-style language model |
| [diffusion_digits.py](diffusion_digits.py), [diffusion_hardware.py](diffusion_hardware.py) | a diffusion model that draws digits |
