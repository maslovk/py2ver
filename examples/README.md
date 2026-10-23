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
