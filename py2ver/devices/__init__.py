"""
Drivers of chips, each a module: the chip's registers, the steps that work it
(generators, used with `yield from` in a process: see process.py), and a model
of it for testbenches, on the bus's pins:

    adxl345      the ADXL345 accelerometer (I2C; on the DE0-Nano at 0x1D)
    adc128s022   the ADC128S022 ADC, 8 inputs of 12 bits (SPI mode 3)
    eeprom24     a 24LC02-like I2C EEPROM, 256 bytes (on the DE0-Nano at 0x50)
    i2c_target   the I2C models' common part: a chip of registers on the bus

A bus's settings are a record (py2ver.protocol.I2C, SPI), given to each step
after the port:

    from py2ver.devices.adxl345 import adxl345_read_xyz, adxl345_start
    from py2ver.protocol import I2C

    BUS = I2C(q=125)                                  # 100 kHz at 50 MHz

    def tilt(port, bus=BUS):
        yield from adxl345_start(port, bus)
        while True:
            x, y, z, nack = yield from adxl345_read_xyz(port, bus)
            ...
"""
