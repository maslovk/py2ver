"""
Drivers for chips. Each module has the chip's register map, the steps that
talk to it, and a model of the chip to test your design against:

    adxl345      ADXL345 accelerometer (I2C; on the DE0-Nano at 0x1D)
    adc128s022   ADC128S022 analog-to-digital converter, 8 inputs of 12 bits (SPI)
    eeprom24     24LC02-style EEPROM, 256 bytes (I2C; on the DE0-Nano at 0x50)
    i2c_target   the model the I2C chips' models are built on

A driver step takes the port, then the bus's settings (an I2C or SPI record
from py2ver.protocol):

    from py2ver.devices.adxl345 import adxl345_read_xyz, adxl345_start
    from py2ver.protocol import I2C

    def tilt(port, bus=I2C()):          # 100 kHz at 50 MHz, on pins scl and sda
        yield from adxl345_start(port, bus)
        while True:
            x, y, z, nack = yield from adxl345_read_xyz(port, bus)
            ...
"""
