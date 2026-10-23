/* riscv_computer.py's memory: the program and its constants in ROM from 0,
   its variables and stack in RAM from 0x10000. (Variables start at 0: there
   is nothing to copy a variable's first value from ROM, so give them values
   in main.) */
MEMORY {
    ROM (rx)  : ORIGIN = 0x00000000, LENGTH = 16K
    RAM (rw)  : ORIGIN = 0x00010000, LENGTH = 4K
}
ENTRY(_start)
SECTIONS {
    .text : { *(.text.start) *(.text*) *(.rodata*) *(.srodata*) } > ROM
    .bss (NOLOAD) : { *(.data*) *(.sdata*) *(.bss*) *(.sbss*) *(COMMON) } > RAM
    _stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
