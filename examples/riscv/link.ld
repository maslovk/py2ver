/* riscv_computer.py's memory: one RAM from 0 holding the program, its
   constants and its variables (all loaded into it from the start), the
   stack at its top. */
MEMORY {
    RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 16K
}
ENTRY(_start)
SECTIONS {
    .text : { *(.text.start) *(.text*) *(.rodata*) *(.srodata*) } > RAM
    .data : { *(.data*) *(.sdata*) } > RAM
    .bss : { *(.bss*) *(.sbss*) *(COMMON) } > RAM
    _stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
