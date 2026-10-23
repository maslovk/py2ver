# Where a C program starts on riscv_computer.py's CPU: the stack at the top of
# RAM, main called, then a loop forever when it returns.
    .section .text.start
    .globl _start
_start:
    la   sp, _stack_top
    call main
1:  j    1b
