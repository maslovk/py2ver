# A check of the CPU: every RV32I instruction, each result against what it
# should be. Prints "OK" on the console, or "FAIL n" for the first check that
# went wrong (n its number below), then stops.

    .equ CONSOLE, 0x10000000
    .equ RAM, 0x00010000

# check reg, value, n: reg should be value; if not, check n failed
    .macro check reg, value, n
    li   t6, \value
    li   gp, \n
    bne  \reg, t6, fail
    .endm

    .section .text.start
    .globl _start
_start:
    # ---- arithmetic and logic ----
    li   t0, 5
    li   t1, 7
    add  t2, t0, t1
    check t2, 12, 1
    sub  t2, t0, t1
    check t2, -2, 2
    addi t2, t0, -9
    check t2, -4, 3
    li   t0, -8
    srai t2, t0, 1
    check t2, -4, 4
    srli t2, t0, 28
    check t2, 15, 5
    slli t2, t1, 4
    check t2, 112, 6
    sll  t2, t1, t1
    check t2, 896, 7
    li   t3, 3
    srl  t2, t0, t3
    check t2, 0x1FFFFFFF, 8
    sra  t2, t0, t3
    check t2, -1, 9
    slt  t2, t0, t1
    check t2, 1, 10
    sltu t2, t0, t1
    check t2, 0, 11
    slti t2, t0, -7
    check t2, 1, 12
    sltiu t2, t1, 8
    check t2, 1, 13
    li   t0, 12
    li   t1, 10
    xor  t2, t0, t1
    check t2, 6, 14
    or   t2, t0, t1
    check t2, 14, 15
    and  t2, t0, t1
    check t2, 8, 16
    xori t2, t0, -1
    check t2, -13, 17
    ori  t2, t0, 1
    check t2, 13, 18
    andi t2, t0, 4
    check t2, 4, 19
    lui  t2, 0x12345
    check t2, 0x12345000, 20
here:
    auipc t2, 0
    la   t3, here
    li   gp, 21
    bne  t2, t3, fail

    # ---- branches: each taken and not taken ----
    li   t0, -1
    li   t1, 1
    li   gp, 22
    beq  t0, t1, fail
    beq  t0, t0, 1f
    j    fail
1:  li   gp, 23
    bne  t0, t0, fail
    bne  t0, t1, 1f
    j    fail
1:  li   gp, 24
    blt  t1, t0, fail
    blt  t0, t1, 1f
    j    fail
1:  li   gp, 25
    bge  t0, t1, fail
    bge  t1, t0, 1f
    j    fail
1:  li   gp, 26
    bltu t0, t1, fail
    bltu t1, t0, 1f
    j    fail
1:  li   gp, 27
    bgeu t1, t0, fail
    bgeu t0, t1, 1f
    j    fail
1:
    # ---- jumps and calls ----
    li   a1, 0
    jal  ra, set_a1
    check a1, 42, 28
    li   a1, 0
    la   t0, set_a1
    jalr ra, 0(t0)
    check a1, 42, 29

    # ---- memory: words, halves, bytes, RAM and ROM ----
    li   s0, RAM
    li   t0, 0x12345678
    sw   t0, 0(s0)
    lw   t1, 0(s0)
    check t1, 0x12345678, 30
    lb   t1, 0(s0)
    check t1, 0x78, 31
    lbu  t1, 3(s0)
    check t1, 0x12, 32
    li   t0, 0x80
    sb   t0, 1(s0)
    lb   t1, 1(s0)
    check t1, -128, 33
    lbu  t1, 1(s0)
    check t1, 128, 34
    lw   t1, 0(s0)
    check t1, 0x12348078, 35
    li   t0, 0xBEEF
    sh   t0, 6(s0)
    lh   t1, 6(s0)
    check t1, -16657, 36
    lhu  t1, 6(s0)
    check t1, 0xBEEF, 37
    lw   t1, 4(s0)
    check t1, 0xBEEF0000, 38
    la   t0, rom_word
    lw   t1, 0(t0)
    check t1, 0xCAFEF00D, 39
    lbu  t1, 1(t0)
    check t1, 0xF0, 40
    lh   t1, 2(t0)
    check t1, -13570, 41

    # ---- all well ----
    la   a0, ok
    call print
    j    stop

fail:
    la   a0, failed
    call print
    li   t0, CONSOLE
    li   t1, '0' - 1                 # the tens digit of gp: '0' and up
1:  addi t1, t1, 1
    addi gp, gp, -10
    bgez gp, 1b
    addi gp, gp, 10
    li   t2, '0'
    beq  t1, t2, 2f                 # (no leading 0)
    sb   t1, 0(t0)
2:  addi gp, gp, '0'
    sb   gp, 0(t0)
    li   t1, '\n'
    sb   t1, 0(t0)
stop:
    j    stop

# print a0: the string there, to the console
print:
    li   t0, CONSOLE
1:  lbu  t1, 0(a0)
    beqz t1, 2f
    sb   t1, 0(t0)
    addi a0, a0, 1
    j    1b
2:  ret

set_a1:
    li   a1, 42
    ret

    .align 2
rom_word:
    .word 0xCAFEF00D
ok:
    .string "OK\n"
failed:
    .string "FAIL "
