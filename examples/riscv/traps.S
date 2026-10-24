# A check of the CPU's machine mode: the CSRs, traps (ecall, ebreak, an
# illegal instruction, misaligned atomics), the timer interrupt and mret; of
# user mode: mret into it, an ecall from it; and of loads and stores across a
# word's end. Prints "OK", or "FAIL n" for the first check that went wrong.

    .equ CONSOLE, 0x10000000
    .equ MTIMECMP, 0x11004000

    .macro check reg, value, n
    li   t6, \value
    li   gp, \n
    bne  \reg, t6, fail
    .endm

    .section .text.start
    .globl _start
_start:
    la   t0, handler
    csrw mtvec, t0

    # ---- the CSR instructions ----
    li   t0, 0x1234
    csrw mscratch, t0
    csrr t1, mscratch
    check t1, 0x1234, 1
    li   t0, 0xF0
    csrrs t1, mscratch, t0
    check t1, 0x1234, 2
    csrr t1, mscratch
    check t1, 0x12F4, 3
    csrrc t1, mscratch, t0
    csrr t1, mscratch
    check t1, 0x1204, 4
    csrrwi t1, mscratch, 5
    check t1, 0x1204, 5
    csrrsi t1, mscratch, 2
    csrr t1, mscratch
    check t1, 7, 6
    csrrci t1, mscratch, 1
    csrr t1, mscratch
    check t1, 6, 7
    csrr t1, misa
    check t1, 0x40001101, 8          # (32 bits; I, M, A)
    csrr t1, mhartid
    check t1, 0, 9

    # ---- traps: the handler keeps mcause in s2, mtval in s3, and goes on
    # after the instruction ----
    ecall
    check s2, 11, 10                 # (an ecall from machine mode)
    ebreak
    check s2, 3, 11
    .word 0                          # (all zeros: an illegal instruction)
    check s2, 2, 12
    li   t0, 2
    amoadd.w t1, zero, (t0)          # (an atomic at an address not a multiple of 4)
    check s2, 6, 13
    check s3, 2, 14                  # (mtval: the address)
    li   t0, 1
the_amo:
    amoswap.w t1, t1, (t0)
    check s2, 6, 15
    csrr t1, mepc                    # (mepc: the amoswap's address, which the handler moved on by 4)
    la   t2, the_amo
    addi t2, t2, 4
    li   gp, 16
    bne  t1, t2, fail

    # ---- the timer: mtimecmp 0, so mtime (from 0) is past it at once ----
    li   s2, 0
    li   t0, MTIMECMP
    sw   zero, 0(t0)
    sw   zero, 4(t0)
    li   t0, 0x80
    csrw mie, t0                     # (the timer's interrupt)
    csrsi mstatus, 8                 # (interrupts on: it comes now)
    nop
    check s2, 0x80000007, 17         # (the timer's interrupt)
    csrr t1, mstatus
    andi t1, t1, 8
    check t1, 8, 18                  # (on again after mret)

    # ---- user mode: mret to it (MPP 0); an ecall there is a system call
    # (mcause 8), and the trap saves user mode in MPP; an ebreak, whose
    # handler sets MPP to machine mode, comes back to machine mode ----
    csrci mstatus, 8
    li   t0, 0x1800
    csrc mstatus, t0                 # (MPP: user)
    csrr t1, mstatus
    and  t1, t1, t0
    check t1, 0, 19
    la   t0, in_user
    csrw mepc, t0
    mret
in_user:
    ecall
    check s2, 8, 20                  # (an ecall from user mode)
    li   t0, 0x1800
    and  t1, s4, t0
    check t1, 0, 21                  # (MPP in the trap: user)
    ebreak
    ecall
    check s2, 11, 22                 # (machine mode again)
    li   t0, 0x1800
    and  t1, s4, t0
    check t1, 0x1800, 23             # (MPP in the trap: machine)

    # ---- loads and stores at any address, across a word's end too (bytes
    # 0 to 3: 44 33 22 11, 4 to 7: 88 77 66 55) ----
    li   t0, 0x3000
    li   t1, 0x11223344
    sw   t1, 0(t0)
    li   t1, 0x55667788
    sw   t1, 4(t0)
    lw   t1, 1(t0)
    check t1, 0x88112233, 24
    lhu  t1, 3(t0)
    check t1, 0x8811, 25
    lh   t1, 3(t0)
    check t1, 0xFFFF8811, 26
    li   t1, 0xAABBCCDD
    sw   t1, 3(t0)                   # (bytes 3 to 6)
    lw   t1, 0(t0)
    check t1, 0xDD223344, 27
    lw   t1, 4(t0)
    check t1, 0x55AABBCC, 28
    li   t1, 0xEEFF
    sh   t1, 7(t0)                   # (bytes 7 and 8)
    lw   t1, 4(t0)
    check t1, 0xFFAABBCC, 29
    lw   t1, 8(t0)
    check t1, 0xEE, 30

    la   a0, ok
    call print
    j    stop

handler:
    csrr s2, mcause
    csrr s3, mtval
    csrr s4, mstatus
    li   t5, 3                       # (ebreak: back to machine mode)
    bne  s2, t5, 3f
    li   t5, 0x1800
    csrs mstatus, t5
3:  bltz s2, 1f                      # (an interrupt: go back to the same place)
    csrr t5, mepc                    # (a trap: on to the next instruction)
    addi t5, t5, 4
    csrw mepc, t5
    mret
1:  li   t5, MTIMECMP                # (the timer: put it off for good)
    li   t4, -1
    sw   t4, 4(t5)
    sw   t4, 0(t5)
    mret

fail:
    la   a0, failed
    call print
    li   t0, CONSOLE
    li   t1, '0' - 1
1:  addi t1, t1, 1
    addi gp, gp, -10
    bgez gp, 1b
    addi gp, gp, 10
    li   t2, '0'
    beq  t1, t2, 2f
    sb   t1, 0(t0)
2:  addi gp, gp, '0'
    sb   gp, 0(t0)
    li   t1, '\n'
    sb   t1, 0(t0)
stop:
    j    stop

print:
    li   t0, CONSOLE
1:  lbu  t1, 0(a0)
    beqz t1, 2f
    sb   t1, 0(t0)
    addi a0, a0, 1
    j    1b
2:  ret

ok:
    .string "OK\n"
failed:
    .string "FAIL "
