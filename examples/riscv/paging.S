# A check of the CPU's paging and its TLB, in supervisor mode with page tables
# of its own: a page read through its mapping; mapped elsewhere (the SDRAM),
# sfence.vma, read again: the new page's word, not the TLB's old one;
# unmapped, sfence.vma: a page fault. Prints "OK", or "FAIL n" for the first
# check that went wrong. (It needs the SDRAM: cpu_with_sdram, computer.)
#
# The pages: 0 this program (R X), 1 the data (R W), 2 the root table, 3 the
# leaf table (R W, to be changed); the UART's 4 MB its own (R W).

    .equ CONSOLE, 0x10000000
    .equ ROOT, 0x2000
    .equ LEAF, 0x3000
    .equ DATA, 0x1000
    .equ SDRAM, 0x80000000
    .equ CODE_PTE, 0x4B              # (V R X A)
    .equ DATA_PTE, 0xC7              # (V R W A D)

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
    # (the words: one in the RAM's page 1, one in the SDRAM's first page)
    li   t0, DATA
    li   t1, 0xAAAA0001
    sw   t1, 0(t0)
    li   t0, SDRAM
    li   t1, 0xBBBB0002
    sw   t1, 0(t0)
    # (the root: its entry 0 the leaf table, its entry 0x40 the UART's megapage)
    li   t0, ROOT
    li   t1, (LEAF >> 12 << 10) | 1
    sw   t1, 0(t0)
    li   t1, (CONSOLE >> 12 << 10) | DATA_PTE
    sw   t1, (CONSOLE >> 22 << 2)(t0)
    # (the leaf: pages 0, 1, 3)
    li   t0, LEAF
    li   t1, (0 << 10) | CODE_PTE
    sw   t1, 0(t0)
    li   t1, (DATA >> 12 << 10) | DATA_PTE
    sw   t1, 4(t0)
    li   t1, (LEAF >> 12 << 10) | DATA_PTE
    sw   t1, 12(t0)
    li   t0, 0x80000000 | (ROOT >> 12)       # (satp: Sv32, the root)
    csrw satp, t0
    # (to supervisor mode)
    li   t0, 0x1800
    csrc mstatus, t0
    li   t0, 0x0800
    csrs mstatus, t0
    la   t0, in_supervisor
    csrw mepc, t0
    mret

in_supervisor:
    li   t0, DATA
    lw   t1, 0(t0)
    check t1, 0xAAAA0001, 1          # (through the first mapping)
    lw   t1, 0(t0)
    check t1, 0xAAAA0001, 2          # (again: from the TLB now)
    # (page 1 mapped to the SDRAM's first page instead)
    li   t2, LEAF
    li   t3, (SDRAM >> 12 << 10) | DATA_PTE
    sw   t3, 4(t2)
    sfence.vma
    lw   t1, 0(t0)
    check t1, 0xBBBB0002, 3          # (the new page's word)
    li   t1, 0xCCCC0003
    sw   t1, 4(t0)
    lw   t1, 4(t0)
    check t1, 0xCCCC0003, 4          # (written and read there)
    # (page 1 unmapped)
    sw   zero, 4(t2)
    sfence.vma
    li   s2, 0
    lw   t1, 0(t0)
    check s2, 13, 5                  # (a load's page fault)
    check s3, DATA, 6                # (mtval: the address)

    la   a0, ok
    call print
    j    stop

handler:                             # (machine mode's: the cause kept, on past it)
    csrr s2, mcause
    csrr s3, mtval
    csrr t5, mepc
    addi t5, t5, 4
    csrw mepc, t5
    mret

fail:
    la   a0, failed
    call print
    li   t0, CONSOLE
    addi gp, gp, '0'
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
