// riscv-tests' environment for riscv_computer.py's CPU (written for py2ver:
// riscv-tests asks each machine for its own riscv_test.h). A test starts at
// _start; passing prints "OK", failing prints "FAIL n" (n the failing case,
// TESTNUM) on the console, then the CPU waits there for good. The printing is
// written out where it's used: a test's own code and data are left as they
// are (fence_i writes into the code after its own).
//
// A test runs in machine mode, or (RVTEST_RV32S) in supervisor mode, as
// riscv-tests' own environment runs it: its stvec_handler, if it has one,
// given the traps it makes (a user ecall, a breakpoint, a misaligned fetch,
// page faults), and SSIP and STIP delegated. A trap that reaches machine mode
// unasked prints "TRAP" and its cause, as a letter (A: 0, B: 1, ...).
#ifndef PY2VER_RISCV_TEST_H
#define PY2VER_RISCV_TEST_H

#include "encoding.h"                     // (riscv-tests' own: the CSRs' bits, the causes)

#define RVTEST_RV32U
#define RVTEST_RV64U
#define RVTEST_RV32M
#define RVTEST_RV64M
#define RVTEST_RV32S .set py2ver_supervisor, 1;
#define RVTEST_RV64S RVTEST_RV32S

#define TESTNUM gp

#define PY2VER_PUT(c) li t1, c; sb t1, 0(t0);

#define RVTEST_CODE_BEGIN                                               \
        .section .text.start;                                           \
        .align 2;                                                       \
        .globl _start;                                                  \
_start:                                                                 \
        la t0, py2ver_trap;                                             \
        csrw mtvec, t0;                                                 \
        li TESTNUM, 0;                                                  \
        .ifdef py2ver_supervisor;                                       \
        .weak stvec_handler;                                            \
        la t0, stvec_handler;                                           \
        csrw stvec, t0;                                                 \
        li t0, (1 << 0) | (1 << 3) | (1 << 8) | (1 << 12) | (1 << 13) | (1 << 15); \
        csrw medeleg, t0;                                               \
        li t0, (1 << 1) | (1 << 5);                                     \
        csrw mideleg, t0;                                               \
        li t0, 1 << 11;                         /* MPP: supervisor */   \
        csrw mstatus, t0;                                               \
        la t0, py2ver_test;                                             \
        csrw mepc, t0;                                                  \
        mret;                                                           \
        .endif;                                                         \
        j py2ver_test;                                                  \
        .align 2;                                                       \
py2ver_trap:                                                            \
        li t0, 0x10000000;                                              \
        PY2VER_PUT('T') PY2VER_PUT('R') PY2VER_PUT('A') PY2VER_PUT('P') \
        PY2VER_PUT(' ')                                                 \
        csrr t1, mcause;                                                \
        andi t1, t1, 31;                                                \
        addi t1, t1, 'A';                                               \
        sb t1, 0(t0);                                                   \
        PY2VER_PUT('\n')                                                \
py2ver_trapped:                                                         \
        j py2ver_trapped;                                               \
py2ver_pass:                                                            \
        li t0, 0x10000000;                                              \
        PY2VER_PUT('O') PY2VER_PUT('K') PY2VER_PUT('\n')                \
py2ver_passed:                                                          \
        j py2ver_passed;                                                \
py2ver_fail:                                                            \
        li t0, 0x10000000;                                              \
        PY2VER_PUT('F') PY2VER_PUT('A') PY2VER_PUT('I') PY2VER_PUT('L') \
        PY2VER_PUT(' ')                                                 \
        li t1, '0' - 1;                         /* tens, then ones */   \
py2ver_tens:                                                            \
        addi t1, t1, 1;                                                 \
        addi TESTNUM, TESTNUM, -10;                                     \
        bgez TESTNUM, py2ver_tens;                                      \
        addi TESTNUM, TESTNUM, 10;                                      \
        li t2, '0';                                                     \
        beq t1, t2, py2ver_ones;                                        \
        sb t1, 0(t0);                                                   \
py2ver_ones:                                                            \
        addi TESTNUM, TESTNUM, '0';                                     \
        sb TESTNUM, 0(t0);                                              \
        PY2VER_PUT('\n')                                                \
py2ver_failed:                                                          \
        j py2ver_failed;                                                \
py2ver_test:

#define RVTEST_CODE_END                                                 \
        unimp;

// (the printing: in RVTEST_CODE_BEGIN, before the test's code, where
// fence_i's writes into the code after its own don't reach; named labels,
// not 1: or 2:, so that a test's own "1f" finds its own)
#define RVTEST_PASS                                                     \
        j py2ver_pass;

#define RVTEST_FAIL                                                     \
        j py2ver_fail;

#define RVTEST_DATA_BEGIN                                               \
        .data;                                                          \
        .align 4;

#define RVTEST_DATA_END

#endif
