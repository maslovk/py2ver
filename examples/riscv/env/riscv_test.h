// riscv-tests' environment for riscv_computer.py's CPU (written for py2ver:
// riscv-tests asks each machine for its own riscv_test.h). A test starts at
// _start; passing prints "OK", failing prints "FAIL n" (n the failing case,
// TESTNUM) on the console, then the CPU waits there for good. The printing is
// written out where it's used: a test's own code and data are left as they
// are (fence_i writes into the code after its own).
#ifndef PY2VER_RISCV_TEST_H
#define PY2VER_RISCV_TEST_H

#define RVTEST_RV32U
#define RVTEST_RV64U
#define RVTEST_RV32M
#define RVTEST_RV64M

#define TESTNUM gp

#define RVTEST_CODE_BEGIN                                               \
        .section .text.start;                                           \
        .align 2;                                                       \
        .globl _start;                                                  \
_start:                                                                 \
        li TESTNUM, 0;

#define RVTEST_CODE_END                                                 \
        unimp;

#define PY2VER_PUT(c) li t1, c; sb t1, 0(t0);

// (named labels, not 1: or 2:: a test's own "1f" must find its own)
#define RVTEST_PASS                                                     \
        li t0, 0x10000000;                                              \
        PY2VER_PUT('O') PY2VER_PUT('K') PY2VER_PUT('\n')                \
py2ver_passed:                                                          \
        j py2ver_passed;

#define RVTEST_FAIL                                                     \
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
        j py2ver_failed;

#define RVTEST_DATA_BEGIN                                               \
        .data;                                                          \
        .align 4;

#define RVTEST_DATA_END

#endif
