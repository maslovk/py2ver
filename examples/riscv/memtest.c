// A test of the SDRAM at 0x8000_0000 from riscv_computer.py's CPU. First it
// runs a few instructions stored in the SDRAM. Then, pass after pass, it
// writes every word, changes part of each with a byte or halfword store, and
// reads them back: with a load, or for every other word an atomic add, which
// must find the word as it was and leave it one more. It prints how many
// words were wrong; the LEDs count the passes.
//
// WORDS (all 8M by default) and STRIDE (1) can be given with -D: an odd stride
// reaches every word once, so a few words far apart test every bank.

#define UART_DATA (*(volatile unsigned char *)0x10000000)
#define LEDS      (*(volatile unsigned *)0x10001000)
#define SDRAM     ((volatile unsigned *)0x80000000)
#define ALL_WORDS (8u << 20)                 // 32 MB

#ifndef WORDS
#define WORDS ALL_WORDS
#endif
#ifndef STRIDE
#define STRIDE 1u
#endif
#ifndef PASSES
#define PASSES 0                             // 0: forever
#endif

static void print(const char *s) {
    while (*s)
        UART_DATA = *s++;
}

static void print_number(unsigned n) {
    char digits[10];
    int k = 0;
    do {
        digits[k++] = '0' + n % 10;
        n /= 10;
    } while (n);
    while (k)
        UART_DATA = digits[--k];
}

// The next of a sequence of numbers that looks random (xorshift: never 0)
static unsigned next_random(unsigned x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// What word k holds once part of it has been changed: a byte (k % 8 below 4)
// or a halfword, inverted
static unsigned changed(unsigned x, unsigned k) {
    if ((k & 7) < 4)
        return x ^ (0xFFu << 8 * (k & 3));
    return x ^ (0xFFFFu << 16 * (k & 1));
}

// A function's instructions put into the SDRAM, then called there
static int run_from_sdram(void) {
    volatile unsigned *code = SDRAM + ALL_WORDS - 2;
    code[0] = 0x02a00513;                    // li a0, 42
    code[1] = 0x00008067;                    // ret
    __asm__ volatile ("fence.i");
    int (*function)(void) = (int (*)(void))code;
    return function();
}

int main(void) {
    print("SDRAM test\n");
    print(run_from_sdram() == 42 ? "running from the SDRAM: ok\n" : "running from the SDRAM: wrong\n");
    volatile unsigned char *bytes = (volatile unsigned char *)SDRAM;
    volatile unsigned short *halves = (volatile unsigned short *)SDRAM;
    unsigned seed = 1;
    for (unsigned pass = 1; PASSES == 0 || pass <= PASSES; pass++) {
        unsigned x = seed, a = 0;
        for (unsigned k = 0; k < WORDS; k++) {
            x = next_random(x);
            SDRAM[a] = x;
            a = (a + STRIDE) & (ALL_WORDS - 1);
        }
        x = seed, a = 0;
        for (unsigned k = 0; k < WORDS; k++) {
            x = next_random(x);
            if ((k & 7) < 4)
                bytes[4 * a + (k & 3)] = ~(x >> 8 * (k & 3));
            else
                halves[2 * a + (k & 1)] = ~(x >> 16 * (k & 1));
            a = (a + STRIDE) & (ALL_WORDS - 1);
        }
        unsigned errors = 0;
        x = seed, a = 0;
        for (unsigned k = 0; k < WORDS; k++) {
            x = next_random(x);
            unsigned expected = changed(x, k);
            if (k & 1) {
                unsigned got = __atomic_fetch_add(&SDRAM[a], 1, __ATOMIC_SEQ_CST);
                if (got != expected || SDRAM[a] != expected + 1)
                    errors++;
            } else if (SDRAM[a] != expected) {
                errors++;
            }
            a = (a + STRIDE) & (ALL_WORDS - 1);
        }
        seed = x;
        LEDS = pass;
        print("pass ");
        print_number(pass);
        print(": ");
        print_number(errors);
        print(" words wrong\n");
    }
    print("done\n");
    for (;;)
        ;
}
