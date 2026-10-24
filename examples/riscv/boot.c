// A boot loader for riscv_computer.py's CPU: it takes programs (a Linux
// kernel, its device tree) from the PC over the JTAG UART into memory, and
// starts one. riscv_linux.py's load() is the PC's side. What comes from the
// PC, a record at a time (numbers 4 bytes, the lowest first):
//
//   'L', address, length, then length bytes (a multiple of 4): stored from
//        address on; it answers "loaded <length> at <address>: sum <sum>",
//        the sum (of the bytes, read back from memory) in hexadecimal
//   'G', address, a1: start there (a0 = 0, the hart; a1, the device tree's
//        address for Linux), saying "starting at <address>"
//
// and any other byte is ignored. Its words go to the console, which the PC
// reads when it listens (and the display shows).

#define UART_DATA   (*(volatile unsigned char *)0x10000000)
#define UART_STATUS (*(volatile unsigned char *)0x10000005)   // bit 0: a byte came in
#define LEDS        (*(volatile unsigned *)0x10001000)

static void print(const char *s) {
    while (*s)
        UART_DATA = *s++;
}

static void print_hex(unsigned n) {
    for (int k = 28; k >= 0; k -= 4)
        UART_DATA = "0123456789abcdef"[(n >> k) & 15];
}

static unsigned take_byte(void) {
    while (!(UART_STATUS & 1))
        ;
    return UART_DATA;
}

static unsigned take_word(void) {
    unsigned w = take_byte();
    w |= take_byte() << 8;
    w |= take_byte() << 16;
    return w | take_byte() << 24;
}

int main(void) {
    print("py2ver RISC-V boot loader: waiting for the PC\n");
    for (;;) {
        unsigned record = take_byte();
        if (record == 'L') {
            volatile unsigned *at = (volatile unsigned *)take_word();
            unsigned length = take_word();
            for (unsigned k = 0; k < length / 4; k++) {
                at[k] = take_word();
                if ((k & 0xFFFF) == 0)
                    LEDS = k >> 16;          // (progress: a count each 256 KB)
            }
            unsigned sum = 0;
            volatile unsigned char *bytes = (volatile unsigned char *)at;
            for (unsigned k = 0; k < length; k++)
                sum += bytes[k];
            print("loaded ");
            print_hex(length);
            print(" at ");
            print_hex((unsigned)at);
            print(": sum ");
            print_hex(sum);
            print("\n");
        } else if (record == 'G') {
            unsigned address = take_word();
            unsigned a1 = take_word();
            print("starting at ");
            print_hex(address);
            print("\n");
            ((void (*)(unsigned, unsigned))address)(0, a1);
        }
    }
}
