// A first program for riscv_computer.py's CPU: it greets, counts Fibonacci
// numbers, then echoes what is typed on the PC (juart-terminal), each key
// also on the LEDs.

#define CONSOLE  (*(volatile unsigned *)0x10000000)   // write: a character on the screen
#define KEYBOARD (*(volatile int *)0x10000004)        // read: a key from the PC, or -1
#define LEDS     (*(volatile unsigned *)0x10000008)   // write: the 8 LEDs

static void print(const char *s) {
    while (*s)
        CONSOLE = *s++;
}

static void print_number(unsigned n) {
    char digits[10];
    int k = 0;
    do {
        digits[k++] = '0' + n % 10;      // (no divide instruction: the compiler's own routine)
        n /= 10;
    } while (n);
    while (k)
        CONSOLE = digits[--k];
}

int main(void) {
    print("Hello from RISC-V!\n");
    print("Fibonacci:\n");
    unsigned a = 0, b = 1;
    while (a < 1000) {
        print_number(a);
        print(" ");
        unsigned next = a + b;
        a = b;
        b = next;
    }
    print("\nType, and I echo:\n");
    for (;;) {
        int key = KEYBOARD;
        if (key < 0)
            continue;
        if (key == '\r')
            key = '\n';
        CONSOLE = key;
        LEDS = key;
    }
}
