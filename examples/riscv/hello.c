// A first program for riscv_computer.py's CPU: it greets, counts Fibonacci
// numbers, then echoes what is typed on the PC (juart-terminal), each key
// also on the LEDs.

// The console: a 16550-style UART's registers (as Linux's driver knows them)
#define UART_DATA   (*(volatile unsigned char *)0x10000000)   // write: a character out; read: one in
#define UART_STATUS (*(volatile unsigned char *)0x10000005)   // bit 0: a character came in
#define LEDS        (*(volatile unsigned *)0x10001000)        // write: the 8 LEDs

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
        if (!(UART_STATUS & 1))
            continue;
        int key = UART_DATA;
        if (key == '\r')
            key = '\n';
        UART_DATA = key;
        LEDS = key;
    }
}
