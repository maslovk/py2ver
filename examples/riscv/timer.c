// The timer's interrupt on riscv_computer.py's CPU: once a second the timer
// interrupts whatever is running; the handler counts the seconds (on the LEDs)
// and sets the timer for the next one, while main prints each new second and
// echoes what is typed on the PC.

#define UART_DATA   (*(volatile unsigned char *)0x10000000)
#define UART_STATUS (*(volatile unsigned char *)0x10000005)   // bit 0: a character came in
#define LEDS        (*(volatile unsigned *)0x10001000)
#define MTIMECMP_LO (*(volatile unsigned *)0x11004000)        // the interrupt when mtime gets here
#define MTIMECMP_HI (*(volatile unsigned *)0x11004004)
#define MTIME_LO    (*(volatile unsigned *)0x1100BFF8)        // microseconds since reset
#define MTIME_HI    (*(volatile unsigned *)0x1100BFFC)

static volatile unsigned seconds;
static unsigned long long next_tick;

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

static void set_timer(unsigned long long when) {
    MTIMECMP_HI = 0xFFFFFFFF;            // (no interrupt half-way through the change)
    MTIMECMP_LO = (unsigned)when;
    MTIMECMP_HI = (unsigned)(when >> 32);
}

// The trap handler: the compiler saves the registers it uses and returns with mret
static void __attribute__((interrupt("machine"))) on_timer(void) {
    seconds++;
    LEDS = seconds;
    next_tick += 1000000;
    set_timer(next_tick);
}

int main(void) {
    print("Timer interrupts: a tick a second\n");
    next_tick = (((unsigned long long)MTIME_HI << 32) | MTIME_LO) + 1000000;
    set_timer(next_tick);
    __asm__ volatile ("csrw mtvec, %0" : : "r"(on_timer));
    __asm__ volatile ("csrs mie, %0" : : "r"(1 << 7));        // the timer's interrupt
    __asm__ volatile ("csrs mstatus, %0" : : "r"(1 << 3));    // interrupts on
    unsigned shown = 0;
    for (;;) {
        if (seconds != shown) {
            shown = seconds;
            print("tick ");
            print_number(shown);
            print("\n");
        }
        if (UART_STATUS & 1) {
            int key = UART_DATA;
            UART_DATA = key == '\r' ? '\n' : key;
        }
    }
}
