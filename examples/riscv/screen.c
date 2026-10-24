// Pictures on riscv_computer.py's framebuffer (computer(..., screen="framebuffer")):
// a test pattern (red across, green down), then the Mandelbrot set, drawn a
// pixel at a time into the SDRAM, which the display shows as it changes.
//
// WIDTH and HEIGHT (240 x 320) can be given with -D; PATTERN_ONLY: the test
// pattern alone.

#define UART_DATA   (*(volatile unsigned char *)0x10000000)
#define FRAMEBUFFER ((volatile unsigned short *)0x81FD0000)
#ifndef WIDTH
#define WIDTH 240
#endif
#ifndef HEIGHT
#define HEIGHT 320
#endif

static void print(const char *s) {
    while (*s)
        UART_DATA = *s++;
}

// A color from 8-bit red, green and blue: RGB565
static unsigned short rgb(unsigned red, unsigned green, unsigned blue) {
    return (red >> 3) << 11 | (green >> 2) << 5 | blue >> 3;
}

static void test_pattern(void) {
    for (unsigned y = 0; y < HEIGHT; y++)
        for (unsigned x = 0; x < WIDTH; x++)
            FRAMEBUFFER[y * WIDTH + x] = rgb(x * 256 / WIDTH, y * 256 / HEIGHT, 128);
}

// The Mandelbrot set, the real axis down the screen: numbers with 13 bits
// after the point (8192 is 1.0)
#define ONE 8192
#define LIMIT 64

static void mandelbrot(void) {
    for (int y = 0; y < HEIGHT; y++) {
        int c_re = -2 * ONE - ONE / 5 + y * (3 * ONE) / HEIGHT;
        for (int x = 0; x < WIDTH; x++) {
            int c_im = -ONE - ONE / 10 + x * (2 * ONE + ONE / 5) / WIDTH;
            int re = 0, im = 0, k = 0;
            while (k < LIMIT) {
                int re2 = re * re / ONE, im2 = im * im / ONE;
                if (re2 + im2 > 4 * ONE)
                    break;
                im = 2 * re * im / ONE + c_im;
                re = re2 - im2 + c_re;
                k++;
            }
            FRAMEBUFFER[y * WIDTH + x] = k == LIMIT ? 0 : rgb(k * 4, (k * 16) & 255, 255 - k * 4);
        }
    }
}

int main(void) {
    print("framebuffer: a test pattern\n");
    test_pattern();
    print("done\n");
#ifndef PATTERN_ONLY
    print("the Mandelbrot set\n");
    mandelbrot();
    print("done\n");
#endif
    for (;;)
        ;
}
