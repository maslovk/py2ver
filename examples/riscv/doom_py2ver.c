// Doom (doomgeneric, github.com/ozkl/doomgeneric) on the py2ver RISC-V
// computer under Linux: its picture on the display's framebuffer, its keys
// from the console. riscv_linux.py's MMU build compiles it with doomgeneric's
// files, CMAP256 and a 320 by 200 screen.
//
// The framebuffer (/dev/fb0) is 240 pixels wide and 320 high, RGB565. Doom's
// picture is 320 wide and 200 high, each pixel a palette index (CMAP256), so
// it is drawn turned a quarter turn clockwise, with a 20-pixel border either
// side: hold the display on its side. -ccw turns it the other way. Each pixel
// goes through a table of the palette in RGB565, made again when Doom changes
// its palette (when hit, or picking something up).
//
// The keys come from standard input, the console: the PC's keyboard over the
// JTAG UART, a character at a time (riscv_linux.py --console --keys). A
// terminal sends a character when a key is pressed and repeats it while the
// key is held, but never says when it is released. So a key counts as held
// for KEY_HOLD_MS after its first character (longer than a keyboard waits
// before repeating) and for KEY_REPEAT_MS after each repeat, then is released.
// -keyhold N sets KEY_HOLD_MS.
//
// The controls: the arrow keys move and turn, Space fires, e uses (opens
// doors, presses switches), comma and full stop step sideways, 1 to 7 choose a
// weapon, Tab shows the map, Escape the menu (Enter, y and n answer it), and
// Ctrl-C quits.

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <linux/fb.h>
#include <linux/kd.h>

#include "doomgeneric.h"
#include "doomkeys.h"
#include "i_system.h"
#include "i_video.h"
#include "m_argv.h"

#define KEY_HOLD_MS 500
#define KEY_REPEAT_MS 150
#define EVENTS 64                     // (a power of two)
#define ESCAPE_WAIT_MS 50             // Escape alone, if nothing follows it this soon
#define SHAREWARE_WAD "/usr/share/games/doom/doom1.wad"

// i_video.c's palette (with CMAP256), and its sign that the palette changed
extern struct color colors[256];
extern boolean palette_changed;

static uint8_t *framebuffer;
static unsigned framebuffer_width, framebuffer_height, framebuffer_line_bytes;
static int turn_counterclockwise;
static uint16_t palette_rgb565[256];

// The display's console, drawing nothing while Doom draws, and its mode before
static int console_fd = -1;
static int console_mode_before;

// Standard input, raw while Doom runs, and its settings before
static int keys_raw;
static struct termios keys_before;

static struct timeval started;
static int key_hold_ms = KEY_HOLD_MS;

// For each Doom key held: when it is to be released (in DG_GetTicksMs's
// milliseconds), else 0
static uint32_t released_at[256];

// Presses and releases for Doom: (pressed << 8) | key
static unsigned short events[EVENTS];
static unsigned events_in, events_out;

// An escape sequence (an arrow key's: Escape, [, A) as far as it has come
static unsigned char escape[8];
static int escape_length;
static uint32_t escape_started;


static void restore(void)
{
    if (keys_raw) {
        tcsetattr(STDIN_FILENO, TCSANOW, &keys_before);
        keys_raw = 0;
    }
    if (console_fd >= 0) {
        ioctl(console_fd, KDSETMODE, console_mode_before);
        close(console_fd);
        console_fd = -1;
    }
}

static void on_signal(int signal_number)
{
    restore();
    signal(signal_number, SIG_DFL);
    raise(signal_number);
}

void DG_Init(void)
{
    struct fb_var_screeninfo variable;
    struct fb_fix_screeninfo fixed;
    struct termios raw;
    int fd, p;

    fd = open("/dev/fb0", O_RDWR);
    if (fd < 0)
        I_Error("Doom can't open the framebuffer, /dev/fb0: %s", strerror(errno));
    if (ioctl(fd, FBIOGET_VSCREENINFO, &variable) != 0 || ioctl(fd, FBIOGET_FSCREENINFO, &fixed) != 0)
        I_Error("Doom can't ask the framebuffer its size: %s", strerror(errno));
    if (variable.bits_per_pixel != 16 || variable.xres < DOOMGENERIC_RESY || variable.yres < DOOMGENERIC_RESX)
        I_Error("The framebuffer is %u by %u at %u bits a pixel; Doom needs at least %u by %u at 16 bits",
                variable.xres, variable.yres, variable.bits_per_pixel, DOOMGENERIC_RESY, DOOMGENERIC_RESX);
    framebuffer_width = variable.xres;
    framebuffer_height = variable.yres;
    framebuffer_line_bytes = fixed.line_length;
    framebuffer = mmap(NULL, framebuffer_line_bytes * framebuffer_height, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
    if (framebuffer == MAP_FAILED)
        I_Error("Doom can't map the framebuffer into its memory: %s", strerror(errno));
    memset(framebuffer, 0, framebuffer_line_bytes * framebuffer_height);

    turn_counterclockwise = M_CheckParm("-ccw") > 0;
    p = M_CheckParmWithArgs("-keyhold", 1);
    if (p > 0)
        key_hold_ms = atoi(myargv[p + 1]);

    // The display's console (the foreground one, tty0) stops drawing its text
    // and cursor over Doom's picture
    console_fd = open("/dev/tty0", O_RDWR);
    if (console_fd >= 0 && ioctl(console_fd, KDGETMODE, &console_mode_before) == 0) {
        ioctl(console_fd, KDSETMODE, KD_GRAPHICS);
    } else if (console_fd >= 0) {
        close(console_fd);
        console_fd = -1;
    }

    // Standard input raw: each character as it comes, not echoed, Ctrl-C a
    // character like the others, a read never waiting
    if (tcgetattr(STDIN_FILENO, &keys_before) == 0) {
        raw = keys_before;
        raw.c_lflag &= ~(ICANON | ECHO | ISIG);
        raw.c_iflag &= ~(IXON | ICRNL | INLCR);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        keys_raw = 1;
    }

    atexit(restore);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGSEGV, on_signal);
    signal(SIGABRT, on_signal);

    gettimeofday(&started, NULL);
}

void DG_DrawFrame(void)
{
    const uint8_t *picture = (const uint8_t *) DG_ScreenBuffer;
    unsigned left = (framebuffer_width - DOOMGENERIC_RESY) / 2;
    unsigned top = (framebuffer_height - DOOMGENERIC_RESX) / 2;
    unsigned x, y;

    if (palette_changed) {
        for (int i = 0; i < 256; i++)
            palette_rgb565[i] = ((colors[i].r >> 3) << 11) | ((colors[i].g >> 2) << 5) | (colors[i].b >> 3);
        palette_changed = false;
    }

    // Each of Doom's columns is one of the framebuffer's lines, written two
    // pixels at a time. Turned clockwise, Doom's first column is the top line
    // and its bottom row the left edge; turned counterclockwise, its first
    // column is the bottom line and its top row the left edge.
    for (x = 0; x < DOOMGENERIC_RESX; x++) {
        unsigned line = turn_counterclockwise ? top + DOOMGENERIC_RESX - 1 - x : top + x;
        uint32_t *out = (uint32_t *) (framebuffer + line * framebuffer_line_bytes + left * 2);
        if (turn_counterclockwise) {
            const uint8_t *in = picture + x;
            for (y = 0; y < DOOMGENERIC_RESY; y += 2, in += 2 * DOOMGENERIC_RESX)
                *out++ = palette_rgb565[in[0]] | (uint32_t) palette_rgb565[in[DOOMGENERIC_RESX]] << 16;
        } else {
            const uint8_t *in = picture + (DOOMGENERIC_RESY - 1) * DOOMGENERIC_RESX + x;
            for (y = 0; y < DOOMGENERIC_RESY; y += 2, in -= 2 * DOOMGENERIC_RESX)
                *out++ = palette_rgb565[in[0]] | (uint32_t) palette_rgb565[in[-DOOMGENERIC_RESX]] << 16;
        }
    }
}

void DG_SleepMs(uint32_t ms)
{
    usleep(ms * 1000);
}

uint32_t DG_GetTicksMs(void)
{
    struct timeval now;
    gettimeofday(&now, NULL);
    return (now.tv_sec - started.tv_sec) * 1000 + (now.tv_usec - started.tv_usec) / 1000;
}

static void add_event(int pressed, unsigned char key)
{
    if (events_in - events_out < EVENTS)
        events[events_in++ % EVENTS] = (pressed << 8) | key;
}

// A character for a Doom key came: pressed now if it wasn't held, and held a
// while longer either way
static void key_typed(unsigned char key)
{
    uint32_t now = DG_GetTicksMs();
    if (released_at[key] == 0) {
        add_event(1, key);
        released_at[key] = now + key_hold_ms;
    } else if ((int32_t) (now + KEY_REPEAT_MS - released_at[key]) > 0) {
        released_at[key] = now + KEY_REPEAT_MS;
    }
}

// One character from the console, turned into Doom's key
static void character_came(unsigned char c)
{
    if (escape_length > 0) {
        escape[escape_length++] = c;
        if (escape_length == 2 && c != '[' && c != 'O') {
            // Escape on its own, then another key
            escape_length = 0;
            key_typed(KEY_ESCAPE);
            character_came(c);
        } else if (escape_length > 2 && c >= 0x40 && c <= 0x7E) {
            // The sequence's last character: which arrow key
            escape_length = 0;
            switch (c) {
            case 'A': key_typed(KEY_UPARROW); break;
            case 'B': key_typed(KEY_DOWNARROW); break;
            case 'C': key_typed(KEY_RIGHTARROW); break;
            case 'D': key_typed(KEY_LEFTARROW); break;
            }
        } else if (escape_length == sizeof escape) {
            escape_length = 0;
        }
        return;
    }
    switch (c) {
    case 27:
        escape[0] = c;
        escape_length = 1;
        escape_started = DG_GetTicksMs();
        break;
    case 3: I_Quit(); break;
    case ' ': key_typed(KEY_FIRE); break;
    case 'e': case 'E': key_typed(KEY_USE); break;
    case ',': case '<': key_typed(KEY_STRAFE_L); break;
    case '.': case '>': key_typed(KEY_STRAFE_R); break;
    case '\r': case '\n': key_typed(KEY_ENTER); break;
    case '\t': key_typed(KEY_TAB); break;
    case 8: case 127: key_typed(KEY_BACKSPACE); break;
    default:
        if (c >= 'A' && c <= 'Z')
            c += 'a' - 'A';
        if (c > ' ' && c < 127)
            key_typed(c);
    }
}

int DG_GetKey(int *pressed, unsigned char *key)
{
    unsigned char got[64];
    ssize_t n;
    uint32_t now;

    while ((n = read(STDIN_FILENO, got, sizeof got)) > 0)
        for (ssize_t i = 0; i < n; i++)
            character_came(got[i]);

    now = DG_GetTicksMs();
    if (escape_length == 1 && now - escape_started >= ESCAPE_WAIT_MS) {
        escape_length = 0;
        key_typed(KEY_ESCAPE);
    }
    for (int k = 0; k < 256; k++) {
        if (released_at[k] != 0 && (int32_t) (now - released_at[k]) >= 0) {
            released_at[k] = 0;
            add_event(0, k);
        }
    }

    if (events_out == events_in)
        return 0;
    *pressed = events[events_out % EVENTS] >> 8;
    *key = events[events_out % EVENTS] & 0xFF;
    events_out++;
    return 1;
}

void DG_SetWindowTitle(const char *title)
{
    (void) title;
}

int main(int argc, char **argv)
{
    // doomgeneric looks for a WAD only in the current folder, so unless one is
    // named (-iwad), Doom is given the shareware WAD, where the package puts it
    static char *arguments[64];
    int named = 0;
    for (int i = 1; i < argc; i++)
        named |= strcmp(argv[i], "-iwad") == 0;
    if (!named && argc + 2 < 64) {
        memcpy(arguments, argv, argc * sizeof *argv);
        arguments[argc++] = "-iwad";
        arguments[argc++] = SHAREWARE_WAD;
        argv = arguments;
    }
    doomgeneric_Create(argc, argv);
    for (;;)
        doomgeneric_Tick();
    return 0;
}
