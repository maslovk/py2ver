#!/usr/bin/env python3
import sys, time, argparse, serial

# -------------------- PRBS7 helper --------------------
# Polynomial: x^7 + x^6 + 1 over 7-bit state (bits [6:0])
def prbs7_next(x: int) -> int:
    fb = ((x >> 6) & 1) ^ ((x >> 5) & 1)
    return ((x << 1) & 0x7F) | fb

# -------------------- Main ----------------------------
def main():
    ap = argparse.ArgumentParser(description="PRBS7 UART checker (FPGA TX 8N1, bit7=0).")
    ap.add_argument("port", nargs="?", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=0.1)
    ap.add_argument("--gap-reset", type=float, default=0.200,
                    help="new-burst gap in seconds to force resync")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--preamble", dest="preamble", action="store_true",
                       help="(force) expect preamble 55 AA <seed>")
    group.add_argument("--no-preamble", dest="preamble", action="store_false",
                       help="disable preamble detection")
    ap.add_argument("--confirm", type=int, default=2,
                    help="consecutive matches to declare LOCK (default 2)")
    ap.add_argument("--progress", type=int, default=512,
                    help="progress interval (bytes)")
    ap.set_defaults(preamble=True)   # ✅ preamble ON by default
    a = ap.parse_args()

    ser = serial.Serial(
        a.port, a.baud,
        timeout=a.timeout,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,  # match FPGA 8N1
        rtscts=False, dsrdtr=False, xonxoff=False
    )
    print(f"Listening on {a.port} @ {a.baud} (8N1, preamble={'ON' if a.preamble else 'OFF'}) … CTRL+C to stop")

    # Predictor holds NEXT expected 7-bit payload
    next_val = 0x7F
    have_lock = False
    confirm_left = 0

    pre_idx = 0 if a.preamble else 3  # 0–2 = in preamble, >=3 payload
    last_t = time.time()
    total = bad = 0

    while True:
        b = ser.read(1)
        now = time.time()
        if not b:
            continue

        # Burst gap → reset prediction
        if now - last_t > a.gap_reset:
            have_lock = True
            confirm_left = 0
            pre_idx = 0 if a.preamble else 3
            next_val = 0x7F
        last_t = now

        byte = b[0]
        msb  = (byte >> 7) & 1
        data = byte & 0x7F
        total += 1

        # ----------- Preamble: 55, AA, <seed> -------------
        if pre_idx < 3:
            ok = False
            if pre_idx == 0:
                ok = (byte == 0x55)
            elif pre_idx == 1:
                ok = (byte == 0xAA)
            elif pre_idx == 2:
                ok = (msb == 0)
                if ok:
                    next_val = prbs7_next(data)  # predict successor
                    have_lock = True
                    confirm_left = a.confirm

            if ok:
                pre_idx += 1
            else:
                bad += 1  # noise until found again

            if total % a.progress == 0:
                rate = bad * 100.0 / total
                lock = "SEEK" if pre_idx < 3 else ("LOCK" if confirm_left == 0 else "LOCK?")
                print(f"Progress: {total} bytes, errors={bad} ({rate:.4f}%), next=0x{next_val:02X}, {lock}")
            continue

        # ---------------- Payload ----------------
        if not have_lock:
            next_val = prbs7_next(data)
            confirm_left = a.confirm
            have_lock = True
            continue

        ok = (msb == 0) and (data == next_val)

        if ok:
            if confirm_left > 0:
                confirm_left -= 1
            next_val = prbs7_next(next_val)
        else:
            bad += 1
            next_val = prbs7_next(data)
            confirm_left = a.confirm

        if total % a.progress == 0:
            rate = bad * 100.0 / total
            locked = (confirm_left == 0)
            lock_str = "LOCK" if locked else f"LOCK?({a.confirm - confirm_left}/{a.confirm})"
            print(f"Progress: {total} bytes, errors={bad} ({rate:.4f}%), next=0x{next_val:02X}, {lock_str}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye")
