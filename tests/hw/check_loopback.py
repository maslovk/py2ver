#!/usr/bin/env python3
import sys, time, argparse, serial
from collections import deque

# -------------------- Patterns --------------------
def gen_inc(start=0):
    v = start & 0xFF
    while True:
        yield v
        v = (v + 1) & 0xFF

def gen_alt(a=0x55, b=0xAA):
    v = a & 0xFF
    while True:
        yield v
        v = b if v == a else a

def gen_walking_ones():
    v = 0x01
    while True:
        yield v
        v = ((v << 1) & 0xFF) or 0x01

PATTERNS = {"inc": gen_inc, "alt": gen_alt, "walk1": gen_walking_ones}

def _now(): return time.time()

def main():
    ap = argparse.ArgumentParser(description="UART loopback tester (send pattern, verify echo).")
    ap.add_argument("port", nargs="?", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=0.02, help="serial read timeout (s)")
    ap.add_argument("--pattern", choices=PATTERNS.keys(), default="inc")
    ap.add_argument("--preamble", dest="preamble", action="store_true",
                    help="send 55 AA and wait for echo before counting")
    ap.add_argument("--no-preamble", dest="preamble", action="store_false",
                    help="do not send/wait preamble (start immediately)")
    ap.add_argument("--lock-timeout", type=float, default=0.5,
                    help="fallback to transmit if preamble echo not seen in this many seconds")
    ap.add_argument("--rate", type=float, default=0.0, help="limit TX rate to N bytes/sec (0=unlimited)")
    ap.add_argument("--bytes", type=int, default=0, help="stop after N bytes (0=forever)")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0=forever)")
    ap.add_argument("--progress", type=int, default=512, help="print progress every N sent bytes")
    ap.add_argument("--echo-timeout", type=float, default=1.0,
                    help="consider an expected echo 'missed' after this many seconds")
    ap.add_argument("--gap-reset", type=float, default=0.200,
                    help="RX idle gap (s) triggers resync (flush expected & re-lock if preamble)")
    ap.add_argument("--resync-scan", type=int, default=32,
                    help="on mismatch, scan ahead this many expected bytes to re-align (0=off)")
    ap.add_argument("--start", type=int, default=0, help="initial value for 'inc' pattern")
    ap.set_defaults(preamble=True)
    a = ap.parse_args()

    try:
        ser = serial.Serial(a.port, a.baud, timeout=a.timeout,
                            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE, rtscts=False, dsrdtr=False, xonxoff=False)
    except Exception as e:
        print(f"Error opening {a.port} @ {a.baud}: {e}", file=sys.stderr); sys.exit(2)

    print(f"Loopback test on {a.port} @ {a.baud} (8N1), pattern={a.pattern}, "
          f"preamble={'ON' if a.preamble else 'OFF'} — CTRL+C to stop")

    gen = PATTERNS[a.pattern](a.start) if a.pattern == "inc" else PATTERNS[a.pattern]()
    interval = (1.0 / a.rate) if a.rate > 0 else 0.0

    expected = deque()      # (byte, t_send)
    sent = rcvd = mism = missed = 0
    lat_hist = deque(maxlen=2000)
    t0 = _now(); last_tx_time = 0.0; last_rx_time = t0
    stop_at = t0 + a.seconds if a.seconds > 0 else None

    locked = not a.preamble
    preamble_seq = deque([0x55, 0xAA]) if a.preamble else deque()
    lock_deadline = t0 + a.lock_timeout if a.preamble else None

    if a.preamble:
        ser.reset_input_buffer()
        ser.write(bytes([0x55, 0xAA])); ser.flush()
        print(f"Waiting for preamble echo (55 AA)… timeout {a.lock_timeout:.3f}s")

    def read_some():
        nonlocal last_rx_time
        rb = ser.read(ser.in_waiting or 1)
        if rb: last_rx_time = _now()
        return rb

    try:
        while True:
            now = _now()
            if stop_at and now >= stop_at: break
            if a.bytes and sent >= a.bytes: break

            # Lock fallback
            if not locked and a.preamble and lock_deadline and now >= lock_deadline:
                print("Preamble echo not seen — starting anyway. (Use --no-preamble to skip lock next time.)")
                locked = True
                expected.clear()
                last_tx_time = now

            # TX
            if locked and (interval == 0.0 or (now - last_tx_time) >= interval):
                txb = next(gen) & 0xFF
                ser.write(bytes([txb])); last_tx_time = now
                expected.append((txb, now)); sent += 1
                print(f"TX: 0x{txb:02X}")

                if a.progress > 0 and (sent % a.progress == 0):
                    rate_bps = sent / max(now - t0, 1e-9)
                    err = mism + missed
                    err_pct = (err * 100.0 / max(sent, 1))
                    lat_ms = (sum(lat_hist) / len(lat_hist) * 1000.0) if lat_hist else 0.0
                    nxt = expected[0][0] if expected else None
                    print(f"Progress: sent={sent}, recv={rcvd}, errs={err} "
                          f"(mism={mism}, missed={missed}, {err_pct:.4f}%), "
                          f"tx_rate={rate_bps:.0f} B/s, avg_rtt={lat_ms:.2f} ms, "
                          f"{'LOCK' if locked else 'UNLOCK'}{'' if nxt is None else f', next=0x{nxt:02X}'}")

            # RX
            rb = read_some()
            for b in rb:
                b = b if isinstance(b, int) else b[0]
                print(f"RX: 0x{b:02X}")

                if not locked:
                    if preamble_seq and b == preamble_seq[0]:
                        preamble_seq.popleft()
                        if not preamble_seq:
                            locked = True; expected.clear(); last_tx_time = _now()
                            print("LOCKED on preamble.")
                    continue

                if not expected:
                    mism += 1; continue

                exp_b, t_send = expected[0]
                if b == exp_b:
                    expected.popleft(); rcvd += 1; lat_hist.append(_now() - t_send)
                else:
                    if a.resync_scan > 0:
                        idx = -1; limit = min(a.resync_scan, len(expected))
                        for i in range(limit):
                            if expected[i][0] == b: idx = i; break
                        if idx >= 0:
                            missed += idx
                            for _ in range(idx): expected.popleft()
                            _, t2 = expected.popleft(); rcvd += 1; lat_hist.append(_now() - t2)
                            continue
                    mism += 1

            # Gap-based resync
            if a.gap_reset > 0 and (now - last_rx_time) > a.gap_reset:
                if expected:
                    missed += len(expected); expected.clear()
                if a.preamble:
                    locked = False; preamble_seq = deque([0x55, 0xAA])
                    lock_deadline = _now() + a.lock_timeout
                    print("RX gap: re-locking on preamble…")
                last_rx_time = now

            # Timeout cull
            while expected and (now - expected[0][1]) > a.echo_timeout:
                expected.popleft(); missed += 1

        # small drain
        end_wait = _now() + 0.25
        while _now() < end_wait and expected:
            rb = read_some()
            for b in rb:
                b = b if isinstance(b, int) else b[0]
                print(f"RX: 0x{b:02X}")
                if not expected: mism += 1; continue
                exp_b, t_send = expected[0]
                if b == exp_b:
                    expected.popleft(); rcvd += 1; lat_hist.append(_now() - t_send)
                else:
                    mism += 1

    except KeyboardInterrupt:
        pass
    finally:
        total_time = max(_now() - t0, 1e-9)
        err = mism + missed
        err_pct = (err * 100.0 / max(sent, 1))
        tx_rate = sent / total_time; rx_rate = rcvd / total_time
        if lat_hist:
            lat_avg_ms = sum(lat_hist)/len(lat_hist)*1000.0
            l = sorted(lat_hist); n = len(l)
            p50 = l[int(0.50*(n-1))]*1000.0; p95 = l[int(0.95*(n-1))]*1000.0; p99 = l[int(0.99*(n-1))]*1000.0
        else:
            lat_avg_ms = p50 = p95 = p99 = 0.0

        print("\n=== Loopback Summary ===")
        print(f"Sent:     {sent}")
        print(f"Received: {rcvd}")
        print(f"Mismatches: {mism}")
        print(f"Missed:     {missed} (>{a.echo_timeout:.3f}s late)")
        print(f"Error rate: {err_pct:.6f}%")
        print(f"TX rate: {tx_rate:.1f} B/s, RX rate: {rx_rate:.1f} B/s")
        print(f"Latency (ms): avg={lat_avg_ms:.3f}, p50={p50:.3f}, p95={p95:.3f}, p99={p99:.3f}")
        try: ser.close()
        except Exception: pass

if __name__ == "__main__":
    main()
