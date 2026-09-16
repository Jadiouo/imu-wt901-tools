# serial_snoop.py
import argparse, time, re, sys
import serial

FLOAT_RE = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')

def open_serial(port, baud, timeout=1.0):
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        time.sleep(0.5)  # 給裝置一點時間
        return ser
    except Exception as e:
        print(f"[ERR] Open serial failed: {e}")
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Simple serial monitor / IMU float parser")
    ap.add_argument("--port", required=True, help="e.g., COM5")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--raw", action="store_true", help="Print raw lines only (no parsing)")
    ap.add_argument("--expect", type=int, default=6, help="Expected number of floats per line")
    ap.add_argument("--rate", type=float, default=1.0, help="Stats print interval seconds")
    ap.add_argument("--max", type=int, default=0, help="Stop after N lines (0 = infinite)")
    args = ap.parse_args()

    ser = open_serial(args.port, args.baud, timeout=1.0)
    print(f"[INFO] Listening on {args.port} @ {args.baud} (raw={args.raw})")

    ok = err = total = 0
    last_stat = time.time()
    start = last_stat
    last_msg = ""

    try:
        while True:
            line = ser.readline()
            if not line:
                # 沒有資料，稍等一下避免 CPU 忙迴圈
                time.sleep(0.01)
                # 也在統計列出「無資料」狀態
                if time.time() - last_stat >= args.rate:
                    span = time.time() - last_stat
                    print(f"[STAT] no-data for {span:.1f}s  total={total}")
                    last_stat = time.time()
                continue

            total += 1
            try:
                s = line.decode(errors="ignore").rstrip("\r\n")
            except Exception:
                s = repr(line)
            last_msg = s

            if args.raw:
                print(s)
                ok += 1
            else:
                nums = FLOAT_RE.findall(s)
                if len(nums) >= args.expect:
                    vals = [float(x) for x in nums[:args.expect]]
                    # 顯示簡潔：ax ay az | gx gy gz
                    if args.expect >= 6:
                        ax, ay, az, gx, gy, gz = vals[:6]
                        print(f"AX={ax:8.3f} AY={ay:8.3f} AZ={az:8.3f} | "
                              f"GX={gx:8.3f} GY={gy:8.3f} GZ={gz:8.3f}")
                    else:
                        print(" ".join(f"{v:.3f}" for v in vals))
                    ok += 1
                else:
                    err += 1
                    print(f"[PARSE-ERR] need {args.expect} floats, got {len(nums)} :: {s}")

            # 每 args.rate 秒輸出一次吞吐率
            now = time.time()
            if now - last_stat >= args.rate:
                dur = now - last_stat
                since_start = now - start
                rate_ok = ok / dur if dur > 0 else 0.0
                rate_total = (ok + err) / dur if dur > 0 else 0.0
                print(f"[STAT] ok/s={rate_ok:.1f}  total/s={rate_total:.1f}  "
                      f"ok={ok} err={err}  last='{last_msg[:80]}'")
                ok = err = 0
                last_stat = now

            if args.max and total >= args.max:
                print(f"[INFO] Stopped after {args.max} lines.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
