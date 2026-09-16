# serial_autobaud.py
import time, re, sys, serial
from serial.tools import list_ports

BAUDS = [115200, 57600, 38400, 19200, 9600, 230400, 460800, 921600, 1000000]
FLOAT_RE = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')

def score_line(s, expect=6):
    nums = FLOAT_RE.findall(s)
    return min(len(nums), expect)

def try_baud(port, baud, secs=2.0, expect=6):
    ok = 0; last = ""
    try:
        with serial.Serial(port, baud, timeout=0.05) as ser:
            t0 = time.time()
            while time.time() - t0 < secs:
                b = ser.readline()
                if not b:
                    continue
                try:
                    s = b.decode(errors="ignore")
                except:
                    s = repr(b)
                last = s.strip()
                ok += (score_line(s, expect) >= expect)
        return ok, last
    except Exception as e:
        return -1, f"<open err: {e}>"

def main():
    port = sys.argv[1] if len(sys.argv)>1 else None
    if not port:
        print("Usage: python serial_autobaud.py COM3")
        print("Available ports:")
        for p in list_ports.comports():
            print(" ", p.device, p.description)
        sys.exit(1)

    best = None
    for b in BAUDS:
        ok, last = try_baud(port, b)
        print(f"[{b:7}] score={ok:3}  sample: {last[:80]}")
        if best is None or ok > best[0]:
            best = (ok, b, last)
    if best and best[0] >= 3:
        print(f"\n>>> Likely baud: {best[1]} (score={best[0]})")
        print(f"    Sample: {best[2]}")
    else:
        print("\n!!! 沒找到像 6 浮點數的輸出。可能：")
        print(" - 裝置用的是『二進位協議』而非文字")
        print(" - 鮑率不在常見值，或串口被占用/沒在送資料")
        print(" - 其實不是這個 COM 埠")

if __name__ == "__main__":
    main()
