# wit_imu_stream.py
# Parse WitMotion/WT901-like 11-byte binary frames on Windows (8N1).
# Frames: 0x55 0x51(acc) / 0x52(gyro) / 0x53(angle) / 0x59(quat) + 8 data + 1 checksum
import serial, struct, time, argparse, sys
from collections import deque

TYPES = {0x51: "ACC", 0x52: "GYRO", 0x53: "ANGLE", 0x59: "QUAT"}

def checksum_ok(pkt):
    # sum of first 10 bytes & 0xFF should equal byte10
    return (sum(pkt[:10]) & 0xFF) == pkt[10]

def to_i16(lo, hi):
    return struct.unpack('<h', bytes([lo, hi]))[0]

def parse_packet(pkt):
    """
    pkt: 11 bytes starting with 0x55, pkt[1] is type
    Returns (typ, data_dict)
    """
    typ = pkt[1]
    if typ not in TYPES:
        return None, None
    d = {}
    # payload bytes 2..9
    x = to_i16(pkt[2], pkt[3])
    y = to_i16(pkt[4], pkt[5])
    z = to_i16(pkt[6], pkt[7])
    t = to_i16(pkt[8], pkt[9])  # temperature or extra

    if typ == 0x51:  # Acceleration: range ±16 g
        g = 16.0
        d["ax_g"] = x / 32768.0 * g
        d["ay_g"] = y / 32768.0 * g
        d["az_g"] = z / 32768.0 * g
        d["temp_C"] = t / 100.0
    elif typ == 0x52:  # Gyro: range ±2000 deg/s
        dps = 2000.0
        d["gx_dps"] = x / 32768.0 * dps
        d["gy_dps"] = y / 32768.0 * dps
        d["gz_dps"] = z / 32768.0 * dps
        d["temp_C"] = t / 100.0
    elif typ == 0x53:  # Angle: ±180 deg
        d["roll_deg"]  = x / 32768.0 * 180.0
        d["pitch_deg"] = y / 32768.0 * 180.0
        d["yaw_deg"]   = z / 32768.0 * 180.0
        d["temp_C"] = t / 100.0
    elif typ == 0x59:  # Quaternion
        d["qw"] = x / 32768.0
        d["qx"] = y / 32768.0
        d["qy"] = z / 32768.0
        d["qw_extra"] = t / 32768.0  # some firmwares pack it differently
    return TYPES[typ], d

def open_serial(port, baud, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.02):
    return serial.Serial(port=port, baudrate=baud, bytesize=bytesize, parity=parity, stopbits=stopbits, timeout=timeout)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=9600)  # many Wit IMUs default 9600
    ap.add_argument("--csv", help="write CSV to this path (columns: t,ax,ay,az,gx,gy,gz,roll,pitch,yaw)")
    ap.add_argument("--rate", type=float, default=50.0, help="print limit (Hz)")
    ap.add_argument("--verbose", action="store_true", help="also print raw frame types seen")
    args = ap.parse_args()

    try:
        ser = open_serial(args.port, args.baud)
    except Exception as e:
        print(f"[ERR] open serial failed: {e}")
        sys.exit(1)

    print(f"[INFO] Listening on {args.port} @ {args.baud} (binary)… Ctrl+C to stop.")
    buf = deque(maxlen=64)

    # latest values bucket
    ax=ay=az=gx=gy=gz=roll=pitch=yaw=float('nan')
    last_print = 0.0
    start = time.time()

    csv_f = open(args.csv, "w", buffering=1) if args.csv else None
    if csv_f:
        csv_f.write("t,ax,ay,az,gx,gy,gz,roll,pitch,yaw\n")

    try:
        while True:
            b = ser.read(1)
            if not b:
                continue
            if b[0] != 0x55:
                continue
            pkt = b + ser.read(10)  # total 11 bytes
            if len(pkt) != 11 or not checksum_ok(pkt):
                continue
            typ, data = parse_packet(pkt)
            if not typ:
                continue

            if args.verbose:
                print(f"[{typ}] {data}")

            if typ == "ACC":
                # convert g -> m/s^2  (for與你前面腳本一致)
                ax = data["ax_g"] * 9.80665
                ay = data["ay_g"] * 9.80665
                az = data["az_g"] * 9.80665
            elif typ == "GYRO":
                gx = data["gx_dps"]; gy = data["gy_dps"]; gz = data["gz_dps"]
            elif typ == "ANGLE":
                roll  = data["roll_deg"]; pitch = data["pitch_deg"]; yaw = data["yaw_deg"]

            now = time.time()
            if now - last_print >= 1.0/max(1.0, args.rate):
                t = now - start
                # 只有在 acc 與 gyro 都有值時才輸出一行
                if all(not (v != v) for v in (ax,ay,az,gx,gy,gz)):  # not NaN
                    line = f"{t:.3f},{ax:.5f},{ay:.5f},{az:.5f},{gx:.5f},{gy:.5f},{gz:.5f}"
                    # 角度若已到，也一併列印
                    if not (roll != roll):
                        line += f",{roll:.3f},{pitch:.3f},{yaw:.3f}"
                    print(line)
                    if csv_f:
                        csv_f.write(line + "\n")
                last_print = now

    except KeyboardInterrupt:
        print("\n[INFO] stopped by user.")
    finally:
        try:
            ser.close()
        except: pass
        if csv_f: csv_f.close()

if __name__ == "__main__":
    main()
