# 1_collect.py
import time
import csv
import argparse
import numpy as np
import serial

def parse_line(line):
    parts = [p.strip() for p in line.decode(errors='ignore').split(',')]
    vals = [float(x) for x in parts[:6]]  # AX,AY,AZ,GX,GY,GZ
    ax, ay, az, gx, gy, gz = vals
    return np.array([ax, ay, az]), np.array([gx, gy, gz])

def estimate_gyro_bias(ser, seconds=5.0):
    print(f"[Calib] Keep IMU still for {seconds} s to estimate gyro bias...")
    buf = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        line = ser.readline()
        try:
            _, g = parse_line(line)
            buf.append(g)
        except Exception:
            pass
    bias = np.mean(np.array(buf), axis=0)
    print("[Calib] Gyro bias (deg/s):", bias)
    return bias

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="COM port, e.g., COM5")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", default="imu_raw.csv")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds to record")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)

    gyro_bias = estimate_gyro_bias(ser, seconds=5.0)

    print(f"[Record] Writing to {args.out} for {args.duration} s")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz"])
        t0 = time.time()
        while time.time() - t0 < args.duration:
            line = ser.readline()
            if not line:
                continue
            try:
                a, g = parse_line(line)
                # 校正：減去 gyro 零偏；加速度幅值正規化（避免單位差異）
                g = g - gyro_bias
                anorm = np.linalg.norm(a) + 1e-9
                a = a / anorm
                t = time.time() - t0
                w.writerow([f"{t:.6f}", *a.tolist(), *g.tolist()])
            except Exception:
                continue

    print("[Done] File saved:", args.out)
    ser.close()

if __name__ == "__main__":
    main()
