# 3_realtime_viz_game.py
# Realtime IMU viz + WASD mapping, parsing WitMotion/WT901 0x55 binary frames.
# Frames: 0x55 0x51 (ACC) / 0x52 (GYRO) / 0x59 (QUAT), each 11 bytes, checksum sum(first 10) & 0xFF == byte10
import time
import threading
import argparse
import struct
import numpy as np
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pynput.keyboard import Controller
from collections import deque

keyboard = Controller()

# ------------------ math utils ------------------
def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def quat_multiply(q, r):
    w1,x1,y1,z1 = q; w2,x2,y2,z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_rotate(q, v):
    qc = np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    return quat_multiply(quat_multiply(q, vq), qc)[1:]

def quat_to_euler(q):
    w,x,y,z = q
    # roll
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))
    # pitch
    sinp = 2*(w*y - z*x)
    sinp = np.clip(sinp, -1, 1)
    pitch = np.degrees(np.arcsin(sinp))
    # yaw
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw

# ------------------ AHRS (6-axis) ------------------
class Madgwick6:
    def __init__(self, beta=0.08, fs=100.0):
        self.beta, self.fs = beta, fs
        self.q = np.array([1.,0.,0.,0.], dtype=float)

    def update(self, acc_dir, gyro_dps):
        a = _normalize(acc_dir)
        g = np.radians(gyro_dps)
        q = self.q
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - a[0],
            2*(q[0]*q[1] + q[2]*q[3]) - a[1],
            2*(0.5 - q[1]**2 - q[2]**2) - a[2]
        ])
        J = np.array([
            [-2*q[2],  2*q[3], -2*q[0], 2*q[1]],
            [ 2*q[1],  2*q[0],  2*q[3], 2*q[2]],
            [     0., -4*q[1], -4*q[2],     0.]
        ])
        step = J.T @ f
        n = np.linalg.norm(step)
        if n > 1e-12:
            step /= n
        omega = np.array([0., *g])
        q_dot = 0.5 * quat_multiply(q, omega) - self.beta * step
        q = q + q_dot / self.fs
        self.q = _normalize(q)
        return self.q

class Mahony6:
    def __init__(self, Kp=0.8, Ki=0.01, fs=100.0):
        self.Kp, self.Ki, self.fs = Kp, Ki, fs
        self.q = np.array([1.,0.,0.,0.], dtype=float)
        self.int_err = np.zeros(3, dtype=float)

    def update(self, acc_dir, gyro_dps):
        a = _normalize(acc_dir)
        g = np.radians(gyro_dps)
        q = self.q
        g_est = quat_rotate(q, np.array([0,0,1.0]))
        err = np.cross(g_est, a)
        if self.Ki > 0: self.int_err += err / self.fs
        else: self.int_err[:] = 0.0
        g_corr = g + self.Kp*err + self.Ki*self.int_err
        omega = np.array([0., *g_corr])
        q_dot = 0.5 * quat_multiply(q, omega)
        q = q + q_dot / self.fs
        self.q = _normalize(q)
        return self.q

# ------------------ key mapping ------------------
class KeyMapper:
    def __init__(self, pitch_th=10.0, roll_th=10.0, dead=6.0):
        self.pitch_th = pitch_th
        self.roll_th = roll_th
        self.dead = dead
        self.state = {"W":False, "S":False, "A":False, "D":False}

    def _press(self, k):
        if not self.state[k]:
            self.state[k] = True
            keyboard.press(k)

    def _release(self, k):
        if self.state[k]:
            self.state[k] = False
            keyboard.release(k)

    def update(self, roll, pitch):
        # Pitch → W/S
        if pitch > self.pitch_th:
            self._press('W'); self._release('S')
        elif pitch < -self.pitch_th:
            self._press('S'); self._release('W')
        elif abs(pitch) < self.dead:
            self._release('W'); self._release('S')
        # Roll → A/D
        if roll > self.roll_th:
            self._press('D'); self._release('A')
        elif roll < -self.roll_th:
            self._press('A'); self._release('D')
        elif abs(roll) < self.dead:
            self._release('A'); self._release('D')

# ------------------ binary parser (WitMotion) ------------------
def checksum_ok(pkt):
    # 11 bytes; sum of first 10 & 0xFF == last
    return (sum(pkt[:10]) & 0xFF) == pkt[10]

def i16(lo, hi):
    return struct.unpack('<h', bytes([lo, hi]))[0]

def parse_wit_packet(pkt):
    """
    pkt: 11 bytes, starts with 0x55
    returns (typ_str, data_dict) or (None, None)
    """
    if len(pkt) != 11 or pkt[0] != 0x55 or not checksum_ok(pkt):
        return None, None
    typ = pkt[1]
    lo2, hi2, lo3, hi3, lo4, hi4, lo5, hi5 = pkt[2:10]
    x = i16(lo2, hi2)
    y = i16(lo3, hi3)
    z = i16(lo4, hi4)
    w = i16(lo5, hi5)
    if typ == 0x51:  # ACC ±16g, temp/extra in w
        g = 16.0
        return "ACC", {
            "ax_g": x/32768.0*g,
            "ay_g": y/32768.0*g,
            "az_g": z/32768.0*g,
            "temp_C": w/100.0
        }
    elif typ == 0x52:  # GYRO ±2000 dps
        dps = 2000.0
        return "GYRO", {
            "gx_dps": x/32768.0*dps,
            "gy_dps": y/32768.0*dps,
            "gz_dps": z/32768.0*dps,
            "temp_C": w/100.0
        }
    elif typ == 0x59:  # QUAT q0,q1,q2,q3
        return "QUAT", {
            "qw": x/32768.0,
            "qx": y/32768.0,
            "qy": z/32768.0,
            "qz": w/32768.0
        }
    else:
        return None, None

def cube_vertices():
    s = 0.5
    V = np.array([
        [-s,-s,-s],[ s,-s,-s],[ s, s,-s],[-s, s,-s],
        [-s,-s, s],[ s,-s, s],[ s, s, s],[-s, s, s],
    ], dtype=float)
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    return V, edges

def main():
    ap = argparse.ArgumentParser(description="Realtime IMU 3D + WASD (WitMotion binary)")
    ap.add_argument("--port", required=True, help="e.g., COM3")
    ap.add_argument("--baud", type=int, default=9600, help="Wit default is often 9600")
    ap.add_argument("--fs", type=float, default=100.0, help="assumed IMU update rate (Hz) for AHRS")
    ap.add_argument("--filter", choices=["madgwick","mahony"], default="madgwick", help="used when no quaternion or prefer_quat=False")
    ap.add_argument("--beta", type=float, default=0.08, help="Madgwick beta")
    ap.add_argument("--kp", type=float, default=0.8, help="Mahony Kp")
    ap.add_argument("--ki", type=float, default=0.02, help="Mahony Ki")
    ap.add_argument("--pitch_th", type=float, default=10.0)
    ap.add_argument("--roll_th", type=float, default=10.0)
    ap.add_argument("--dead", type=float, default=6.0)
    ap.add_argument("--prefer_quat", action="store_true", help="use device quaternion if available")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # open serial
    ser = serial.Serial(args.port, args.baud, timeout=0.02)
    time.sleep(0.3)

    # --- estimate gyro bias (keep still) ---
    print("[Calib] Keep IMU still for 3 s ...")
    t0 = time.time()
    gbuf = []
    while time.time() - t0 < 3.0:
        b = ser.read(1)
        if not b or b[0] != 0x55:
            continue
        pkt = b + ser.read(10)
        typ, data = parse_wit_packet(pkt)
        if typ == "GYRO":
            gbuf.append([data["gx_dps"], data["gy_dps"], data["gz_dps"]])
    gyro_bias = np.mean(gbuf, axis=0) if len(gbuf) else np.zeros(3)
    print("[Calib] Gyro bias (deg/s):", gyro_bias if len(gbuf) else "[no gyro during calib]")

    # --- filter / state ---
    if args.filter == "madgwick":
        fuser = Madgwick6(beta=args.beta, fs=args.fs)
    else:
        fuser = Mahony6(Kp=args.kp, Ki=args.ki, fs=args.fs)

    mapper = KeyMapper(pitch_th=args.pitch_th, roll_th=args.roll_th, dead=args.dead)

    latest_q = np.array([1.,0.,0.,0.], dtype=float)
    latest_rpy = (0.,0.,0.)
    have_quat = False
    last_ok_ts = 0.0

    # for debug stats
    ok_cnt = err_cnt = 0
    last_stat = time.time()
    last_typ = ""

    # --- visualization setup ---
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    V0, edges = cube_vertices()
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
    ax.set_box_aspect([1,1,1])

    # shared latest raw
    ax_mps2 = ay_mps2 = az_mps2 = np.nan
    gx_dps = gy_dps = gz_dps = np.nan

    def reader_loop():
        nonlocal latest_q, latest_rpy, have_quat, last_ok_ts
        nonlocal ax_mps2, ay_mps2, az_mps2, gx_dps, gy_dps, gz_dps
        nonlocal ok_cnt, err_cnt, last_stat, last_typ

        while True:
            b = ser.read(1)
            if not b or b[0] != 0x55:
                continue
            pkt = b + ser.read(10)
            typ, data = parse_wit_packet(pkt)
            if not typ:
                err_cnt += 1
                continue

            last_typ = typ
            if typ == "ACC":
                # convert g to m/s^2 (與先前腳本一致). 也可直接當方向量使用。
                ax_mps2 = data["ax_g"] * 9.80665
                ay_mps2 = data["ay_g"] * 9.80665
                az_mps2 = data["az_g"] * 9.80665
            elif typ == "GYRO":
                gx_dps = data["gx_dps"] - gyro_bias[0]
                gy_dps = data["gy_dps"] - gyro_bias[1]
                gz_dps = data["gz_dps"] - gyro_bias[2]
            elif typ == "QUAT":
                q = np.array([data["qw"], data["qx"], data["qy"], data["qz"]], dtype=float)
                q = _normalize(q)
                if args.prefer_quat:
                    latest_q = q
                    latest_rpy = quat_to_euler(q)
                    have_quat = True
                    mapper.update(*latest_rpy[:2])  # roll, pitch
                    last_ok_ts = time.time()
                ok_cnt += 1
                # continue to next loop (don't AHRS-update this cycle)
                continue

            # If not using device quaternion (or no quat frames), run AHRS with acc+gyro:
            if not args.prefer_quat or not have_quat:
                # need both acc and gyro numbers
                if not (np.isnan(ax_mps2) or np.isnan(gx_dps)):
                    a_dir = _normalize(np.array([ax_mps2, ay_mps2, az_mps2], dtype=float))
                    g = np.array([gx_dps, gy_dps, gz_dps], dtype=float)
                    q = fuser.update(a_dir, g)
                    latest_q = q
                    latest_rpy = quat_to_euler(q)
                    mapper.update(*latest_rpy[:2])  # roll, pitch
                    last_ok_ts = time.time()
                    ok_cnt += 1
                else:
                    err_cnt += 1

            # print stats each second in debug
            now = time.time()
            if args.debug and (now - last_stat) >= 1.0:
                print(f"[RX] ok/s={ok_cnt}  err/s={err_cnt}  last={last_typ}  prefer_quat={args.prefer_quat} have_quat={have_quat}")
                ok_cnt = err_cnt = 0
                last_stat = now

    threading.Thread(target=reader_loop, daemon=True).start()

    def update(_):
        q = latest_q
        V = np.array([quat_rotate(q, v) for v in V0])
        ax.cla()
        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
        ax.set_box_aspect([1,1,1])
        r, p, y = latest_rpy
        stale = (time.time() - last_ok_ts > 1.0)
        title = f"{'QUAT' if (args.prefer_quat and have_quat) else args.filter.upper()} | R={r:5.1f}° P={p:5.1f}° Y={y:5.1f}°"
        if stale: title += "  [NO NEW DATA ≥1s]"
        ax.set_title(title)
        for e in edges:
            xs = [V[e[0],0], V[e[1],0]]
            ys = [V[e[0],1], V[e[1],1]]
            zs = [V[e[0],2], V[e[1],2]]
            ax.plot(xs, ys, zs)
        # axis arrows
        ax.quiver(0,0,0, 1,0,0, length=0.8)
        ax.quiver(0,0,0, 0,1,0, length=0.8)
        ax.quiver(0,0,0, 0,0,1, length=0.8)
        return []

    ani = FuncAnimation(fig, update, interval=10, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
