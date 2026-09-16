# realtime_viz.py
import time
import argparse
import threading
import numpy as np
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt

# ---------- 小工具 ----------
def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / (n + eps)

def quat_multiply(q, r):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_rotate(q, v):
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    vq = np.array([0, *v])
    return quat_multiply(quat_multiply(q, vq), qc)[1:]

def quat_to_euler(q):
    w, x, y, z = q
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

def butter_lowpass(data, fs, cutoff=5.0, order=2):
    b, a = butter(order, cutoff/(0.5*fs), btype='low')
    return filtfilt(b, a, data, axis=0)

def cube_vertices():
    s = 0.5
    V = np.array([
        [-s,-s,-s],[ s,-s,-s],[ s, s,-s],[-s, s,-s],
        [-s,-s, s],[ s,-s, s],[ s, s, s],[-s, s, s],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    return V, edges

# ---------- 6軸 Madgwick / Mahony ----------
class Madgwick6:
    def __init__(self, beta=0.08, fs=100.0):
        self.beta = beta
        self.fs = fs
        self.q = np.array([1.,0.,0.,0.])
    def update(self, acc, gyro_deg_s):
        a = _normalize(acc)
        g = np.radians(gyro_deg_s)
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
        step = _normalize(step)
        omega = np.array([0., *g])
        q_dot = 0.5 * quat_multiply(q, omega) - self.beta * step
        q = q + q_dot / self.fs
        self.q = _normalize(q)
        return self.q

class Mahony6:
    def __init__(self, Kp=0.8, Ki=0.01, fs=100.0):
        self.Kp, self.Ki, self.fs = Kp, Ki, fs
        self.q = np.array([1.,0.,0.,0.])
        self.int_err = np.zeros(3)
    def update(self, acc, gyro_deg_s):
        a = _normalize(acc)
        g = np.radians(gyro_deg_s)
        q = self.q
        g_est = quat_rotate(q, np.array([0,0,1]))
        err = np.cross(g_est, a)
        if self.Ki > 0: self.int_err += err / self.fs
        else: self.int_err[:] = 0
        g_corr = g + self.Kp*err + self.Ki*self.int_err
        omega = np.array([0., *g_corr])
        q_dot = 0.5 * quat_multiply(q, omega)
        q = q + q_dot / self.fs
        self.q = _normalize(q)
        return self.q

# ---------- 串口/解析 ----------
def parse_line(line):
    # 期望: "AX,AY,AZ,GX,GY,GZ"
    parts = [p.strip() for p in line.decode(errors='ignore').split(',')]
    vals = [float(x) for x in parts[:6]]
    ax, ay, az, gx, gy, gz = vals
    return np.array([ax, ay, az]), np.array([gx, gy, gz])  # acc, gyro(deg/s)

def estimate_gyro_bias(ser, seconds=3.0):
    print(f"[Calib] Keep IMU still for {seconds} s ...")
    buf = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        line = ser.readline()
        if not line: continue
        try:
            _, g = parse_line(line)
            buf.append(g)
        except: pass
    bias = np.mean(np.array(buf), axis=0) if buf else np.zeros(3)
    print("[Calib] Gyro bias (deg/s):", bias)
    return bias

# ---------- 主程式 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="e.g., COM5")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--fs", type=float, default=100.0, help="IMU output rate (Hz)")
    ap.add_argument("--filter", choices=["madgwick","mahony"], default="madgwick")
    ap.add_argument("--beta", type=float, default=0.08, help="Madgwick beta")
    ap.add_argument("--kp", type=float, default=0.8, help="Mahony Kp")
    ap.add_argument("--ki", type=float, default=0.01, help="Mahony Ki")
    ap.add_argument("--acc_lp", type=float, default=5.0, help="acc lowpass cutoff Hz")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)

    gyro_bias = estimate_gyro_bias(ser, seconds=3.0)

    # 濾波器
    if args.filter == "madgwick":
        fuser = Madgwick6(beta=args.beta, fs=args.fs)
    else:
        fuser = Mahony6(Kp=args.kp, Ki=args.ki, fs=args.fs)

    # 畫面
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    V0, edges = cube_vertices()
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
    ax.set_box_aspect([1,1,1])
    ax.set_title("Realtime IMU Orientation")

    latest_q = np.array([1.,0.,0.,0.])
    latest_rpy = (0.,0.,0.)

    # 低通濾波器狀態（簡單滑動窗口）
    win_N = max(3, int(args.fs * 0.05))  # 約 50 ms 視窗
    acc_win = []

    def reader_loop():
        nonlocal latest_q, latest_rpy, acc_win
        while True:
            line = ser.readline()
            if not line: 
                continue
            try:
                a, g = parse_line(line)
                g = g - gyro_bias
                # acc 幅值正規化 + 輕微平滑
                a = a / (np.linalg.norm(a) + 1e-9)
                acc_win.append(a)
                if len(acc_win) > win_N:
                    acc_win = acc_win[-win_N:]
                a_smooth = np.mean(acc_win, axis=0) if acc_win else a
                q = fuser.update(a_smooth, g)
                latest_q = q
                latest_rpy = quat_to_euler(q)
            except Exception:
                continue

    t = threading.Thread(target=reader_loop, daemon=True)
    t.start()

    def update(_):
        q = latest_q
        V = np.array([quat_rotate(q, v) for v in V0])
        ax.cla()
        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
        ax.set_box_aspect([1,1,1])
        r, p, y = latest_rpy
        ax.set_title(f"Filter: {args.filter}  |  R={r:5.1f}°  P={p:5.1f}°  Y={y:5.1f}°")
        for e in edges:
            xs = [V[e[0],0], V[e[1],0]]
            ys = [V[e[0],1], V[e[1],1]]
            zs = [V[e[0],2], V[e[1],2]]
            ax.plot(xs, ys, zs)
        # 坐標軸箭頭
        ax.quiver(0,0,0, 1,0,0, length=0.8)  # X
        ax.quiver(0,0,0, 0,1,0, length=0.8)  # Y
        ax.quiver(0,0,0, 0,0,1, length=0.8)  # Z
        return []

    # interval~10ms；若 CPU 壓力大，可調大一點
    ani = FuncAnimation(fig, update, interval=10, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
