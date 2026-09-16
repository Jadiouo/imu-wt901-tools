# 2_offline_viz.py
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from imu_filters import Madgwick6, Mahony6, quat_rotate, quat_to_euler

def butter_lowpass(data, fs, cutoff=5.0, order=3):
    b, a = butter(order, cutoff/(0.5*fs), btype='low')
    return filtfilt(b, a, data, axis=0)

def cube_vertices():
    s = 0.5
    # 8 vertices of a cube
    V = np.array([
        [-s,-s,-s],[ s,-s,-s],[ s, s,-s],[-s, s,-s],
        [-s,-s, s],[ s,-s, s],[ s, s, s],[-s, s, s],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    return V, edges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="imu_raw.csv")
    ap.add_argument("--fs", type=float, default=100.0, help="assumed sample rate (Hz)")
    ap.add_argument("--filter", choices=["madgwick","mahony"], default="madgwick")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    t = df["t"].values
    fs = args.fs
    acc = df[["ax","ay","az"]].values
    gyro = df[["gx","gy","gz"]].values  # deg/s

    # 平滑加速度（降低高頻雜訊）
    acc_f = butter_lowpass(acc, fs, cutoff=5.0, order=3)

    # 初始化濾波器
    if args.filter == "madgwick":
        fuser = Madgwick6(beta=0.08, fs=fs)
    else:
        fuser = Mahony6(Kp=0.8, Ki=0.01, fs=fs)

    quats, rpy = [], []
    for a, g in zip(acc_f, gyro):
        q = fuser.update(a, g)
        quats.append(q)
        rpy.append(quat_to_euler(q))
    quats = np.array(quats)
    rpy = np.array(rpy)  # [N, 3]

    # 1) RPY 曲線
    plt.figure()
    plt.plot(t, rpy[:,0], label="roll (deg)")
    plt.plot(t, rpy[:,1], label="pitch (deg)")
    plt.plot(t, rpy[:,2], label="yaw (deg)")
    plt.xlabel("time (s)")
    plt.ylabel("deg")
    plt.title(f"Offline {args.filter.capitalize()} RPY")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) 3D 方塊動畫
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    V0, edges = cube_vertices()
    lines = []
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
    ax.set_box_aspect([1,1,1])
    ax.set_title("3D Orientation (Offline)")

    def update(i):
        q = quats[i]
        V = np.array([quat_rotate(q, v) for v in V0])
        ax.cla()
        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
        ax.set_box_aspect([1,1,1])
        ax.set_title("3D Orientation (Offline)")
        for e in edges:
            xs = [V[e[0],0], V[e[1],0]]
            ys = [V[e[0],1], V[e[1],1]]
            zs = [V[e[0],2], V[e[1],2]]
            ax.plot(xs, ys, zs)
        return []

    ani = FuncAnimation(fig, update, frames=len(quats), interval=1000/fs, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
