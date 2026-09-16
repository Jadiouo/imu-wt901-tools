# imu_filters.py
import numpy as np

def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / (n + eps)

def quat_multiply(q, r):
    # (w, x, y, z)
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_rotate(q, v):
    # rotate vector v by quaternion q
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    vq = np.array([0, *v])
    return quat_multiply(quat_multiply(q, vq), q_conj)[1:]

def quat_to_euler(q):
    # returns roll, pitch, yaw in degrees
    w, x, y, z = q
    # roll (x-axis rotation)
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))
    # pitch (y-axis)
    sinp = 2*(w*y - z*x)
    sinp = np.clip(sinp, -1, 1)
    pitch = np.degrees(np.arcsin(sinp))
    # yaw (z-axis)
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw

class Madgwick6:
    def __init__(self, beta=0.1, fs=100.0):
        self.beta = beta
        self.fs = fs
        self.q = np.array([1., 0., 0., 0.])  # w,x,y,z

    def update(self, acc, gyro_deg_s):
        # acc: m/s^2 or g; gyro: deg/s
        # normalize acc to g-direction only
        a = _normalize(acc)
        # convert gyro to rad/s
        g = np.radians(gyro_deg_s)

        q = self.q
        # objective function: align gravity (0,0,1) with -acc (or acc?)
        # here we expect acc points to gravity; we use reference (0,0,1)
        # gradient descent step from Madgwick's paper (6-axis version)
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

        # q_dot = 0.5 * q ⊗ ω - beta * step
        omega = np.array([0., *g])  # 0, gx, gy, gz
        q_dot = 0.5 * quat_multiply(q, omega) - self.beta * step

        q = q + q_dot / self.fs
        self.q = _normalize(q)
        return self.q

class Mahony6:
    def __init__(self, Kp=1.0, Ki=0.0, fs=100.0):
        self.Kp = Kp
        self.Ki = Ki
        self.fs = fs
        self.q = np.array([1., 0., 0., 0.])
        self.int_err = np.zeros(3)

    def update(self, acc, gyro_deg_s):
        a = _normalize(acc)
        g = np.radians(gyro_deg_s)

        q = self.q
        # Estimated gravity vector from current orientation
        g_est = quat_rotate(q, np.array([0, 0, 1]))
        # error is cross product between measured acc (gravity dir) and estimated
        err = np.cross(g_est, a)

        if self.Ki > 0.0:
            self.int_err += err / self.fs
        else:
            self.int_err[:] = 0.0

        g_corr = g + self.Kp * err + self.Ki * self.int_err
        omega = np.array([0., *g_corr])
        q_dot = 0.5 * quat_multiply(q, omega)

        q = q + q_dot / self.fs
        self.q = _normalize(q)
        return self.q
