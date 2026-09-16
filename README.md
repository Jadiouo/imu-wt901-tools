# imu-wt901-tools

WitMotion WT901 系列 IMU 的 Python 小工具：透過序列埠收資料、離線 / 即時姿態視覺化、以及把 IMU 傾斜對應成 WASD 按鍵的小遊戲控制器。

姿態解算用自己實作的 6 軸 Madgwick / Mahony 濾波器（`imu_filters.py`），不依賴外部 AHRS 套件。

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `wit_imu_stream.py` | 解析 WitMotion 11-byte 二進位封包（`0x55 0x51` ACC / `0x52` GYRO / `0x53` ANGLE / `0x59` QUAT），即時列印，可加 `--csv` 寫檔 |
| `1_collect.py` | 從序列埠錄製原始資料到 CSV（欄位 `t,ax,ay,az,gx,gy,gz,roll,pitch,yaw`） |
| `2_offline_viz.py` | 讀 CSV，低通濾波後跑 Madgwick / Mahony，畫出姿態曲線 |
| `realtime_viz.py` | 即時 3D 姿態視覺化（matplotlib） |
| `3_realtime_viz_game.py` | 即時視覺化 + 依 pitch / roll 門檻送出 WASD 按鍵（`pynput`），可當體感搖桿 |
| `imu_filters.py` | `Madgwick6`、`Mahony6` 濾波器與四元數工具（`quat_multiply`、`quat_rotate`、`quat_to_euler`） |
| `serial_snoop.py` | 觀察序列埠輸出、統計每行浮點數個數與速率，用來確認資料格式 |
| `serial_autobaud.py` | 自動嘗試常見鮑率，找出 IMU 實際的 baud rate |
| `imu_raw.csv` | 範例錄製資料 |

## 安裝

```bash
pip install -r requirements.txt
```

## 使用範例（Windows，COM 埠請依實際情況修改）

```bash
# 找 IMU 用的鮑率
python serial_autobaud.py

# 即時解析封包並寫 CSV
python wit_imu_stream.py --port COM5 --baud 9600 --csv imu_raw.csv

# 錄 60 秒原始資料
python 1_collect.py --port COM5 --baud 115200 --duration 60 --out imu_raw.csv

# 離線視覺化（Madgwick 或 Mahony）
python 2_offline_viz.py --csv imu_raw.csv --fs 100 --filter madgwick

# 即時 3D 姿態
python realtime_viz.py --port COM5 --baud 115200 --fs 100

# 即時姿態 + WASD 體感控制（傾斜超過 10° 觸發，6° 死區）
python 3_realtime_viz_game.py --port COM3 --baud 9600 --pitch_th 10 --roll_th 10 --dead 6
```

`3_realtime_viz_game.py` 若裝置本身有輸出四元數，可加 `--prefer_quat` 直接使用裝置姿態而非自行解算。
