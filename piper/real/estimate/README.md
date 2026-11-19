# 真机重力辨识数据采集

在项目根目录运行：

```bash
python3 -m piper.real.estimate.trajectory_demo
```

使用普通位置模式，先从当前位置平滑移动到起点，再按六个姿态循环移动。每段移动 3 秒、停留 2 秒，只在停留的最后 1 秒采集**实测角度和力矩**。Ctrl+C 退出时保持当前位置，并将数据保存到本目录 `data/gravity.npz`（再次采集会覆盖）；异常退出时也会尝试保存已有数据。没有样本时不生成文件。

npz 内容：

| 字段 | 内容 |
| --- | --- |
| `t` | `(N,)` 相对采样时间，单位 s |
| `q` | `(N, 6)` 实测关节角度，单位 rad |
| `tau` | `(N, 6)` 实测电流估算力矩，单位 N·m |
| `joint_names` | 六列依次为 `joint1`～`joint6` |
| `firmware` | 当前真机固件版本 |
| `torque_source` | `motor_current_estimate`，表明力矩来源 |

不保存速度、加速度或目标角度。`trajectory.py` 仅在本真机包内生成目标轨迹，不依赖或改动仿真 demo。

读取数据：

```python
import numpy as np

with np.load("piper/real/estimate/data/gravity.npz") as data:
    q = data["q"]
    tau = data["tau"]
```

重力辨识使用静态关系 `tau ≈ g(q)`。采样时间是读取 SDK 缓存时的主机时间，不是硬件同步时间。停留窗口按目标轨迹划分，未通过速度反馈验证静止；若真机到位较慢，需要延长停留时间。力矩沿用接口的电流估算、符号和固件限制；反馈缺失轴保留 NaN，辨识前应剔除。这些观测尚未做摩擦或外力补偿。
