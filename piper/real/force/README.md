# 真机 MIT 与 XYZ VMC 阻抗

项目根目录运行其中一个入口：

```bash
python3 -m piper.real.force.mit
python3 -m piper.real.force.vmc
```

两个入口复用 `vmc.py` 中的启动函数，调用现有 `PinocchioControl.ik(q, INITIAL_POSITION)`，以普通位置模式、10% 速度到达 `[0.249335, 0.0308932, 0.266156] m` 后再切换到力控制。IK 沿用工具水平、yaw 自由的约束，不跟踪固定 R。IK 拒绝时不发送运动指令，30 秒未到达则退出。

MIT 模式用重力前馈和关节 PD 保持初始化得到的关节角，增益在 `mit.py` 中为 `KP=5.0`、`KD=0.01`。

VMC 模式的目标球从给定 p 开始。在数字孪生窗口双击目标球，按住 Ctrl + 鼠标右键拖动，改变世界系 XYZ 目标。运行阶段只控制平移，姿态自由；工具水平、yaw 自由仅用于启动就位。关闭窗口或 Ctrl+C 时，沿用真机接口的退出位置保持逻辑。

阻抗力计算、逐轴限幅和重力叠加在 `vmc.py` 的 `compute_impedance()` 中；`PinocchioControl.force_to_torque(q, force)` 只负责 `J_xyz.T @ force`：

```text
F = clip(kp * (target - position) - kd * velocity, -max_force, max_force)
tau = g(q) + J_xyz.T @ F
```

空间力参数在 `vmc.py` 顶部设置：

| 参数 | XYZ 默认值 | 单位 |
| --- | --- | --- |
| `KP` | `[60, 60, 60]` | N/m |
| `KD` | `[3, 3, 3]` | N·s/m |
| `MAX_FORCE` | `[20, 20, 20]` | N，每轴独立限幅 |

没有额外的关节 MIT PD，VMC 入口的 MIT kp/kd 为零。速度来自实际关节角按实际时间间隔差分，经过 10 Hz 一阶低通，再通过雅可比换算为末端速度。

`[20, 20, 20] N` 可同时输出，不按合力模长缩放。这是计算的空间力指令限幅，不是测量的接触力上限；关节指令还包含重力补偿，并受接口力矩饱和限制。控制和显示使用同一份 `arm_official.xml`，夹爪按模型中性姿态固定。


诊断时分别运行（前一个退出后再运行下一个）：

```bash
python3 -m piper.real.force.mit --diagnostics /tmp/piper-pd.jsonl
python3 -m piper.real.force.vmc --diagnostics /tmp/piper-vmc.jsonl
```

日志追加保存计算力矩、SDK 实参（含目标角、kp/kd、t_ref）、原始电流及其同副本换算值、控制模式和 SDK 时间戳。记录频率沿用各入口的打印频率，不改变控制律。SDK 参数记录不代表 CAN 线上编码值或驱动器接收确认；各类反馈来自异步缓存，应先比较稳定片段，不把当前指令与刚读到的反馈视为严格同步的一对。
