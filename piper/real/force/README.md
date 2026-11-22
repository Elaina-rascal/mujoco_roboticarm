# 真机 MIT、XYZ VMC 阻抗与 M(q) PD

项目根目录运行其中一个入口：

```bash
python3 -m piper.real.force.mit
python3 -m piper.real.force.vmc
python3 -m piper.real.force.Mq_pd
```

MIT 和 VMC 均使用 `piper/interface/piper_new.py` 的 `pyAgxArm` 接口（容器内安装 1.0.0），与 `real` 下的位置控制和辨识采集入口一致。新接口先连接 SDK、等待接收帧率大于零，再通过 `sdk.get_firmware(timeout=3.0, min_interval=0.0)` 取得软件版本；无完整回复时最多请求三次，并打印重试进度。接收等待每次最多三秒；未收到有效反馈或出现 CAN 通信错误时直接退出。这样避免 SDK 在冷启动帧率仍为零时清空版本回复，也避免一次回复丢失后只等待、不重发。严格要求固件 **> 1.8.8** 后才设置控制模式；`S-V1.8-8` 及以下、未知版本或最终查询超时均断开并拒绝启动。版本查询与报文解析完全交给 SDK，不自行发送 CAN 查询或解包。符合门槛的固件使用 `v189` 配置，当前 `S-V1.9-1` 对应此配置。

六轴前馈以输出端 N·m 交给 `move_mit(t_ff=...)`，分轴缩放和编码由新版 SDK 完成；外层不统一除以 4，也不套用 1.5-8 的反号。反馈直接读取 SDK 的 `torque`（N·m），不重复乘系数。此前的前馈对比、JSONL 诊断和 CAN 采集实验代码已移除，结论保存在 [力矩前馈调查记录](TORQUE_FEEDFORWARD.md)。

两个入口复用 `vmc.py` 中的启动函数，调用现有 `PinocchioControl.ik(q, INITIAL_POSITION)`，以普通位置模式、10% 速度到达 `[0.249335, 0.0308932, 0.266156] m` 后再切换到力控制。IK 沿用工具水平、yaw 自由的约束，不跟踪固定 R。IK 拒绝时不发送运动指令，30 秒未到达则退出。

MIT 模式用重力前馈和关节 PD 保持初始化得到的关节角，增益在 `mit.py` 中为 `KP=0.5`、`KD=0.01`。

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

空间阻抗由 XYZ 控制律计算。本次接口迁移保留 VMC 当前传入的 MIT `kp=0.1`、`kd=1e-6`；它们不是有意义的关节位置保持增益，其中 `kp=0.1` 小于 SDK 的最小编码步长，会量化为零。速度来自实际关节角按实际时间间隔差分，经过 10 Hz 一阶低通，再通过雅可比换算为末端速度。

`[20, 20, 20] N` 可同时输出，不按合力模长缩放。这是计算的空间力指令限幅，不是测量的接触力上限；关节指令还包含重力补偿，并受接口力矩饱和限制。控制和显示使用同一份 `arm_official.xml`，夹爪按模型中性姿态固定。


`Mq_pd.py` 是独立的关节空间 M(q) PD 入口，复用 VMC 的位置就位流程和数字孪生。目标球变化时先调用五自由度 IK 求目标关节角；IK 未通过检查则保持上一次关节目标。随后在每个 200 Hz 控制周期计算：

```text
e = q_target - q
tau = M(q) @ (KP * e - KD * dq) + C(q,dq) @ dq + g(q)
```

力矩计算调用 `PinocchioControl.computed_torque()`；KP=100 s^-2、KD=20 s^-1，六轴参数在 `Mq_pd.py` 顶部配置。dq 为关节角按实际时间差分、经 10 Hz 低通得到的估计值。目标关节速度为零；IK 虽不约束 yaw，但得到具体关节解后，六个关节均跟踪该解。力矩通过新接口以 N·m 发送，MIT `kp=kd=0`，避免重复叠加关节 PD；指令由接口按 SDK 力矩范围限幅。打印给定值为接口限幅前的力矩，反馈为 SDK 的电流估算值。退出或异常时关闭数字孪生，并调用接口的位置保持退出逻辑。
