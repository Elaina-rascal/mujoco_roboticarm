# IK + M(q) 关节空间力矩控制

`computed_torque.py` 为新增的 M(q) 控制仿真；`gravity.py` 保留原纯重力补偿仿真，二者分别运行。

```bash
python3 -m piper.sim.torque.computed_torque
python3 -m piper.sim.torque.computed_torque --headless --duration 5
# 原纯重力补偿仿真：
python3 -m piper.sim.torque.gravity
```

双击绿色目标球，Ctrl + 右键拖动位置。启动先用 IK 求初始关节配置；目标球变化后，调用 `PinocchioControl.ik(q, target_position)`，按已有五自由度约束求解 XYZ 和水平姿态，yaw 自由。IK 失败时保留上一次有效关节目标。

每个控制周期计算关节误差，再调用 `piper/controllers/pinocchio_controller.py` 的控制方法：

```python
error = pin.difference(control.model, q, q_target)
torque = control.computed_torque(q, dq, error, kp=KP, kd=KD)
```

控制律为：

```text
ddq_command = Kp * error - Kd * dq
tau = M(q) @ ddq_command + C(q,dq) @ dq + g(q)
```

这里的 `e` 是六个关节的误差（rad），不是末端空间误差。五自由度只描述 IK 的末端约束；选定一个 IK 解后，六个关节都跟踪该解，因此不能把此模式理解为力矩层面释放 yaw。

默认 Kp=100 s^-2、Kd=20 s^-1，控制周期 0.01 s，夹爪固定总开度 0.04 m。目标关节速度为零；当前版本不计算运动目标的速度和加速度前馈。M、科氏/离心项和重力按当前状态计算，控制器采用原官方模型和固定夹爪开度，默认仿真被控对象另加参数偏差与摩擦（见下文）。接口根据模型执行器范围限幅。

可在机械臂上施加鼠标扰动，松开后返回关节目标。原 `controller.py` 中的 `GravityController` 保留作为独立纯重力补偿工具；本入口使用 `PinocchioControl.computed_torque()`。

## 仿真模型与控制模型分离

默认仿真被控对象为 `piper/models/arm_mujoco.xml`，独立复制自原官方模型；控制器的 IK、M、C、g 仍使用原 `arm_official.xml`。其他仿真入口和真机模型不变。使用 `--nominal` 可切回原模型做对照：

```bash
python3 -m piper.sim.torque.computed_torque --headless --duration 5
python3 -m piper.sim.torque.computed_torque --headless --duration 5 --nominal
```

新模型保持几何、质心位置、关节轴线及限位不变，将每个运动部件的质量和完整惯量张量按相同比例缩放，保留惯量的物理一致性：

| 部件 | 质量与惯量倍率 |
| --- | --- |
| link1 / link2 / link3 | 1.04 / 1.06 / 0.95 |
| link4 / link5 / link6 | 1.05 / 1.08 / 0.96 |
| gripper_base / link7 / link8 | 1.05 / 1.03 / 1.03 |

| 关节 | J1 | J2 | J3 | J4 | J5 | J6 |
| --- | --- | --- | --- | --- | --- | --- |
| frictionloss (N·m) | 0.24 | 0.50 | 0.44 | 0.30 | 0.36 | 0.15 |
| damping (N·m·s/rad) | 0.04 | 0.08 | 0.08 | 0.02 | 0.02 | 0.01 |

上述偏差与摩擦仅为仿真示例，并非实机辨识值；控制器不额外补偿这些新增摩擦。`frictionloss` 用于模拟干摩擦，`damping` 用于黏性阻力，未单独模拟最大静摩擦与滑动摩擦的差异。仿真仍关闭物体接触，本次用于验证自由空间模型失配与关节摩擦造成的误差。

### 静摩擦下的数值爬行

仅提高 `frictionloss` 并不保证关节在阈值内严格静止：MuJoCo 默认用软约束求解干摩擦（`solimpfriction[0]=0.9`、`solreffriction[0]=0.02 s`），可能出现持续低速爬行。新仿真模型为关节设置 `solimpfriction="0.9999 0.9999 0.001 0.5 2"` 和 `solreffriction="0.004 1"`，降低约束软化导致的爬行；0.004 s 为当前物理步长的两倍。控制器和干摩擦力矩上限保持不变。此设置更接近静摩擦卡住，但依然不是数学上的严格零速锁止。

同一初始关节偏差 `[0.03,0.02,-0.03,0.03,0.03,-0.03] rad`、相同 PD 下，MuJoCo 4.0.1 离线对比：20 s 时默认软约束的最大关节速度约 0.0528°/s，较硬约束约 0.0000610°/s。位置误差仍然存在，不能将数值爬行误认为积分补偿。
