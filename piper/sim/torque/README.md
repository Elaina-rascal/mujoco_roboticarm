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

默认 Kp=100 s^-2、Kd=20 s^-1，控制周期 0.01 s，夹爪固定总开度 0.04 m。目标关节速度为零；当前版本不计算运动目标的速度和加速度前馈。M、科氏/离心项和重力按当前状态计算，采用与仿真相同的官方模型和固定夹爪开度。接口根据模型执行器范围限幅。

可在机械臂上施加鼠标扰动，松开后返回关节目标。原 `controller.py` 中的 `GravityController` 保留作为独立纯重力补偿工具；本入口使用 `PinocchioControl.computed_torque()`。
