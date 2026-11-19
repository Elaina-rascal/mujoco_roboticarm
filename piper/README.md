# PiPER

```text
piper/
├── controllers/
│   └── pinocchio_control.py   # 真机用 FK / IK / 重力计算
├── interface/
│   ├── base.py                # 真机与仿真的四个共同方法
│   ├── piper.py               # PiperInterface：SDK、CAN、单位和方向换算
│   └── mujoco.py              # MujocoInterface：位置 / 力矩仿真
├── models/                    # XML、URDF、网格和来源说明
├── sim/
│   ├── position/              # run.py 和本场景的 controller.py
│   ├── torque/                # run.py 和本场景的 controller.py
│   └── estimate/    # 离线重力参数辨识
└── real/
    ├── position/run.py
    ├── torque/run.py
    ├── estimate/    # 真机轨迹和重力辨识数据采集
    └── digital_twin.py        # 真机反馈显示和目标球交互
```

在项目根目录按模块运行：

```bash
python3 -m piper.sim.position.run
python3 -m piper.sim.torque.run
python3 -m piper.sim.estimate.gravity_identification
python3 -m piper.sim.estimate.trajectory_demo

# 以下命令会连接并控制真机，需要先配置 CAN 和安装 piper_sdk。
python3 -m piper.real.position.run
python3 -m piper.real.torque.run
python3 -m piper.real.estimate.trajectory_demo
```

仿真和控制器依赖 `numpy`、`mujoco`、支持 MJCF 的 `pin`；导入真机入口不连接硬件。

接口统一继承 `ArmInterface`，关节按 `joint1`～`joint6` 排列：

| 方法 | 单位 |
| --- | --- |
| `send_joint_positions(positions)` | rad |
| `send_joint_torques(torques)` | 输出端 N·m |
| `read_joint_positions()` | rad |
| `read_joint_torques()` | 输出端 N·m |

`MujocoInterface(mode="position")` 用于位置控制，`mode="torque"` 用于直接力矩控制。仿真步进、目标球和夹爪交互也在这个类中；不提供使能和断连方法。仿真力矩反馈是执行器力矩，不含位置模式下模型额外施加的自动重力补偿。

`PiperInterface` 额外管理真机连接、`enable()`、`has_feedback()` 和 `close()`。力矩通过 MIT 发送，默认纯前馈；真机力矩入口还传入原有的保持角度和 PD 增益。固件校验、1.5-8 的方向与缩放、退出时位置保持逻辑保留。真机力矩反馈仍为电流估算值。

控制器按各自场景使用，不统一 `compute` 签名，也不依赖具体硬件接口。位置仿真保留原来的姿态约束，力矩仿真保留全模型重力计算。真机位置控制继续使用 `models/arm.xml`；真机力矩和两个仿真入口继续使用 `models/arm_official.xml`，模型参数未修改。

验证（不连接真机）：

```bash
python3 -m unittest discover -s tests -p 'test_piper_interfaces.py' -v
```
