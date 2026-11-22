# 六自由度笛卡尔阻抗仿真

在项目根目录、已安装 MuJoCo 和 Pinocchio 的环境中运行：

```bash
python3 -m piper.sim.impedance.run
# 无窗口运行，不连接 CAN：
python3 -m piper.sim.impedance.run --headless --duration 5
```

初始目标等于机械臂当前完整位姿，初始关节角选在非奇异位形。双击目标球选中，Ctrl + 右键平移，Ctrl + 左键旋转，Shift 切换操作平面。较长的坐标轴显示实际工具姿态，较短的坐标轴显示目标姿态。选中机械臂后可通过鼠标施加扰动，松开后回到目标附近。目标应在关节限位内可达；任意拖动产生的目标不保证可达，奇异位形附近也不能独立实现六个方向。

控制器位于 `piper/controllers/impedance_controller.py`，与 MuJoCo 和 CAN 接口无关：

```python
controller = ImpedanceController()
command = controller.compute(q, dq, target_position, target_rotation)
torque = command.torque
```

`target_rotation` 为工具 B 系相对世界系的 3×3 旋转矩阵；B 系与现有 `PinocchioControl.fk()` 和仿真实际坐标轴一致。位置原点为模型 `ee`。所有向量均在世界系表达，排列为 `[x, y, z, rx, ry, rz]`：

```text
e_position = p_target - p
e_rotation = Log(R_target @ R.T)
twist = J_world @ dq
wrench = clip(Kp * error + Kd * (target_twist - twist), -limit, limit)
tau = J_world.T @ wrench + gravity(q)
```

默认目标速度为零；目标移动时依靠误差追踪，程序化轨迹可传入 `target_twist` 提供目标线速度和角速度。此控制为弹簧阻尼加重力补偿，不包含指定任务空间惯性的逆动力学整形。仿真控制频率 500 Hz；固定夹爪总开度 0.04 m，并按相同开度计算重力。力矩经仿真接口按模型执行器限幅。

| 参数 | 平移 XYZ | 旋转 XYZ |
| --- | --- | --- |
| Kp | 60 N/m | 5 N·m/rad |
| Kd | 3 N·s/m | 0.5 N·m·s/rad |
| 逐分量限幅 | ±20 N | ±5 N·m |

完整姿态使用三个旋转误差分量，没有沿用位置 IK 的五维 `H` 投影，也不默认放开 yaw。旋转 Log 向量描述整体相对旋转，不是三个独立欧拉角；把某个增益置零仅改变该方向的控制作用，不能当作精确释放某个欧拉角约束。接近 180° 时轴角表示存在分支，不适合用突变目标跨越该处。

省略 `target_rotation` 时，控制器退化为 XYZ 阻抗，三个角方向的弹簧和阻尼力矩均为零，供原真机 VMC 使用。
