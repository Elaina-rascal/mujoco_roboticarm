# 官方 PiPER 动力学模型（仿真）

`../arm_official.xml` 使用以下官方参数，独立于原来的 `../arm.xml`。

- 六轴： https://github.com/agilexrobotics/agilex-arm-gravity-compensation/blob/8e2545c00ec9ec7590414946ab5f913274c7b643/piper/piper/urdf/piper_description.urdf
- 夹爪及双指： https://github.com/agilexrobotics/piper_ros/blob/humble/src/piper_description/urdf/piper_description.urdf （2026-11-18 获取）
- STL 来自上述两个项目对应的 meshes 目录，保留原文件。
- `source.urdf` 保存合并后的来源参数，未修改质量、质心和惯量；六轴关节坐标使用动力学例程的版本。

六轴质量 3.14 kg；官方夹爪主体基线 0.45 kg；双指各 0.025 kg。
`source.urdf` 保留官方原始参数（夹爪主体 0.45 kg），供对照。

## 当前末端质量试验

按用户指定，以官方新版模型为基础，仅把 gripper_base 质量设为 **1.2 kg**。
质心和完整惯量矩阵恢复为官方基线，不沿用此前 0.67 kg 试算的惯量缩放。
link6 仍为 0.007 kg，双指各 0.025 kg，J6 下游总质量 **1.257 kg**。
不包含 D435、支架或额外电机，不引用自定义相机 XML。
这是用户指定的质量试验值，不是辨识或称重结果。
`source.urdf` 保留官方基线（gripper_base 0.45 kg）。

转换时保留完整惯量矩阵、关节原点/轴/限位；URDF 的 RPY 转为 MuJoCo 的 wxyz 四元数。
补充现有仿真需要的 ee、target、home、位置执行器和双指联动约束。
场景和执行器增益沿用现有仿真；armature=0.005 是现有仿真设置，不是官方标定值。
位置场景开启 body gravcomp；力矩场景会关闭它并由 Pinocchio 提供 g(q)。

运行：

```bash
python3 -m piper.sim.position.run
python3 -m piper.sim.torque.gravity
```

`piper.real.force.mit`、`piper.real.force.vmc` 的动力学及数字孪生、位置和力矩仿真使用 `arm_official.xml`；
`piper.real.position.run` 默认仍使用原 `arm.xml`。
