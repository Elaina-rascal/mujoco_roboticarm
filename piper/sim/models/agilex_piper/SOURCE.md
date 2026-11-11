# 模型来源

- 上游：https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/agilex_piper
- 仓库提交：`370c43965323c0944b433b202aac5058b5f907a9`
- 下载日期：2026-11-11
- 许可证：MIT，见本目录 LICENSE；保留上游 piper.xml、assets、README 和 CHANGELOG。
- 上游由 AgileX PiPER ROS 模型转换而来，详细过程见 README.md。

仿真使用 ../../arm.xml，为上游 piper.xml 的场景适配副本：

- 网格路径改为 models/agilex_piper/assets。
- 在 link6 局部坐标 [0, 0, 0.13503] 增加 ee site，作为两指之间的工具中心。
- 添加地面、灯光、天空、可拖动的 mocap 目标球。
- 物理步长设为 0.002 s，保留上游重力补偿，关闭接触、移除关节库仑摩擦，便于位置 IK 实验。
- home 夹爪改为总开度 0.04 m；保留六轴运动学、惯性、关节限位、位置执行器和双指联动约束。

本示例验证自由空间中的位置 IK，不用于验证碰撞或真实硬件动力学。
