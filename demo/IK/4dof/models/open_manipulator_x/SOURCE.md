# 模型来源与修改

- 上游：ROBOTIS-GIT/open_manipulator
- 地址：https://github.com/ROBOTIS-GIT/open_manipulator
- 提交：b83daec7b98c08d00adb856d7e31f2b0065d3be9
- 获取日期：2026-11-02
- MJCF 路径：open_manipulator_description/mujoco/open_manipulator_x/open_manipulator_x.xml
- 网格路径：open_manipulator_description/meshes/open_manipulator_x
- 许可证：Apache-2.0，完整内容见 LICENSE。

本目录 original.xml、assets 中的 STL 网格保持上游原样。
../../arm.xml 为 demo 适配版本，已标注修改：

1. 删除夹爪的两个滑动关节、夹爪执行器及 mimic 约束，夹爪固定在零位。
2. 保留四个原有关节、其位置执行器、关节限位及惯性参数。
3. 添加 ee site、绿色 mocap 目标球、地面和相机取景设置。
4. 使用 0.002 s 步长、implicitfast 积分、理想重力补偿，关闭接触。
5. 设置 home = [0, -0.4, 0.5, -0.1] rad，调整可视网格颜色提高可读性。
6. 将网格引用改为相对本 demo 的路径。

仿真只使用 MJCF/STL 数据，不依赖上游 ROS 功能包或 launch。
