# 上游来源

- 仓库：https://github.com/google-deepmind/mujoco_menagerie
- 目录：`kuka_iiwa_14`
- 提交：`f3b6013629c7b94f8ee93fd977b161fa8350f9bc`
- 获取日期：2026-10-30
- 许可证：BSD-3-Clause，完整内容见 `LICENSE`。

本目录的 `iiwa14.xml`、网格、许可证及上游说明保持原样。
`../../arm.xml` 是基于上游机器人模型和场景的 demo 适配版本：

- 调整网格路径、相机、地面反射及场景取景范围。
- 将末端 attachment_site 命名为 ee，并显示红色标记。
- 添加无动力学自由度的绿色 mocap 目标球。
- 设置 0.002 s 物理步长，启用连杆重力补偿并关闭接触。
- 保留原有关节定义、执行器、惯性参数和 home 姿态，不做 q 映射。

模型替换仅涉及新建的 demo/IK，控制算法仍留空。
