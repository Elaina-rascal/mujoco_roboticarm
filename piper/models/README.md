# XML link 质心 / joint 查看器

在项目根目录运行，也可以直接运行 `piper/models/view_joints.py`：

```bash
python3 -m piper.models.view_joints
python3 -m piper.models.view_joints --model arm --pose home
python3 -m piper.models.view_joints --report
```

默认从零位显示 `arm_official.xml`，半透明外壳中仅标出 `link1` 的质心。`Tab` 在 link / joint 两种模式间切换，`J/K` 逐个选择对象，每次只显示一个对象的标记。按 `M` 切换 XML，保留姿态、视角及选择；`gripper_base` 仅存在于官方模型，切换到另一模型时改选 `link6`。脚本只执行正运动学，不推进物理仿真。

| 按键 | 功能 |
| --- | --- |
| `Tab` | 切换 link 质心 / joint 关节模式 |
| `L` / `V` | 直接进入 link / joint 模式 |
| `M` | 切换两个 XML |
| `J` / `K` | 选择当前模式下一个 / 上一个 link 或 joint |
| `[` / `]` | 调整选中 joint；link 模式调整其最近的上游关节，固定底座不移动；转动默认每次 5°，双指每次 1 mm |
| `Z` / `H` | 恢复零位 / 当前 XML 的 home 姿态 |
| `F` | 显示 / 隐藏当前对象所属连杆的局部 XYZ，默认关闭 |
| `T` | 半透明 / 不透明外壳 |
| `P` | 输出两个 XML 的完整关节定义、各 body 的质量及局部/世界质心 |

link 模式的粉色球精确位于所选 body 自身的质心，即 `data.xipos`；粉线连接 body 原点与质心。窗口显示 body 名称、质量（kg）和局部质心（m，对应 `<inertial pos>`）。这里显示各 body 自身参数，不是下游合成质心，也不是此前辨识得到的等效参数。官方模型的 `gripper_base` 单独显示，不合并到 `link6`。

joint 模式只显示所选关节的黄色正轴箭头：转动按右手定则，滑动沿箭头方向。打开 `F` 后，红/绿/蓝箭头为所属 body 的局部 X/Y/Z，不是惯性主轴；link 模式画在 body 原点，joint 模式画在关节锚点。

J7/J8 按 XML 的等式约束反向联动。为保证切换时姿态一致，键盘调整范围采用两个 XML 的关节限位交集；各自完整限位见报告。`--step` 修改转动步长，单位 deg。用鼠标旋转、平移、缩放视角，关闭窗口退出。

报告中的 `body.pos/quat` 是所属连杆相对父连杆的固定变换，四元数顺序为 wxyz；`joint.pos/axis` 是所属连杆局部坐标系中的关节锚点和正轴。MuJoCo 会在固定变换后应用关节运动。`world.anchor/+axis` 是当前姿态下的世界坐标结果。对比两套 XML 时，应同时看世界关节轴与连杆固定变换，不能只比较局部 `axis="0 0 1"`。

`--report --pose zero` 和 `--report --pose home` 可在没有显示器的环境中比较两套定义；终端中的长度单位为 m，转动角度转换为 deg。
