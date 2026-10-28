# mujoco_roboticarm

UR5e 机械臂 MuJoCo 仿真，通过 Python 函数直接设置关节位置或力矩、读取状态，支持 Pinocchio 逆运动学。无需 ROS 2、功能包构建或 launch。

## 安装

建议在 Python 虚拟环境中安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install mujoco numpy
```

IK 模式额外需要支持 MJCF 的 Pinocchio：

```bash
python3 -m pip install pin
```

## 一键启动

```bash
python3 run.py
```

默认打开 MuJoCo 窗口，以模型的 home 姿态进行位置控制。关闭窗口或按 Ctrl+C 退出。
脚本可以从任意目录通过完整路径调用。

启动脚本不接收命令行参数。直接在 `run.py` 中修改控制逻辑：

- `MODE = "position"`：保持 home 姿态，通过 `arm.set_control([...])` 设置六个关节目标角（rad）。
- `MODE = "torque"`：通过 `arm.set_control([...])` 设置六个关节力矩（N·m），默认全零，不补偿重力。
- `MODE = "ik"`：直接调用 `IKController`，在构造处修改目标点及圆轨迹设置。

关节顺序为 `shoulder_pan_joint`、`shoulder_lift_joint`、`elbow_joint`、`wrist_1_joint`、`wrist_2_joint`、`wrist_3_joint`。
控制量按执行器范围限幅。需要状态时直接调用 `arm.joint_state()`。
IK 保持末端单位旋转，半径默认为 0；目标不可达时返回迭代结果，不保证收敛。

## Python 直接控制

在仓库根目录执行，或将 `src` 加入 Python 模块路径：

```python
import sys
sys.path.insert(0, "src")
from mujoco_arm.simulation import MujocoArm

arm = MujocoArm(mode="position")
arm.set_control([0, -1.57, 1.57, -1.57, -1.57, 0])
for _ in range(1000):
    arm.step()
print(arm.joint_state())
print(arm.body_transforms())  # 世界系位姿，四元数顺序 w, x, y, z
```

力矩模式使用 `MujocoArm(mode="torque")`，通过相同的 `set_control()` 接口输入力矩。

## 目录

```text
run.py           一键启动入口
models/universal_robots_ur5e/  模型、网格与上游许可证
src/mujoco_arm/           仿真循环、控制与状态读取
src/control_py/           Pinocchio IK 求解器与直接调用控制器
src/control/              可选的独立 C++ IK 示例与实验脚本
```

## 可选 C++ 示例

Python 仿真不需要 CMake。安装 Pinocchio C++ 开发依赖后，可单独编译 IK 示例；它读取关节角并输出求解结果：

```bash
cmake -S . -B build/cpp -DBUILD_CPP_IK=ON
cmake --build build/cpp
./build/cpp/src/control/invkinematic
```

容器配置仅保持开发环境运行；进入容器后执行同样的 `python3 run.py` 命令。

## 验证

安装基本依赖后执行：

```bash
python3 -m unittest discover -s tests -v
```

覆盖位置跟踪、力矩映射及限幅、输入检查。
