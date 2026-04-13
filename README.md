# mujoco_roboticarm

## MuJoCo + ROS 2 机械臂角度发布示例

已新增 ROS 2 包 `mujoco_arm_publisher`，包含：

- 六自由度机械臂示例
- ROS 2 节点，周期推进仿真并发布 6 个关节角度和角速度
- launch 文件，一键启动

节点会优先尝试加载 MuJoCo 安装目录中的机械臂模型（如 `ur5e`/`panda` 等）。
如果本机未找到可用 6 轴模型，则自动回退到仓库内置模型：`models/six_dof_arm.xml`。

### 目录

```text
src/mujoco_arm_publisher/
├── launch/mujoco_arm.launch.py
├── models/six_dof_arm.xml
└── mujoco_arm_publisher/arm_joint_publisher_node.py
```

### 依赖

1. ROS 2（Humble/Iron/Jazzy 均可）
2. MuJoCo Python 包

安装 MuJoCo Python 包：

```bash
pip install mujoco
```

### 构建

在工作区根目录执行：

```bash
colcon build --packages-select mujoco_arm_publisher
source install/setup.bash
```

### 运行

```bash
ros2 launch mujoco_arm_publisher mujoco_arm.launch.py
```

### 查看角度数据

新开一个终端：

```bash
source install/setup.bash
ros2 topic echo /mujoco/joint_states
```

你会看到 `sensor_msgs/JointState`，其中：

- `name`: 6 个关节名
- `position`: 6 个关节当前角度（弧度）
- `velocity`: 6 个关节当前角速度（弧度/秒）
