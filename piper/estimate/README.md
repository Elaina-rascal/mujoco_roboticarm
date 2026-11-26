# 启动摩擦估计

`FrictionEstimator` 只使用给定力矩和关节角度，不读取反馈力矩，不自行连接机械臂。

```python
from piper.estimate.friction_estimator import FrictionEstimator

# arm 是已连接、使能的新版 PiperInterface，control 提供 gravity(q) 和模型限位。
estimator = FrictionEstimator(arm, control)
result = estimator.measure(3)  # 测 J3，关节编号为 1～6
print(result)
```

流程：当前位置静止 → 当前姿态重力补偿 → 每隔 0.5 s 增加 0.05 N·m → 检测到转动时记下启动阈值 → 停止加档并低速再走设定行程（当前 3°） → 慢速回到原姿态 → 反向重复 → 回位。

测试轴不加位置 PD；其他五轴用 PD 保持起始位置。默认目标速度 0.5°/s，超过 3°/s 退出；附加力矩上限 1 N·m，单向或回位超时 60 s。启动后的速度反馈只用于调速，不计入启动阈值。速度调节不能保证真实硬件绝不超速，检测超速后会切回当前位置保持；没有运行真机测试。起始姿态需远离关节限位，且测试期间无外部接触。

返回的 `positive`、`negative` 是相对模型重力的有符号启动附加力矩（N·m）：

```text
friction = (positive - negative) / 2
bias = (positive + negative) / 2
```

`friction` 是近似双向对称条件下的等效启动摩擦；`bias` 反映残余偏置。它们基于指令而非实测轴力矩，仍受重力模型、接口量化和启动检测延迟影响；不等于滑动摩擦或完整摩擦模型。参数直接在类的初始化中修改。

先运行离线 MuJoCo 示例：`python3 -m piper.sim.estimate.friction_estimation --joint 3`。结果及模型配置见 [仿真说明](../sim/estimate/README.md#双向启动摩擦估计)。估计类默认使用真实时间；仿真入口仅把 `clock` 和 `sleep` 换成仿真时间与物理步进。
