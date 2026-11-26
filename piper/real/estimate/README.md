# 真机重力辨识数据采集

## 六关节启动摩擦测试

```bash
python3 -m piper.real.estimate.friction_estimation
```

使用新版接口依次测试 J1～J6。先手动选好无外部接触、无碰撞且六轴均远离限位的起始姿态；程序检查各轴双向行程，不自动寻找姿态。每档增加 0.05 N·m，记录启动阈值后以约 0.5°/s 继续走设定行程（当前 3°），慢速回位，再测反向；取双向阈值的半差。只用指令力矩和角度，不读反馈力矩，方法及参数见 [共用估计类](../../estimate/README.md)。

完成一个关节并回位后再测下一个，终端逐轴打印结果。附加力矩上限 1 N·m、速度上限 3°/s、单向或回位超时 60 s；任一轴失败则停止后续测试。Ctrl+C、异常或结束时切回位置保持；最终打印已完成的结果。本次仅新增入口，未在真机运行。

## 重力辨识

真机采集入口统一使用 `piper.interface.piper_new.PiperInterface`（`pyAgxArm`），通过 SDK 查询主控版本，仅允许固件严格高于 1.8.8。目标角使用 rad，反馈力矩直接采用 SDK 提供的 N·m，不额外叠加旧接口的 `/4` 或方向换算。离线参数拟合不连接硬件。

用已采集的 `data/gravity.npz` 做 Y 矩阵参数更新并打印前后变化：

```bash
python3 -m piper.real.estimate.gravity_estimator
```

`GravityEstimator` 使用 `Y(q, 0, 0)`，只更新各连杆的 `m、m*cx、m*cy、m*cz`，不辨识转动惯量。求解 `Y*delta = tau - Y*原参数`，再加上修正量；最小范数解保留不可辨识方向的原值。打印回归秩、更新前后参数、变化量及拟合 RMSE。质量单位 kg，一阶质量矩单位 kg·m。

当前试验按反馈只有幅值处理：`tau_signed = abs(tau_feedback) * sign(g_XML(q))`，逐样本、逐关节补方向。`g_XML` 始终使用对象初始化时加载的原始 XML 参数（当前为 `ARM_MODEL` / `arm.xml`），重复拟合也不会改用辨识参数决定符号。先取绝对值会消除采集接口此前人为加的逐轴固定符号；不修改 npz、真机读取接口或 XML。

对于 `|g_XML| ≤ 0.001 N·m` 的观测，不使用模型猜测方向并从拟合中剔除，包括理想情况下的 J1；不会把实测反馈强制变成零。脚本同时输出参与拟合观测的补方向 RMSE，以及全部有效观测的幅值 RMSE，后者保留近零重力轴的误差。方向是模型推断值，模型方向错误时会带来系统误差。

重力数据只能确定部分参数组合，所以结果是等效重力参数，不能当作唯一的真实质量与质心；不保证物理惯性的可行性。更新值保存在类的 `parameters` 中，`gravity(q)` 使用这些参数计算重力，不修改原 XML 或 Pinocchio 模型的惯性。夹爪按现有真机控制器固定为模型中性姿态，采样时应保持相应开度与负载。本次补符号仅适用于静态、主要由重力决定方向的观测，不等同于获得真实有符号力矩反馈。

```python
import numpy as np

from piper.real.estimate.gravity_estimator import GravityEstimator

with np.load("piper/real/estimate/data/gravity.npz") as data:
    estimator = GravityEstimator().fit(data["q"], data["tau"])
    gravity_torque = estimator.gravity(data["q"][0])
```

在项目根目录运行：

```bash
python3 -m piper.real.estimate.trajectory_demo
```

使用普通位置模式，先从当前位置平滑移动到起点，再按六个姿态循环移动。每段移动 3 秒、停留 2 秒，只在停留的最后 1 秒采集**实测角度和力矩**。Ctrl+C 退出时保持当前位置，并将数据保存到本目录 `data/gravity.npz`（再次采集会覆盖）；异常退出时也会尝试保存已有数据。没有样本时不生成文件。

npz 内容：

| 字段 | 内容 |
| --- | --- |
| `t` | `(N,)` 相对采样时间，单位 s |
| `q` | `(N, 6)` 实测关节角度，单位 rad |
| `tau` | `(N, 6)` 实测电流估算力矩，单位 N·m |
| `joint_names` | 六列依次为 `joint1`～`joint6` |
| `firmware` | 当前真机固件版本 |
| `torque_source` | `motor_current_estimate`，表明力矩来源 |

不保存速度、加速度或目标角度。`trajectory.py` 仅在本真机包内生成目标轨迹，不依赖或改动仿真 demo。

读取数据：

```python
import numpy as np

with np.load("piper/real/estimate/data/gravity.npz") as data:
    q = data["q"]
    tau = data["tau"]
```

重力辨识使用静态关系 `tau ≈ g(q)`。采样时间是读取 SDK 缓存时的主机时间，不是硬件同步时间。停留窗口按目标轨迹划分，未通过速度反馈验证静止；若真机到位较慢，需要延长停留时间。力矩沿用接口的电流估算、符号和固件限制；反馈缺失轴保留 NaN，辨识前应剔除。这些观测尚未做摩擦或外力补偿。
