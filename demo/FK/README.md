# FK 小练习

使用 NumPy 和 SciPy（`python3 -m pip install numpy scipy`）。`lie.py` 提供基础变换；`fk.py` 留有待你实现的 `fk(q, axes, home)` 接口和四组几何核对例子。

```bash
python3 demo/FK/fk.py
```

先实现 `fk()` 再运行。检查会同时比较末端的位置与朝向。

约定：

- 使用右手坐标系和列向量，角度为 rad，长度为 m。
- `T_AB` 把 B 系坐标变为 A 系坐标：`p_A = T_AB @ p_B`（齐次坐标）。
- 六维向量顺序是 `[w, v]`，`hat()` 返回李代数矩阵，`exp_se3()` 返回群元素。
- `axes` 的每一行是**零位时在世界坐标系中表达**的旋量轴，练习使用空间旋量约定。
- 转动关节的 `w` 是单位轴向量，轴上一点为 `r` 时 `v = -np.cross(w, r)`。
- 移动关节的 `w = 0`，`v` 是单位移动方向。
- `exp_se3(S * q)` 中的 `q` 已乘入整个六维向量；不能只乘 `w`。

可直接调用：

```python
import numpy as np
from lie import skew, hat, exp_so3, exp_se3, make_transform, inverse_transform

R = exp_so3([0, 0, np.pi / 2])
T = make_transform(R, [1, 0, 0])
point = T @ [1, 0, 0, 1]  # [1, 1, 0, 1]
T_inv = inverse_transform(T)

S = np.array([0, 0, 1, 0, -1, 0])  # 绕经过 [1, 0, 0] 的 z 轴旋转
joint_transform = exp_se3(S * (np.pi / 2))
```

`v` 是旋量的线速度部分，通常不能直接当成变换矩阵中的平移。
`lie.py` 中的实现也可以逐行读，与自己推导的公式对照。

## 世界坐标中的速度、位置转旋量

```python
from scipy.linalg import expm
from lie import world_to_twist, rotation_axis, twist_to_axis, hat, exp_se3

# w 是角速度，p 是刚体上一点的世界位置，v 是该点的世界线速度。
w = [0, 0, 2]
p = [1, 0, 0]
v = [0, 0, 0]
xi = world_to_twist(w, v, p)  # [0, 0, 2, 0, -2, 0]
X = hat(xi)                  # 4×4 se(3) 矩阵
T = exp_se3(xi * 0.1)        # 恒定空间旋量作用 0.1 秒的 SE(3) 变换
# 等价：T = expm(X * 0.1)

# 已知固定旋转轴的世界方向和轴上一点：
S = rotation_axis([0, 0, 1], [1, 0, 0])
T_joint = exp_se3(S * (np.pi / 2))
direction, point, pitch = twist_to_axis(S)
# direction = [0, 0, 1]，point = [1, 0, 0]，pitch = 0
```

这里 `ws = w`、`vs = v - w × p` 是同一世界坐标系内，将点速度转换成空间旋量；
`v` 不是坐标系原点处的速度，也不是位置。若输入来自局部坐标系，需要先把向量旋转到世界系。
只有一个世界位置或一个 SE(3) 位姿，无法唯一确定关节轴，还需要轴方向或运动信息。

区分三个对象：六维旋量 `xi`、李代数矩阵 `hat(xi)`（se3）、指数得到的刚体变换 `exp_se3(xi)`（SE3）。
SE3 矩阵本身不做 hat。这里没有实现 SE3 的对数映射。

检查基础函数：

```bash
python3 -m unittest discover -s demo/FK -p 'test_*.py' -v
```
