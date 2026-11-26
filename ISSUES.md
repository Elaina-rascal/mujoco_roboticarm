# 全局问题记录

## MuJoCo：无积分时，关节仍在静摩擦下缓慢爬行



对于独立单关节、仅考虑这一条干摩擦约束，可写为：

$$
\tau_f=
\operatorname{clip}\left(
\frac{a_{\mathrm{参考}}-a_{\mathrm{自由}}}{1/I+R},
-\tau_{\max},\tau_{\max}
\right),\qquad
a_{\mathrm{参考}}=-b\dot q
$$

| 变量 | 含义及模型参数 |
| --- | --- |
| $I$ | 单关节等效惯量 |
| $a_{\mathrm{自由}}$ | 不计该干摩擦时，由电机、重力等产生的加速度；不是先执行一次运动 |
| $\tau_{\max}$ | 摩擦力矩上限，对应 `frictionloss` |
| $R$ | 约束软化量，$R=(1-d_0)\hat A/d_0$，其中 $d_0=\text{solimpfriction[0]}$，$\hat A$ 为逆惯量估计 |
| $b$ | 速度抑制强度；正数格式下 $b=2/(d_w t_c)$，$d_w=\text{solimpfriction[1]}$，$t_c=\text{solreffriction[0]}$ |

`solreffriction` 第二项对干摩擦不起作用；`solimpfriction` 后三项不影响这条零位置残差的摩擦约束。关节 XML 的 `damping` 是另加的黏性阻尼，不是公式中的 $b$。

**为什么会爬行：** 零速时，若剩余驱动力矩为 $u$，则 $a_{\mathrm{自由}}=u/I$。未触及摩擦上限时，$\tau_f=-u/(1+IR)$。只要 $R>0$，它就不能完全抵消 $u$；产生少量速度后，$-b\dot q$ 再参与求解，可能形成非零稳态爬行。

**当前处理：** 仿真模型 `piper/models/arm_mujoco.xml` 使用：

```xml
solimpfriction="0.9999 0.9999 0.001 0.5 2"
solreffriction="0.004 1"
```

增大 $d_0$ 减小 $R$，减小 $t_c$ 增大 $b$，均可降低爬行；不改变摩擦上限。当前步长为 0.002 s，时间常数取其两倍。MuJoCo 将阻抗限制在 `[0.0001, 0.9999]`，写 `1` 也不会得到严格硬约束。

### 默认参数与当前参数代入

沿用单关节简化近似，取剩余驱动力矩 $u=0.1$ N·m、摩擦上限 $\tau_{\max}=0.36$ N·m，当前速度为零。公式进一步简化为：

$$
\tau_f\approx\operatorname{clip}(-d_0 u,-\tau_{\max},\tau_{\max})
$$

| 项目 | MuJoCo 默认约束参数 | 当前约束参数 |
| --- | --- | --- |
| `solimpfriction` | `0.9 0.95 0.001 0.5 2` | `0.9999 0.9999 0.001 0.5 2` |
| `solreffriction` | `0.02 1` | `0.004 1` |
| $b=2/(d_w t_c)$ | 105.263 s⁻¹ | 500.050 s⁻¹ |
| 零速时 $a_{\mathrm{参考}}=-b\dot q$ | 0 | 0 |
| 摩擦力矩 $\tau_f$ | $-0.9\times0.1=-0.09$ N·m | $-0.9999\times0.1=-0.09999$ N·m |
| 剩余净力矩 $u+\tau_f$ | **0.01 N·m** | **0.00001 N·m** |

两组均未达到 0.36 N·m 的摩擦上限。当前参数使零速时剩余净力矩降为默认值的 **1/1000**；这不是把摩擦上限调大。零速瞬间 $b$ 不贡献阻力；开始滑动后，当前更大的 $b$ 会通过 $a_{\mathrm{参考}}=-b\dot q$ 更强地抑制速度。此表只比较约束参数，摩擦上限在两列中保持相同，并非原模型的默认零摩擦。

**适用范围：** 上式是单关节解释式。完整机械臂通过 $A=JM^{-1}J^T$ 耦合求解各约束力，不能把无约束解逐轴 `clip` 当作一般解；求出摩擦后再计入总力矩、计算加速度并积分。

原文：[摩擦约束](https://mujoco.readthedocs.io/en/4.0.1/computation/index.html#friction-loss)、[约束力求解](https://mujoco.readthedocs.io/en/4.0.1/computation/index.html#dual-problem)、[参数映射](https://mujoco.readthedocs.io/en/4.0.1/modeling.html#friction)。
