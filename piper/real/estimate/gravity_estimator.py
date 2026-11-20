"""用 Pinocchio 重力回归矩阵更新参数，只需要静态 q 和 tau。"""

from pathlib import Path

import numpy as np
import pinocchio as pin

from piper.models import OFFICIAL_MODEL


SIGN_TORQUE_EPS = 1e-3  # N·m；模型重力接近零时，不用它猜测方向。


class GravityEstimator:
    def __init__(self):
        full_model = pin.buildModelFromMJCF(str(OFFICIAL_MODEL))
        fingers = [full_model.getJointId(name) for name in ("joint7", "joint8")]
        # 与现有真机控制器一致，固定夹爪为模型中性姿态，保留六轴。
        self.model = pin.buildReducedModel(full_model, fingers, pin.neutral(full_model))
        self.data = self.model.createData()
        self.zero = np.zeros(self.model.nv)
        # 每根连杆的十个惯性参数中，只取前四个：m、m*cx、m*cy、m*cz。
        self.columns = [10 * i + k for i in range(self.model.njoints - 1) for k in range(4)]
        self.parameters = np.concatenate([
            self.model.inertias[i].toDynamicParameters()[:4]
            for i in range(1, self.model.njoints)
        ])
        # 补方向始终依据 XML 初值，不随拟合结果或重复调用 fit 改变。
        self.reference_parameters = self.parameters.copy()

    def regressor(self, q):
        Y = pin.computeJointTorqueRegressor(
            self.model, self.data, np.asarray(q, dtype=float), self.zero, self.zero
        )
        return Y[:, self.columns]

    def gravity(self, q):
        """用当前辨识参数预测 joint1～joint6 的重力补偿力矩（N·m）。"""
        return self.regressor(q) @ self.parameters

    def fit(self, q, tau):
        """以原始模型重力的 sign 给反馈幅值补方向，再求最小范数修正。"""
        q, tau = np.asarray(q, dtype=float), np.asarray(tau, dtype=float)
        if q.ndim != 2 or q.shape[1] != 6 or tau.shape != q.shape:
            raise ValueError("q 和 tau 必须为相同形状的 N×6 数组")
        valid_q = np.all(np.isfinite(q), axis=1)
        q, tau = q[valid_q], tau[valid_q]
        finite_tau = np.isfinite(tau).ravel()
        if not np.any(finite_tau):
            raise ValueError("没有有效的角度和力矩数据")
        full_Y = np.vstack([self.regressor(angles) for angles in q])
        reference_gravity = full_Y @ self.reference_parameters
        valid_tau = finite_tau & (np.abs(reference_gravity) > SIGN_TORQUE_EPS)
        if not np.any(valid_tau):
            raise ValueError("模型重力均接近零，无法据此补全反馈方向")
        # 先取幅值，覆盖此前接口中人为加的逐轴固定符号，再逐姿态补方向。
        signed_tau = np.abs(tau.ravel()) * np.sign(reference_gravity)
        Y = full_Y[valid_tau]
        measured = signed_tau[valid_tau]
        before = self.parameters.copy()
        residual = measured - Y @ before
        # rcond 丢弃近零奇异值；不可辨识方向不更新，避免把原参数直接置零。
        delta, _, rank, _ = np.linalg.lstsq(Y, residual, rcond=1e-6)
        self.parameters += delta

        print("反馈补方向：abs(tau_feedback) * sign(g_XML(q))；符号固定使用原始 XML 参数。")
        print(f"跳过 |g_XML| ≤ {SIGN_TORQUE_EPS:g} N·m 的观测："
              f"{np.count_nonzero(finite_tau & ~valid_tau)} 条（不把未知方向的反馈置零）。")
        print(f"重力回归矩阵秩：{rank}/{Y.shape[1]}，有效力矩观测：{len(measured)}")
        print("参数                 更新前           更新后           变化量")
        for i, (old, new) in enumerate(zip(before, self.parameters)):
            joint = self.model.names[i // 4 + 1]
            parameter = ("m", "m*cx", "m*cy", "m*cz")[i % 4]
            print(f"{joint}.{parameter:<5} {old:14.6f} {new:14.6f} {new-old:+14.6f}")
        before_rmse = np.sqrt(np.mean(residual**2))
        after_rmse = np.sqrt(np.mean((measured - Y @ self.parameters)**2))
        print(f"补方向后力矩 RMSE（参与拟合的观测）：{before_rmse:.6f} → {after_rmse:.6f} N·m")
        magnitude = np.abs(tau.ravel()[finite_tau])
        magnitude_before = np.abs((full_Y @ before)[finite_tau])
        magnitude_after = np.abs((full_Y @ self.parameters)[finite_tau])
        print("幅值 RMSE（全部有效观测，含近零重力轴）："
              f"{np.sqrt(np.mean((magnitude - magnitude_before)**2)):.6f} → "
              f"{np.sqrt(np.mean((magnitude - magnitude_after)**2)):.6f} N·m")
        print("m 单位 kg，m*c 单位 kg·m；结果为等效重力参数，不代表唯一的物理质量/质心。")
        print("方向由原始模型推断，不是实测方向；模型方向错误时，补符号也会错误。")
        return self


def main():
    path = Path(__file__).parent / "data" / "gravity.npz"
    with np.load(path, allow_pickle=False) as saved:
        if saved["joint_names"].tolist() != [f"joint{i}" for i in range(1, 7)]:
            raise ValueError("数据列必须按 joint1～joint6 排列")
        GravityEstimator().fit(saved["q"], saved["tau"])


if __name__ == "__main__":
    main()
