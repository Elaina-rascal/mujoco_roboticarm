"""随机修改 link2 的质量，用静态重力力矩离线辨识，再对比结果。"""

from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin


MODEL_PATH = Path(__file__).resolve().parents[1] / "arm_official.xml"


def set_mass(model, joint_id, mass):
    """修改质量，质心不变，转动惯量按质量同比缩放。"""
    inertia = model.inertias[joint_id]
    model.inertias[joint_id] = pin.Inertia(
        mass, inertia.lever.copy(), inertia.inertia * mass / inertia.mass
    )


def collect_data(model, rng):
    """MuJoCo 采集 300 组静态姿态和六轴重力力矩，夹爪保持半开。"""
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    mj_data = mujoco.MjData(mj_model)
    mj_model.body_gravcomp[:] = 0
    positions = np.tile(pin.neutral(model), (300, 1))
    torques = np.zeros((300, 6))

    # 两个引擎按关节名对应，避免依赖各自的关节排列顺序。
    for i in range(1, 9):
        joint = model.joints[model.getJointId(f"joint{i}")]
        if i <= 6:
            lower = model.lowerPositionLimit[joint.idx_q]
            upper = model.upperPositionLimit[joint.idx_q]
            positions[:, joint.idx_q] = rng.uniform(lower, upper, 300)
        else:
            positions[:, joint.idx_q] = 0.02 if i == 7 else -0.02

    for sample, q in enumerate(positions):
        for i in range(1, 9):
            joint = model.joints[model.getJointId(f"joint{i}")]
            mj_joint = mj_model.joint(f"joint{i}")
            mj_data.qpos[mj_joint.qposadr[0]] = q[joint.idx_q]
        # qvel 默认为零，因此 qfrc_bias 只包含重力项。
        mujoco.mj_forward(mj_model, mj_data)
        for i in range(1, 7):
            dof = mj_model.joint(f"joint{i}").dofadr[0]
            torques[sample, i - 1] = mj_data.qfrc_bias[dof]
    return positions, torques


def gravity(model, positions):
    """Pinocchio 计算每个姿态下 joint1～joint6 的重力力矩。"""
    data = model.createData()
    dofs = [model.joints[model.getJointId(f"joint{i}")].idx_v for i in range(1, 7)]
    torques = []
    for q in positions:
        tau = pin.computeGeneralizedGravity(model, data, q)
        torques.append(tau[dofs].copy())
    return np.array(torques)


def identify_mass(model, joint_id, positions, measured_torques):
    """其余参数已知，最小二乘求这根连杆的质量修正量。"""
    mass = model.inertias[joint_id].mass
    predicted = gravity(model, positions)

    # 质心固定时，重力力矩对质量是线性的。
    # 增加 1 kg 得到每 kg 对力矩的贡献 A：g(m + dm) = g(m) + A * dm。
    heavier = model.copy()
    set_mass(heavier, joint_id, mass + 1.0)
    contribution = gravity(heavier, positions) - predicted

    A = contribution.reshape(-1, 1)
    b = (measured_torques - predicted).reshape(-1)
    delta_mass, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank != 1:
        raise ValueError("这些姿态无法辨识该连杆的质量")
    estimated_mass = mass + float(delta_mass[0])
    if estimated_mass <= 0:
        raise ValueError("辨识得到非正质量，请检查采样数据")
    return estimated_mass


def main():
    rng = np.random.default_rng(42)
    original = pin.buildModelFromMJCF(str(MODEL_PATH))
    joint_id = original.getJointId("joint2")
    original_mass = original.inertias[joint_id].mass

    # 1. 随机修改 link2 的质量，原始模型留作真值。
    initial = original.copy()
    initial_mass = original_mass * rng.uniform(0.5, 1.5)
    set_mass(initial, joint_id, initial_mass)

    # 2. 前 200 个姿态用于辨识，后 100 个只用于验证。
    positions, measured_torques = collect_data(original, rng)
    estimated_mass = identify_mass(initial, joint_id, positions[:200], measured_torques[:200])
    estimated = initial.copy()
    set_mass(estimated, joint_id, estimated_mass)

    # 3. 比较质量，以及未参与辨识的姿态上的重力力矩误差。
    print("link2 质量对比 (kg)")
    print(f"原始：{original_mass:.6f}")
    print(f"扰动：{initial_mass:.6f}")
    print(f"辨识：{estimated_mass:.6f}")
    print("\n独立测试集重力力矩 RMSE (N·m)")
    for name, model in (("原始", original), ("扰动", initial), ("辨识", estimated)):
        error = gravity(model, positions[200:]) - measured_torques[200:]
        print(f"{name}：{np.sqrt(np.mean(error ** 2)):.6e}")


if __name__ == "__main__":
    main()
