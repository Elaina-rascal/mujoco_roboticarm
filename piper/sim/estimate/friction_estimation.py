"""运行 python3 -m piper.sim.estimate.friction_estimation --joint 3。"""

import argparse

import mujoco
import numpy as np

from piper.controllers.pinocchio_controller import PinocchioControl
from piper.estimate.friction_estimator import FrictionEstimator
from piper.interface.mujoco import MujocoInterface
from piper.models import MUJOCO_MODEL, OFFICIAL_MODEL


class FrictionSimulation:
    """为估计类提供位置/力矩切换与仿真时钟，不使用力矩反馈。"""

    def __init__(self):
        self.sim = MujocoInterface("position", model_path=MUJOCO_MODEL)
        model = self.sim.model
        model.opt.timestep = 0.001
        self.sim.data.qpos[self.sim.q_ids] = [0.0, 1.0, -1.1, 0.0, 0.4, 0.0]
        self.sim.set_gripper(0.04)
        self.gravity_compensation = model.body_gravcomp.copy()
        self.torque_limits = model.actuator_forcerange[self.sim.actuator_ids, 1].copy()
        self.mode = None
        self.peak_speed = 0.0
        self.minimum = self.read_joint_positions()
        self.maximum = self.minimum.copy()
        self.set_mode("position")
        self.sim.send_joint_positions(self.read_joint_positions())

    def set_mode(self, mode):
        if mode == self.mode:
            return
        model = self.sim.model
        ids = self.sim.actuator_ids
        model.actuator_gainprm[ids] = 0
        model.actuator_biasprm[ids] = 0
        if mode == "position":
            # 仅回位阶段使用较强的位置伺服，克服摩擦回到同一起点。
            model.body_gravcomp[:] = self.gravity_compensation
            model.actuator_gainprm[ids, 0] = 600
            model.actuator_biastype[ids] = mujoco.mjtBias.mjBIAS_AFFINE
            model.actuator_biasprm[ids, 1] = -600
            model.actuator_biasprm[ids, 2] = -20
        else:
            model.body_gravcomp[:] = 0
            model.actuator_gainprm[ids, 0] = 1
            model.actuator_biastype[ids] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_ctrllimited[ids] = mode == "position"
        self.mode = mode
        self.sim.mode = mode

    def read_joint_positions(self):
        return self.sim.read_joint_positions()

    def send_joint_positions(self, q, speed_percent=5):
        self.set_mode("position")
        self.sim.send_joint_positions(q)

    def send_joint_torques(self, torque, positions=None, kp=0.0, kd=0.0):
        self.set_mode("torque")
        self.sim.send_joint_torques(torque)

    def clock(self):
        return float(self.sim.data.time)

    def sleep(self, duration):
        steps = round(duration / self.sim.model.opt.timestep)
        for _ in range(steps):
            self.sim.step()
            q = self.read_joint_positions()
            self.minimum = np.minimum(self.minimum, q)
            self.maximum = np.maximum(self.maximum, q)
            speed = np.max(np.abs(self.sim.data.qvel[self.sim.v_ids]))
            self.peak_speed = max(self.peak_speed, float(speed))


def main():
    parser = argparse.ArgumentParser(description="MuJoCo 双向启动摩擦辨识，不连接真机")
    parser.add_argument("--joint", type=int, choices=range(1, 7), default=3)
    args = parser.parse_args()
    arm = FrictionSimulation()
    control = PinocchioControl(model_path=OFFICIAL_MODEL, gripper_width=0.04)
    estimator = FrictionEstimator(arm, control)
    estimator.clock = arm.clock
    estimator.sleep = arm.sleep
    reference = arm.read_joint_positions()
    result = estimator.measure(args.joint)
    index = args.joint - 1
    truth = arm.sim.model.dof_frictionloss[arm.sim.v_ids[index]]
    print(f"J{args.joint}：正向阈值 {result['positive']:.4f} N·m；反向阈值 {result['negative']:.4f} N·m")
    print(f"估计摩擦 {result['friction']:.4f} N·m；模型设定 {truth:.4f} N·m")
    print(f"残余偏置 {result['bias']:.4f} N·m；全过程最大关节速度 {np.rad2deg(arm.peak_speed):.4f}°/s")
    print(f"正向行程 {np.rad2deg(arm.maximum[index] - reference[index]):.4f}°；"
          f"反向行程 {np.rad2deg(reference[index] - arm.minimum[index]):.4f}°")
    print(f"累计仿真时间 {arm.clock():.3f} s")


if __name__ == "__main__":
    main()
