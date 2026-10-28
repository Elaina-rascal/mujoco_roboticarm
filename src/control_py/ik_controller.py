import math
from pathlib import Path

from .pinocchio_ik import PinocchioIKSolver


class IKController:
    """直接接收关节数组，返回圆轨迹目标对应的关节角。"""

    def __init__(self, center=(0.35, 0.15, 0.5), radius=0.0, omega=6.0):
        model_path = Path(__file__).resolve().parents[2] / "models" / "universal_robots_ur5e" / "ur5e.xml"
        self.solver = PinocchioIKSolver(model_path)
        self.center = center
        self.radius = radius
        self.omega = omega

    def update(self, joint_names, positions, simulation_time):
        x, y, z = self.center
        phase = self.omega * simulation_time
        self.solver.set_target_point(
            x + self.radius * math.cos(phase),
            y + self.radius * math.sin(phase),
            z,
        )
        q = self.solver.solve(joint_names, positions)
        return self.solver.q_to_joint_positions(q, joint_names, positions)
