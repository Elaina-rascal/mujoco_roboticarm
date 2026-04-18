from pathlib import Path
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from pinocchio_ik import PinocchioIKSolver


class SimpleIKNode(Node):
    """ROS2 节点层: 只负责通信和调用 IK 求解器。"""

    def __init__(self) -> None:
        super().__init__("simple_ik")

        # 模型路径约定为当前目录下的 model/ur5e.xml
        current_dir = Path(__file__).resolve().parent
        model_path = current_dir / "model" / "ur5e.xml"

        try:
            self.solver = PinocchioIKSolver(model_path=model_path)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"无法初始化 Pinocchio 求解器: {exc}")
            raise

        self.sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            1,
        )
        self.pub = self.create_publisher(JointState, "/ik_joint_target", 10)

        # 圆轨迹目标参数：末端目标点将围绕 center 在 x-y 平面画圆。
        self.circle_center = np.array([0.35, 0.15, 0.5], dtype=float)
        self.circle_radius = 0.0
        self.circle_omega = 6  # rad/s
        self.start_time_ns = self.get_clock().now().nanoseconds

        self.get_logger().info("IK 节点初始化成功 (Python 拆分版)")

    def _update_circular_target(self) -> None:
        """根据当前时间计算圆轨迹目标并写入 IK 求解器。"""
        now_ns = self.get_clock().now().nanoseconds
        t = (now_ns - self.start_time_ns) * 1e-9

        target_x = self.circle_center[0] + self.circle_radius * math.cos(self.circle_omega * t)
        target_y = self.circle_center[1] + self.circle_radius * math.sin(self.circle_omega * t)
        target_z = self.circle_center[2]

        self.solver.set_target_point(float(target_x), float(target_y), float(target_z))

    def joint_state_callback(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return

        # 在每次求解前更新圆轨迹目标点，实现“不断画圈”的末端目标。
        self._update_circular_target()

        # 调用 Pinocchio 层完成 IK 求解
        q = self.solver.solve(list(msg.name), list(msg.position))

        out_msg = JointState()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.name = list(msg.name)
        out_msg.position = self.solver.q_to_joint_positions(
            q=q,
            joint_names=list(msg.name),
            fallback_positions=list(msg.position),
        )

        self.pub.publish(out_msg)


def main(args) -> None:
    rclpy.init(args=args)
    node = SimpleIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == "__main__":
    main(args=None) 