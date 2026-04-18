from pathlib import Path

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

        self.get_logger().info("IK 节点初始化成功 (Python 拆分版)")

    def joint_state_callback(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return

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