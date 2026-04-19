import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path


class TorqueMujocoArm(Node):
    def __init__(self):
        super().__init__('torque_mujoco_arm')

        xml = str(
            Path(__file__).resolve().parent.parent
            / 'models'
            / 'universal_robots_ur5e'
            / 'scene_torque.xml'
        )
        self.model = mujoco.MjModel.from_xml_path(xml)  # type: ignore
        self.data = mujoco.MjData(self.model)  # type: ignore

        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.target_subscriber = self.create_subscription(
            JointState,
            '/ik_joint_target',
            self._on_target_joint_state,
            10,
        )

        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)  # type: ignore
            for i in range(self.model.njnt)
            if self.model.jnt_type[i] == 3
        ]

        self.joint_name_to_act = {}
        for joint_name in self.joint_names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)  # type: ignore
            if act_id >= 0:
                self.joint_name_to_act[joint_name] = int(act_id)

        self.target_torques = np.zeros(len(self.data.ctrl), dtype=float)
        self.has_target = False

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.dt = self.model.opt.timestep
        self.timer = self.create_timer(self.dt, self._simulation_step)
        self.get_logger().info('MuJoCo UR5e 力矩控制仿真已启动')

    def _on_target_joint_state(self, msg: JointState):
        if not msg.name or len(msg.effort) < len(msg.name):
            return

        updated = False
        for index, joint_name in enumerate(msg.name):
            if joint_name not in self.joint_name_to_act:
                continue

            act_id = self.joint_name_to_act[joint_name]
            if act_id >= len(self.target_torques):
                continue

            self.target_torques[act_id] = float(msg.effort[index])
            updated = True

        if updated:
            self.has_target = True

    def _simulation_step(self):
        if self.has_target:
            dof_count = min(len(self.data.ctrl), len(self.joint_names))
            for i in range(dof_count):
                tau = float(self.target_torques[i])
                self.data.ctrl[i] = tau

        mujoco.mj_step(self.model, self.data)  # type: ignore

        if self.viewer.is_running():
            self.viewer.sync()
        else:
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TorqueMujocoArm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.viewer.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
