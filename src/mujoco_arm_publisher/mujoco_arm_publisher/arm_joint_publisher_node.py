import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

class SimpleMujocoArm(Node):
    def __init__(self):
        super().__init__('simple_mujoco_arm')
        
        # 1. 加载内置的 UR5e 模型 (MuJoCo 3.0+ 标准路径)
        # 如果报错找不到文件，请确保已安装最新版 mujoco: pip install -U mujoco
        xml = str(Path(__file__).resolve().parent.parent / 'models' / 'universal_robots_ur5e' / 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(xml) # type: ignore
        # self.model = mujoco.MjModel.from_xml_path('universal_robots_ur5e/ur5e.xml')
        self.data = mujoco.MjData(self.model) # type: ignore
        
        # 2. ROS2 发布者 (发布关节状态)
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        
        # 3. 获取关节名称 (用于 ROS2 消息)
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)  # type: ignore
                           for i in range(self.model.njnt) if self.model.jnt_type[i] == 3] # Hinge joints
        
        # 4. 启动可视化窗口 (被动模式，不会阻塞主线程)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # 5. 定时器：控制仿真步进和发布
        self.dt = self.model.opt.timestep
        self.timer = self.create_timer(self.dt, self._simulation_step)
        self.get_logger().info("MuJoCo UR5e 仿真已启动")

    def _simulation_step(self):
        # 简单的正弦波控制，让机械臂动起来
        current_time = self.data.time
        for i in range(min(len(self.data.ctrl), 6)):
            self.data.ctrl[i] = 0.5 * np.sin(current_time + i * 0.5)

        # 仿真步进
        mujoco.mj_step(self.model, self.data) # type: ignore
        
        # 同步可视化窗口
        if self.viewer.is_running():
            self.viewer.sync()
        else:
            rclpy.shutdown() # 窗口关闭则停止节点

        # 发布 ROS2 JointState
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.data.qpos[:len(self.joint_names)].tolist()
        msg.velocity = self.data.qvel[:len(self.joint_names)].tolist()
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleMujocoArm()
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