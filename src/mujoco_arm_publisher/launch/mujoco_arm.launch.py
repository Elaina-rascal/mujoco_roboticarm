from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package='mujoco_arm_publisher',
            executable='arm_joint_publisher',
            name='mujoco_arm_joint_publisher',
            output='screen',
        ),
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
        )
    ])
