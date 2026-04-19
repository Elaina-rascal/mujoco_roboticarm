from glob import glob
from setuptools import setup

package_name = 'mujoco_arm_publisher'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', glob('models/*.xml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'mujoco>=3.1.0'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='MuJoCo two-link arm demo that publishes joint angles over ROS 2.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_joint_publisher = mujoco_arm_publisher.arm_joint_publisher_node:main',
            'arm_joint_torque_publisher = mujoco_arm_publisher.torque_node:main',
        ],
    },
)
