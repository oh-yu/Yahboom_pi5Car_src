"""Gazebo と RL 訓練ノード（ランダムエージェント）を同時起動する launch.

学習は長時間走るため通常は `ros2 run yahboomcar_rl train` を直接叩く方が便利。
ここでは動作確認用に random_agent を紐付けている。
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    desc_share = get_package_share_directory("yahboomcar_description")
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(desc_share, "launch", "display_gazebo.launch.py")
        )
    )

    # Gazebo 起動後に少し待って RL ノードを立ち上げる
    rl_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package="yahboomcar_rl",
                executable="random_agent",
                name="random_agent",
                output="screen",
                arguments=["--episodes", "3"],
            )
        ],
    )

    return LaunchDescription([gazebo_launch, rl_node])
