"""
Launch all behaviors and the FSM that selects which behavior controls /cmd_vel.

Usage:
  ros2 launch robobehaviors_fsm behaviors_with_fsm.launch.py
Then set mode via:
  ros2 topic pub /behavior_mode std_msgs/String "data: 'teleop'"
  ros2 topic pub /behavior_mode std_msgs/String "data: 'wall'"
  ros2 topic pub /behavior_mode std_msgs/String "data: 'square'"
  ros2 topic pub /behavior_mode std_msgs/String "data: 'stop'"
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            # Teleop
            Node(
                package="robobehaviors_fsm",
                executable="teleop",
                name="teleop",
                remappings=[("cmd_vel", "teleop/cmd_vel")],
            ),
            # Wall follower
            Node(
                package="robobehaviors_fsm",
                executable="wall_follower",
                name="wall_follower",
                remappings=[("cmd_vel", "wall_follower/cmd_vel")],
            ),
            # Drive square
            Node(
                package="robobehaviors_fsm",
                executable="drive_square",
                name="drive_square",
                remappings=[("cmd_vel", "drive_square/cmd_vel")],
            ),
            # FSM
            Node(
                package="robobehaviors_fsm",
                executable="fsm",
                name="behavior_fsm",
            ),
        ]
    )
