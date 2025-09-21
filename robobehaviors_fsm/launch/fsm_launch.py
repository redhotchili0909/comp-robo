"""
Launch all behaviors and the FSM that selects which behavior controls /cmd_vel.
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description() -> LaunchDescription:

    return LaunchDescription(
        [
            # Teleop -> teleop/cmd_vel
            Node(
                package="robobehaviors_fsm",
                executable="teleop",
                name="teleop",
                output="screen",
                remappings=[("cmd_vel", "teleop/cmd_vel")],
            ),
            # Wall Follower -> wall_follower/cmd_vel
            Node(
                package="robobehaviors_fsm",
                executable="wall_follower",
                name="wall_follower",
                output="screen",
                remappings=[("cmd_vel", "wall_follower/cmd_vel")],
            ),
            # Drive Square -> drive_square/cmd_vel
            Node(
                package="robobehaviors_fsm",
                executable="drive_square",
                name="drive_square",
                output="screen",
                remappings=[("cmd_vel", "drive_square/cmd_vel")],
            ),
            # FSM / mux node
            Node(
                package="robobehaviors_fsm",
                executable="fsm",
                name="behavior_fsm",
                output="screen",
            ),
        ]
    )
