#!/usr/bin/env python3
"""
Finite State Manager that multiplexes behavior velocity topics to /cmd_vel.

- Inputs:
  - /behavior_mode (std_msgs/String): one of [idle, teleop, wall, square, person]
  - Behavior Twist topics: /teleop/cmd_vel, /wall_follower/cmd_vel, /drive_square/cmd_vel, /person_follower/cmd_vel
- Outputs:
  - /cmd_vel (geometry_msgs/Twist): selected behavior's commands (or zero when idle/estop)
  - /fsm/state (std_msgs/String): current FSM state

Behavior nodes can run continuously; only the selected topic is forwarded.
"""

from typing import Optional
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rcl_interfaces.msg import SetParametersResult


class BehaviorFSM(Node):
    def __init__(self):
        super().__init__('behavior_fsm')

        # Output publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.state_pub = self.create_publisher(String, 'fsm/state', 10)

        # Internal state
        self.mode: str = 'IDLE'  # IDLE|TELEOP|WALL|SQUARE|PERSON|ESTOP

        # Behavior input subscriptions
        self.create_subscription(Twist, 'teleop/cmd_vel', self._on_teleop_cmd, 10)
        self.create_subscription(Twist, 'wall_follower/cmd_vel', self._on_wall_cmd, 10)
        self.create_subscription(Twist, 'drive_square/cmd_vel', self._on_square_cmd, 10)
        self.create_subscription(Twist, 'person_follower/cmd_vel', self._on_person_cmd, 10)

        # Control inputs
        self.create_subscription(String, 'behavior_mode', self._on_mode_cmd, 10)

        # Parameters for control
        self.declare_parameter('mode', 'idle')  # idle|teleop|wall|square|person|estop

        self.add_on_set_parameters_callback(self._on_set_parameters)

        # Debug state
        self._last_command = {k: 0.0 for k in ('TELEOP', 'WALL', 'SQUARE', 'PERSON')}
        self._publish_state()
        self.get_logger().info('Behavior FSM started in IDLE. Set /behavior_mode to select a behavior.')

    @staticmethod
    def _parse_mode(text: str) -> Optional[str]:
        if text is None:
            return None
        t = str(text).strip().lower()
        mapping = {
            'idle': 'IDLE',
            'teleop': 'TELEOP',
            'wall': 'WALL', 'wall_follower': 'WALL',
            'square': 'SQUARE', 'drive_square': 'SQUARE',
            'person': 'PERSON', 'person_follower': 'PERSON',
            'estop': 'ESTOP',
        }
        return mapping.get(t)

    # Input callbacks for behavior Twist topics
    def _forward(self, src: str, msg: Twist):
        if self.mode != src:
            return
        self.cmd_pub.publish(msg)

    def _on_teleop_cmd(self, msg: Twist):
        self._forward('TELEOP', msg)

    def _on_wall_cmd(self, msg: Twist):
        self._forward('WALL', msg)

    def _on_square_cmd(self, msg: Twist):
        self._forward('SQUARE', msg)

    def _on_person_cmd(self, msg: Twist):
        self._forward('PERSON', msg)

    # Control inputs
    def _on_mode_cmd(self, msg: String):
        requested = self._parse_mode(msg.data)
        if requested is None:
            self.get_logger().warn(f"Unknown mode '{msg.data}'. Valid: idle, teleop, wall, square, person,estop")
            return

        if requested == 'ESTOP':
            self._set_mode('ESTOP')
            return

        if requested == 'IDLE':
            self._set_mode('IDLE')
            return

        if self.mode != requested:
            self._publish_stop()

        self._set_mode(requested)
    
    def _set_mode(self, mode: str):

        if mode == self.mode:
            self._publish_state()
            return
        
        self.mode = mode

        self._publish_state()
        
        if mode in ('IDLE', 'ESTOP'):
            self._publish_stop()
        else:
            self.get_logger().info(f"Active behavior: {mode}")

    def _publish_state(self):
        out = String()
        out.data = self.mode
        self.state_pub.publish(out)

    def _publish_stop(self):
        zero = Twist()
        zero.linear.x = 0.0
        zero.angular.z = 0.0
        self.cmd_pub.publish(zero)

    def _on_set_parameters(self, params):
        """Handle parameter updates for 'mode'"""
        desired_mode_param = None
        
        for p in params:
            if p.name == 'mode':
                desired_mode_param = p
        
        # Validate mode value if provided
        if desired_mode_param is not None:
            mode = self._parse_mode(str(desired_mode_param.value))
            
            if mode is None:
                return SetParametersResult(successful=False, reason="Invalid mode. Use: idle, teleop, wall, square, person, estop")

            self._on_mode_cmd(String(data=mode))

        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorFSM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
