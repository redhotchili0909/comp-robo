#!/usr/bin/env python3
"""
Finite State Manager that multiplexes behavior velocity topics to /cmd_vel.

- Inputs:
  - /behavior_mode (std_msgs/String): one of [idle, teleop, wall, square, person]
  - /estop (std_msgs/Bool): true = emergency stop
  - Behavior Twist topics: /teleop/cmd_vel, /wall_follower/cmd_vel, /drive_square/cmd_vel, optionally /person_follower/cmd_vel
- Outputs:
  - /cmd_vel (geometry_msgs/Twist): selected behavior's commands (or zero when idle/estop)
  - /fsm/state (std_msgs/String): current FSM state

Behavior nodes can run continuously; only the selected topic is forwarded.
"""

from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool


class BehaviorFSM(Node):
    def __init__(self):
        super().__init__('behavior_fsm')

        # Output publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.state_pub = self.create_publisher(String, 'fsm/state', 10)

        # Internal state
        self.mode: str = 'IDLE'  # IDLE|TELEOP|WALL|SQUARE|PERSON|ESTOP
        self.estop: bool = False
        self.desired_mode: Optional[str] = None

        # Behavior input subscriptions
        self.create_subscription(Twist, 'teleop/cmd_vel', self._on_teleop_cmd, 10)
        self.create_subscription(Twist, 'wall_follower/cmd_vel', self._on_wall_cmd, 10)
        self.create_subscription(Twist, 'drive_square/cmd_vel', self._on_square_cmd, 10)
        self.create_subscription(Twist, 'person_follower/cmd_vel', self._on_person_cmd, 10)

        # Control inputs
        self.create_subscription(String, 'behavior_mode', self._on_mode_cmd, 10)
        self.create_subscription(Bool, 'estop', self._on_estop_cmd, 10)

        # Safety timer to continuously enforce zero while IDLE/ESTOP
        self.zero_timer = self.create_timer(0.2, self._publish_stop)

        self._publish_state()
        self.get_logger().info('Behavior FSM started in IDLE. Set /behavior_mode to select a behavior.')

    # Input callbacks for behavior Twist topics
    def _forward(self, src: str, msg: Twist):
        if self.estop:
            return
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

    # Region: Control inputs
    def _on_mode_cmd(self, msg: String):
        requested = self._normalize_mode(msg.data)
        if requested is None:
            self.get_logger().warn(f"Unknown mode '{msg.data}'. Valid: idle, teleop, wall, square, person")
            return

        if self.estop:
            # Remember the intent, but remain in ESTOP
            self.desired_mode = requested if requested != 'IDLE' else None
            self._set_mode('ESTOP')
            return

        if requested == 'IDLE':
            self._set_mode('IDLE')
            return

        if self.mode != requested:
            self._publish_stop()
        self._set_mode(requested)

    def _on_estop_cmd(self, msg: Bool):
        if msg.data and not self.estop:
            self.estop = True
            self._publish_stop()
            self._set_mode('ESTOP')
            return

        if not msg.data and self.estop:
            self.estop = False
            # Apply any desired mode or return to IDLE
            next_mode = self.desired_mode or 'IDLE'
            self.desired_mode = None
            self._set_mode(next_mode)

    # Region: Helpers
    @staticmethod
    def _normalize_mode(text: str) -> Optional[str]:
        if not text:
            return None
        t = text.strip().lower()
        aliases = {
            'idle': 'IDLE', 'stop': 'IDLE', 'none': 'IDLE',
            'teleop': 'TELEOP', 'manual': 'TELEOP',
            'wall': 'WALL', 'wall_follow': 'WALL', 'wall_follower': 'WALL',
            'square': 'SQUARE', 'drive_square': 'SQUARE',
            'person': 'PERSON', 'follow': 'PERSON', 'person_follower': 'PERSON',
            'estop': 'ESTOP',
        }
        return aliases.get(t)

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
        if self.estop or self.mode in ('IDLE', 'ESTOP'):
            zero = Twist()
            zero.linear.x = 0.0
            zero.angular.z = 0.0
            self.cmd_pub.publish(zero)


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
