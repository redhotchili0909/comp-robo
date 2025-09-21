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
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from rcl_interfaces.msg import SetParametersResult


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

        # Parameters for control
        self.declare_parameter('mode', 'idle')  # idle|teleop|wall|square|person
        self.declare_parameter('estop', False)
        self.declare_parameter('debug', False)
        self._debug = bool(self.get_parameter('debug').value)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # Debug state
        self._last_command = {k: 0.0 for k in ('TELEOP', 'WALL', 'SQUARE', 'PERSON')}
        self._publish_state()
        self.get_logger().info('Behavior FSM started in IDLE. Set /behavior_mode to select a behavior.')

    @staticmethod
    def _normalize_mode(text: str) -> Optional[str]:
        if text is None:
            return None
        t = str(text).strip().lower()
        mapping = {
            'idle': 'IDLE', 'stop': 'IDLE', 'none': 'IDLE',
            'teleop': 'TELEOP', 'manual': 'TELEOP',
            'wall': 'WALL', 'wall_follow': 'WALL', 'wall_follower': 'WALL',
            'square': 'SQUARE', 'drive_square': 'SQUARE',
            'person': 'PERSON', 'follow': 'PERSON', 'person_follower': 'PERSON',
            'estop': 'ESTOP',
        }
        return mapping.get(t)

    # Input callbacks for behavior Twist topics
    def _forward(self, src: str, msg: Twist):
        if self.estop:
            return
        if self.mode != src:
            return
        self.cmd_pub.publish(msg)
        if self._debug:
            now = time.monotonic()
            if now - self._last_command.get(src, 0.0) > 1.0:
                self.get_logger().info(
                    f"Forwarding {src}: lin={msg.linear.x:.3f} ang={msg.angular.z:.3f}")
                self._last_command[src] = now

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

    def _on_set_parameters(self, params):
        """Handle parameter updates for 'mode' and 'estop'."""
        # Validate first
        desired_mode_param = None
        desired_estop_param = None
        for p in params:
            if p.name == 'mode':
                desired_mode_param = p
            elif p.name == 'estop':
                desired_estop_param = p
        # Validate mode value if provided
        if desired_mode_param is not None:
            m = self._normalize_mode(str(desired_mode_param.value))
            if m is None:
                return SetParametersResult(successful=False, reason="Invalid mode. Use: idle, teleop, wall, square, person")
        # Validate estop value if provided
        if desired_estop_param is not None and not isinstance(desired_estop_param.value, bool):
            return SetParametersResult(successful=False, reason="estop must be boolean")
        # Apply in safe order: estop first
        if desired_estop_param is not None:
            self._on_estop_cmd(Bool(data=bool(desired_estop_param.value)))
        if desired_mode_param is not None:
            m = self._normalize_mode(str(desired_mode_param.value))
            # Only apply mode if not in ESTOP
            if not self.estop:
                self._on_mode_cmd(String(data=m))
            else:
                self.desired_mode = m if m and m != 'IDLE' else None
        return SetParametersResult(successful=True)

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
