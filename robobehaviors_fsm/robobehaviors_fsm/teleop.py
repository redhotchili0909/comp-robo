#!/usr/bin/env python
"""Teleoperation for Neato using number pad keys."""

import tty
import select
import sys
import termios
from threading import Event
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class Teleop(Node):
    """Node that reads keyboard input and publishes velocity commands."""

    def __init__(self):
        super().__init__('teleop')
        self._enabled = Event()
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self._exit_requested = False

        # Track terminal settings only if input is a TTY
        self.settings = None
        if sys.stdin.isatty():
            try:
                self.settings = termios.tcgetattr(sys.stdin)
            except termios.error as e:
                self.get_logger().warn(f"Could not read TTY settings: {e}.")

        # FSM state subscription
        self.create_subscription(String, 'fsm/state', self._on_fsm_state, 10)

    # FSM callbacks
    def _on_fsm_state(self, msg: String):
        if msg.data == 'TELEOP':
            if not self._enabled.is_set():
                self.get_logger().info("Activated by FSM (TELEOP)")
                self.get_logger().info("Use number pad keys (Num Lock off).")
            self._enabled.set()
        else:
            if self._enabled.is_set():
                self.get_logger().info("Deactivated by FSM (leaving TELEOP)")
            self._enabled.clear()
            self._stop_robot()

    def _process_key(self, key: str):
        """Process a key press and publish the corresponding Twist message."""
     
        twist = Twist()

        if key == '8':
            twist.linear.x = 0.3
            twist.angular.z = 0.0
        elif key == '2':
            twist.linear.x = -0.3
            twist.angular.z = 0.0
        elif key == '4':
            twist.linear.x = 0.0
            twist.angular.z = 1.0
        elif key == '6':
            twist.linear.x = 0.0
            twist.angular.z = -1.0
        elif key == '5':
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif key == '7':
            twist.linear.x = 0.3
            twist.angular.z = 1.0
        elif key == '9':
            twist.linear.x = 0.3
            twist.angular.z = -1.0
        elif key == '1':
            twist.linear.x = -0.3
            twist.angular.z = 1.0
        elif key == '3':
            twist.linear.x = -0.3
            twist.angular.z = -1.0
        elif key == '\x03':
            self.get_logger().info("Exiting teleop node.")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            if rclpy.ok():
                self.publisher_.publish(twist)
            self._exit_requested = True
            return
        else:
            self.get_logger().info("Invalid key pressed. Use number pad keys (Num Lock off)")
            return

        self.publisher_.publish(twist)
        self.get_logger().debug(f"Published command: linear.x={twist.linear.x}, angular.z={twist.angular.z}")

    def _stop_robot(self):
        if rclpy.ok():
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher_.publish(twist)

    def run_keyboard_mode(self):
        """Run the node in keyboard mode, processing ROS callbacks in between key reads."""
        if not sys.stdin.isatty() or self.settings is None:
            self.get_logger().warn("No TTY detected. Keyboard input disabled; spinning callbacks only.")
            rclpy.spin(self)
            return
        try:
            tty.setraw(sys.stdin.fileno())
            while rclpy.ok() and not self._exit_requested:
                rclpy.spin_once(self, timeout_sec=0.0)
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    key = sys.stdin.read(1)
                    self._process_key(key)
        except KeyboardInterrupt:
            self._exit_requested = True
            self._stop_robot()

def main(args=None):
    """Initialize our node, run it, and clean up on shut down."""
    rclpy.init(args=args)

    node = Teleop()

    try:
        node.run_keyboard_mode()
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted.")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
