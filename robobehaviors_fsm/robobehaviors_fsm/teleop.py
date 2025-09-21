#!/usr/bin/env python
""" Teleoperation for Neato using number pad keys"""

import tty
import select
import sys
import termios
from threading import Event
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool


class Teleop(Node):
    """ Node to read keyboard input and publish velocity commands """

    def __init__(self, use_subscription=True):
        super().__init__('teleop')
        self.use_subscription = use_subscription
        # FSM gating
        self._enabled = Event()
        self._estop = Event()

        # Keyboard terminal settings
        self.settings = None
        if not self.use_subscription:
            if sys.stdin.isatty():
                try:
                    self.settings = termios.tcgetattr(sys.stdin)
                except OSError as e:
                    self.get_logger().warn(f"Failed to init terminal (keyboard mode): {e}. Falling back to subscription mode.")
                    self.use_subscription = True
            else:
                self.get_logger().warn("Keyboard mode requested but stdin is not a TTY; falling back to subscription mode.")
                self.use_subscription = True

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self._exit_requested = False

        self.create_subscription(String, 'fsm/state', self._on_fsm_state, 10)
        self.create_subscription(Bool, 'estop', self._on_estop, 10)

        if self.use_subscription:
            self.create_subscription(String, 'teleop_command', self.command_callback, 10)
            self.get_logger().info("Teleop node started in subscription mode. Use the 'teleop_command' topic to move the robot.")
        else:
            self.get_logger().info("Teleop node started in keyboard mode. Use number pad keys (Num Lock off).")
        if not self._enabled.is_set():
            self.get_logger().info("Teleop is disabled until FSM mode is TELEOP and ESTOP is false. Set: ros2 param set /behavior_fsm mode teleop")

    # FSM callbacks
    def _on_fsm_state(self, msg: String):
        if msg.data == 'TELEOP':
            if not self._enabled.is_set():
                self.get_logger().info("Activated by FSM (TELEOP)")
            self._enabled.set()
        else:
            if self._enabled.is_set():
                self.get_logger().info("Deactivated by FSM (leaving TELEOP)")
            self._enabled.clear()
            self.stop_robot()

    def _on_estop(self, msg: Bool):
        if bool(msg.data):
            self._estop.set()
            self.stop_robot()
        else:
            self._estop.clear()

    def command_callback(self, msg: String):
        self.process_key(msg.data)

    def process_key(self, key: str):
        """ Process a key press and publish the corresponding Twist message """
        if (not self._enabled.is_set() or self._estop.is_set()) and key != '\x03':
            self.get_logger().info("Ignoring input: not in TELEOP or ESTOP active.")
            return
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
            self.get_logger().info("Invalid key pressed.")
            return

        self.publisher_.publish(twist)
        self.get_logger().debug(f"Published command: linear.x={twist.linear.x}, angular.z={twist.angular.z}")

    def stop_robot(self):
        if rclpy.ok():
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher_.publish(twist)

    def restore_terminal_settings(self):
        if self.settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            except OSError:
                pass

    def run_keyboard_mode(self):
        """ Run the node in keyboard mode, processing ROS callbacks in between key reads. """
        if self.settings is None:
            self.get_logger().error("Cannot run keyboard mode: stdin is not a TTY.")
            return
        try:
            tty.setraw(sys.stdin.fileno())
            while rclpy.ok() and not self._exit_requested:
                rclpy.spin_once(self, timeout_sec=0.0)
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    key = sys.stdin.read(1)
                    self.process_key(key)
        except KeyboardInterrupt:
            self._exit_requested = True
        finally:
            self.stop_robot()
            self.restore_terminal_settings()

def main(args=None):
    """Initialize our node, run it, cleanup on shut down"""
    rclpy.init(args=args)

    if '--subscription' in sys.argv:
        use_subscription = True
    elif '--keyboard' in sys.argv:
        use_subscription = False
    else:
        use_subscription = not sys.stdin.isatty()

    node = Teleop(use_subscription=use_subscription)
    node.stop_robot()

    try:
        if use_subscription:
            rclpy.spin(node)
        else:
            node.run_keyboard_mode()
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted.")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

