#!/usr/bin/env python
""" Teleoperation for Neato using number pad keys"""

import tty
import select
import sys
import termios
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class Teleop(Node):
    """ Node to read keyboard input and publish velocity commands """

    def __init__(self, use_subscription=True):
        super().__init__('teleop')
        self.use_subscription = use_subscription
        self.settings = termios.tcgetattr(sys.stdin)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self._exit_requested = False

        if self.use_subscription:
            self.create_subscription(String, 'teleop_command', self.command_callback, 10)
            self.get_logger().info("Teleop node started in subscription mode. Use the 'teleop_command' topic to move the robot.")
        else:
            self.get_logger().info("Teleop node started in keyboard mode. Use number pad keys to move the robot.")

    def command_callback(self, msg):
        """ Callback to process teleop commands from the subscription """
        self.process_key(msg.data)

    def process_key(self, key):
        """ Process a key press and publish the corresponding Twist message """
        twist = Twist()

        if key == '8':  # Forward
            twist.linear.x = 0.3
            twist.angular.z = 0.0
        elif key == '2':  # Backward
            twist.linear.x = -0.3
            twist.angular.z = 0.0
        elif key == '4':  # Turn left
            twist.linear.x = 0.0
            twist.angular.z = 1.0
        elif key == '6':  # Turn right
            twist.linear.x = 0.0
            twist.angular.z = -1.0
        elif key == '5':  # Stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif key == '7':  # Forward-left
            twist.linear.x = 0.3
            twist.angular.z = 1.0
        elif key == '9':  # Forward-right
            twist.linear.x = 0.3
            twist.angular.z = -1.0
        elif key == '1':  # Backward-left
            twist.linear.x = -0.3
            twist.angular.z = 1.0
        elif key == '3':  # Backward-right
            twist.linear.x = -0.3
            twist.angular.z = -1.0
        elif key == '\x03':
            self.get_logger().info("Exiting teleop node.")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            try:
                if rclpy.ok():
                    self.publisher_.publish(twist)
            except Exception:
                pass
            self._exit_requested = True
            return
        else:
            self.get_logger().info("Invalid key pressed.")
            return

        self.publisher_.publish(twist)
        self.get_logger().info(f"Published command: linear.x={twist.linear.x}, angular.z={twist.angular.z}")

    def stop_robot(self):
        """ Publish a stop command to ensure the robot stops moving, only if ROS is still OK. """
        try:
            if rclpy.ok():
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.publisher_.publish(twist)
                self.get_logger().info("Published stop command to ensure the robot is stationary.")
        except Exception:
            pass

    def restore_terminal_settings(self):
        """ Restore terminal settings """
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def get_key(self):
        """ Get a single key press from the terminal """
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run_keyboard_mode(self):
        """ Run the node in keyboard mode """
        try:
            while rclpy.ok() and not self._exit_requested:
                key = self.get_key()
                self.process_key(key)
        except KeyboardInterrupt:
            self.get_logger().info("Keyboard mode interrupted.")
            self._exit_requested = True
        finally:
            self.stop_robot()
            self.restore_terminal_settings()

def main(args=None):
    """Initialize our node, run it, cleanup on shut down"""
    rclpy.init(args=args)

    use_subscription = '--keyboard' not in sys.argv
    node = Teleop(use_subscription=use_subscription)

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

