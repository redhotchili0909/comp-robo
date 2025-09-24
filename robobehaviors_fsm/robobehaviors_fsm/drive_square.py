"""
This node drives the robot in a square, while implementing an emergency stop topic.
"""

from threading import Thread, Event
from time import sleep
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class DriveSquare(Node):
    """
    A class that implements a node to pilot a robot in a square.
    """

    def __init__(self):
        super().__init__("drive_square")
        self._enabled = Event()

        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(String, "fsm/state", self._on_fsm_state, 10)

        self.run_loop_thread = Thread(target=self.run_loop, daemon=True)
        self.run_loop_thread.start()

    def _on_fsm_state(self, msg: String):
        if msg.data == "SQUARE":
            if not self._enabled.is_set():
                self.get_logger().info("Activated by FSM (SQUARE)")
            self._enabled.set()
        else:
            if self._enabled.is_set():
                self.get_logger().info("Deactivated by FSM (leaving SQUARE)")
            self._enabled.clear()
            self.drive(0.0, 0.0)

    def run_loop(self):
        """
        Executes main driving logic (repeats continuously until shutdown)
        """
        self.drive(0.0, 0.0)
        side_m = 0.5
        dwell_s = 0.25
        while rclpy.ok():
            while rclpy.ok() and not self._enabled.is_set():
                self.drive(0.0, 0.0)
                sleep(0.05)
            if not rclpy.ok():
                break
            for _ in range(4):
                self.get_logger().debug("Driving forward one side")
                self.drive_forward(side_m)
                sleep(dwell_s)
                self.get_logger().debug("Turning left 90 degrees")
                self.turn_left()
                sleep(dwell_s)
        self.get_logger().info("Drive square loop exiting.")

    def drive(self, linear, angular):
        """
        Drives with specified linear and angular velocity

        Args:
            linear (_type_): linear velocity in m/s
            angular (_type_): angular velocity in radians/s
        """
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.vel_pub.publish(msg)

    def turn_left(self):
        """
        Executes 90 degree left turn (interruptible)
        """
        angular_vel = 0.5
        self.drive(linear=0.0, angular=angular_vel)
        sleep(math.pi / angular_vel / 2)
        self.drive(linear=0.0, angular=0.0)

    def drive_forward(self, distance):
        """
        Drive straight for the specified distance (interruptible)

        Args:
            distance (_type_): the distance to drive forward (only positive values)
        """
        forward_vel = 0.2
        if distance <= 0:
            return
        duration = distance / forward_vel
        self.drive(linear=forward_vel, angular=0.0)
        sleep(duration)
        self.drive(0.0, 0.0)


def main(args=None):
    """
    Overall main sequence
    """
    rclpy.init(args=args)
    node = DriveSquare()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
