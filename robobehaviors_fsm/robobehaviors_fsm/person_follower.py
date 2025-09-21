"""
This node has the robot follow a person
"""

import rclpy
from rclpy.node import Node
from threading import Thread, Lock
from time import sleep
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math

ANGULAR_VEL = 0.3
FORWARD_VEL = 0.1


class PersonFollower(Node):
    """
    A class that implements a node to have a robot follow a person
    """

    def __init__(self):
        super().__init__("person_follower")

        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(LaserScan, "scan", self.find_person, 10)

        # thread for long-running following component
        self.run_loop_thread = Thread(target=self.run_loop)
        self.run_loop_thread.start()

        # parameters
        self.vel_lock = Lock()
        self.angular_vel = 0.0
        self.linear_vel = 0.0
        self.turn_angle = 0.0

        self.local_angular_vel = 0.0
        self.local_linear_vel = 0.0
        self.local_turn_angle = 0.0

    def run_loop(self):
        """
        Executes main person following logic
        """
        # Detect sweep of laser scans within range of the front
        # Find the shortest distance
        # Turn to face that point
        # Drive a little forward
        while 1:  # infinite loop for now

            # prevent race conditions
            with self.vel_lock:
                self.local_angular_vel = self.angular_vel
                self.local_linear_vel = self.linear_vel
                self.local_turn_angle = self.turn_angle

            self.turn_follow()
            self.drive_forward()

    def find_person(self, msg: LaserScan):
        """
        Checks the range of laser scans to find the shortest distance

        Args:
            msg (sensor_msgs.msg.LaserScan)

        Updates angle and distance of determined person point
        """
        self.handle_scan(msg)

        with self.vel_lock:
            self.turn_angle = None
            self.angular_vel = None  # TO-DO update in a bit
            self.linear_vel = None

    def handle_scan(self, scan: LaserScan):
        """
        Handles and processes messages received on scan topic

        Args:
            msg (sensor_msgs.msg.LaserScan)

        Returns copy of ranges narrowed to search area
        """
        # get scan ranges (90 to -90)
        rad_90 = 1.57
        range_idx_left = int(
            round((rad_90 - scan.angle_min) / scan.angle_increment)
        )  # get closest to it
        scan.ranges(range_idx_left)  # need to check left of this index
        rad_neg_90 = 1.57
        range_idx_right = int(
            round((scan.angle_max - rad_neg_90) / scan.angle_increment)
        )  # get closest to it
        scan.ranges(range_idx_right)  # need to check right of this index

    def turn_follow(self):
        """
        Turns to the shortest distance point
        """
        angular_vel = self.local_angular_vel
        self.drive(linear=0.0, angular=angular_vel)
        # sleep() for calculated time it takes to turn to that angle

    def drive_forward(self):
        """
        Drive straight to determined person point
        """
        linear_vel = self.local_linear_vel
        self.drive(linear=linear_vel, angular=0.0)
        # sleep() for calculated time it takes to reach like half of distance

    def drive(self, linear, angular):
        """
        Drives with specified linear and angular velocity

        Args:
            linear (_type_): linear velocity in m/s
            angular (_type_): angular velocity in radians/s
        """
        # publish to cmd_vel node with specified linear and angular velocity
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.vel_pub.publish(msg)


def main(args=None):
    """
    Overall main sequences
    """
    rclpy.init(args=args)
    node = PersonFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main = ()
