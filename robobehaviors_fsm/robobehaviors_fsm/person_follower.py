"""
This node has the robot follow a person
"""

import rclpy
from rclpy.node import Node
from threading import Thread
from time import sleep
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math

# """
# Pseudocode:

# Loop:
# Detect sweep of laser scans within range of the front
# Find the shortest distance
# Turn to face that point
# Drive a little forward

# """

ANGULAR_VEL = 0.3
FORWARD_VEL = 0.1


class PersonFollower(Node):
    """
    A class that implements a node to have a robot follow a person
    """

    def __init__(self):
        super().__init__("person_follower")

        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        # create laser scan subscription self.create_subscription()

        # thread for long-running following component
        self.run_loop_thread = Thread(target=self.run_loop)
        self.run_loop_thread.start()

    def run_loop(self):
        """
        Executes main person following logic
        """
        # Detect sweep of laser scans within range of the front
        # Find the shortest distance
        # Turn to face that point
        # Drive a little forward
        while 1:  # infinite loop for now
            self.find_person()
            self.turn_follow()
            self.drive_forward()

    def find_person(self):
        """
        Checks the range of laser scans to find the shortest distance
        """

    def turn_follow(self):
        """
        Turns to the shortest distance point
        """

    def drive_forward(self):
        """
        Drive straight to determined person point
        """

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
