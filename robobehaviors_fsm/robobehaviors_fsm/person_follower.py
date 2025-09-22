"""
This node has the robot follow a person
"""

from time import sleep
from threading import Thread, Lock
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

# start sim first
# ros2 launch robobehaviors_fsm fsm_launch.py
# ros2 param set /behavior_fsm mode [mode selection]


class PersonFollower(Node):
    """
    A class that implements a node to have a robot follow a person
    """

    def __init__(self):
        super().__init__("person_follower")

        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(LaserScan, "scan", self.find_person, 10)

        # parameters
        self.vel_lock = Lock()
        self.angular_vel = 0.0
        self.distance = 0.0
        self.turn_angle = 0.0

        self.local_angular_vel = 0.0
        self.local_distance = 0.0
        self.local_turn_angle = 0.0

        self.angle_time = math.pi / 2

        # thread for long-running following component
        self.run_loop_thread = Thread(target=self.run_loop, daemon=True)
        self.run_loop_thread.start()

    def run_loop(self):
        """
        Executes main person following logic
        """
        # Detect sweep of laser scans within range of the front
        # Find the shortest distance
        # Turn to face that point
        # Drive a little forward
        try:
            while True:  # infinite loop for now
                # prevent race conditions
                with self.vel_lock:
                    self.local_angular_vel = self.angular_vel
                    self.local_distance = self.distance
                    self.local_turn_angle = self.turn_angle

                print("start drive sequence")
                self.turn_follow()
                self.drive_forward()
        except Exception as e:
            self.get_logger().error(f"run_loop crashed: {e}")

    def find_person(self, msg: LaserScan):
        """
        Checks the range of laser scans to find the shortest distance

        Args:
            msg (sensor_msgs.msg.LaserScan)

        Updates angle and distance of determined person point
        """
        filtered_ranges = self.handle_scan(msg)
        # print(filtered_ranges)  # for testing

        # calculate angle of closest point
        # TO-DO: add on center of mass calculation later for more accurate reading
        min_angle_idx = min(
            filtered_ranges, key=filtered_ranges.get
        )  # sort based on distance, return angle idx

        angle = msg.angle_min + (min_angle_idx * msg.angle_increment)
        min_distance = min(filtered_ranges.values())

        with self.vel_lock:
            self.turn_angle = angle
            self.angular_vel = angle / self.angle_time  # TO-DO update in a bit
            self.distance = min_distance

    def handle_scan(self, scan: LaserScan):
        """
        Handles and processes messages received on scan topic

        Args:
            msg (sensor_msgs.msg.LaserScan)

        Returns copy of ranges narrowed to search area
        """
        # get scan ranges (90 to -90)
        rad_90 = math.radians(90)  # get 90 deg in rad
        rad_neg_90 = math.radians(-90)  # get -90 deg in rad

        # print(f"angle_min (deg): {math.degrees(scan.angle_min)}")
        # print(f"angle_max (deg): {math.degrees(scan.angle_max)}")

        range_idx_90 = int(
            round((rad_90 - scan.angle_min) / scan.angle_increment)
        )  # get closest index to it

        rad_neg_90 = 1.57
        range_idx_neg_90 = int(
            round((rad_neg_90 - scan.angle_min) / scan.angle_increment)
        )  # get closest to it

        # combine left side of range with right side of range, preserving indices
        filtered_ranges = []
        for i, r in enumerate(scan.ranges):
            if range_idx_neg_90 <= i <= range_idx_90:
                filtered_ranges.append((i, r))

        return dict(filtered_ranges)  # convert to dict for ease of use

    def turn_follow(self):
        """
        Turns to the shortest distance point
        """
        print("should be turning right now")
        # angular_vel = 0.5
        # self.drive(linear=0.0, angular=angular_vel)
        # sleep(self.local_turn_angle / angular_vel)
        # self.drive(0.0, 0.0)

        angular_vel = self.local_angular_vel
        self.drive(linear=0.0, angular=angular_vel)
        sleep(self.angle_time)
        self.drive(0.0, 0.0)
        # TO-DO: change this to based on fixed ang.vel instead;

    def drive_forward(self):
        """
        Drive straight to determined person point
        """
        print("should be driving forward right now")
        distance = self.local_distance
        forward_vel = 0.7
        if distance <= 0:
            return
        duration = distance / forward_vel
        # if not self._e_stop.is_set():
        self.drive(linear=forward_vel, angular=0.0)
        sleep(duration)
        self.drive(0.0, 0.0)

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
    main()
