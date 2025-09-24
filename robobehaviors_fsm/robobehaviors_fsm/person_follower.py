"""
This node has the robot follow a person
"""

from time import sleep
from threading import Thread, Lock, Event
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class PersonFollower(Node):
    """
    A class that implements a node to have a robot follow a person
    """

    def __init__(self):
        super().__init__("person_follower")

        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(LaserScan, "scan", self.find_person, 10)

        # subscribe to FSM node
        self._enabled = Event()
        self.create_subscription(String, "fsm/state", self._on_fsm_state, 10)

        # parameters
        self.vel_lock = Lock()
        self.angular_vel = 0.0
        self.distance = 0.0
        self.turn_angle = 0.0

        self.local_distance = 0.0
        self.local_turn_angle = 0.0

        self.angle_time = math.pi / 2

        self.is_person = False

        # thread for long-running following component
        self.run_loop_thread = Thread(target=self.run_loop, daemon=True)
        self.run_loop_thread.start()

        self.get_logger().info("person follower initialized")

    def _on_fsm_state(self, msg: String):
        if msg.data == "PERSON":
            if not self._enabled.is_set():
                self.get_logger().info("Activated by FSM (PERSON)")
            self._enabled.set()
        else:
            if self._enabled.is_set():
                self.get_logger().info("Deactivated by FSM (leaving PERSON)")
            self._enabled.clear()
            self.drive(0.0, 0.0)

    def run_loop(self):
        """
        Executes main person following logic
        """
        try:
            while rclpy.ok():
                if self._enabled.is_set():
                    if self.is_person:
                        # prevent race conditions
                        with self.vel_lock:
                            self.local_distance = self.distance
                            self.local_turn_angle = self.turn_angle
                        self.get_logger().debug("start drive sequence")
                        self.turn_follow()
                        self.drive_forward()
                        self.get_logger().debug("end drive sequence")
                    # don't need else is_person since drive should be 0.0
                else:
                    self.drive(linear=0.0, angular=0.0)
        except Exception as e:
            self.get_logger().error(f"run_loop crashed: {e}")

    def turn_follow(self):
        """
        Turns to the shortest distance point
        """
        self.get_logger().debug(f"should be turning {self.local_turn_angle:.2f} now")
        if self.local_turn_angle != 0:  # if not 0, turn
            self.angular_vel = self.local_turn_angle / self.angle_time
            self.drive(linear=0.0, angular=self.angular_vel)
            sleep(self.angle_time)
        self.drive(linear=0.0, angular=0.0)  # stop robot at end

    def drive_forward(self):
        """
        Drive straight to determined person point
        """
        target_distance = self.local_distance
        forward_vel = 0.1

        if target_distance > 0.60:  # further away
            target_distance *= 0.5
        elif target_distance > 0.30:
            target_distance -= 0.3  # 0.25m is touching object, leave a little buffer
        else:
            target_distance = 0.0  # basically there, don't move

        if target_distance > 0:
            self.get_logger().info(f"driving forward {target_distance:.2f}")
            duration = target_distance / forward_vel
            self.drive(linear=forward_vel, angular=0.0)
            sleep(duration)
        self.drive(linear=0.0, angular=0.0)  # stop robot at end

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

    def find_person(self, msg: LaserScan):
        """
        Checks the range of laser scans to find the shortest distance, and updates angle and distance of that determined person point

        Args:
            msg (sensor_msgs.msg.LaserScan)
        """
        filtered_ranges = self.handle_scan(msg)

        # calculate angle  of closest point
        angle = math.radians(min(filtered_ranges, key=filtered_ranges.get))

        min_distance = min(filtered_ranges.values())  # get min distance

        # convert 270 deg+ angle values to negatives, avoid long way spin
        if angle > (math.pi / 2):
            angle = angle - (2 * math.pi)  # get neg from angle - full circle

        if min_distance < 1.5:  # robot idles if not within distance
            self.get_logger().info(
                f"found person {min_distance:.2f} m, {math.degrees(angle):.2f} deg away"
            )
            self.is_person = True  # enable driving
            with self.vel_lock:
                self.turn_angle = angle
                self.distance = min_distance
        else:
            self.is_person = False

    def handle_scan(self, scan: LaserScan):
        """
        Handles and processes messages received on scan topic to get the scans corresponding to the front of the robot

        Args:
            msg (sensor_msgs.msg.LaserScan)

        Returns:
            A dictionary of the indices, distances narrowed to search area
        """
        #  get filtered range corresponding to 90 to -90 clockwise
        filtered_ranges = []
        for i, r in enumerate(scan.ranges):
            if i < 90 or i > 270:
                filtered_ranges.append((i, r))

        return dict(filtered_ranges)  # convert to dict for ease of use


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
