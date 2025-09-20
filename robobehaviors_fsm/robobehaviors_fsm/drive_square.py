"""
This node drives the robot in a square, while implementing an emergency stop topic.
"""

import rclpy
from rclpy.node import Node
from threading import Thread, Event
from time import sleep
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import math


# TO-DO: currently just runs once, modify to keep going until FSM change
class DrawSquare(Node):
    """
    A class that implements a node to pilot a robot in a square.
    """

    def __init__(self):
        super().__init__("draw_square")
        self.e_stop = Event()

        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Bool, "estop", self.handle_estop, 10)

        # create thread to handle long-running drive component
        self.run_loop_thread = Thread(target=self.run_loop)
        self.run_loop_thread.start()

    def handle_estop(self, msg):
        """
        Handles messages received on estop topic

        Args:
            msg (std_msgs.msg.Bool): message that is True if estop and False otherwise
        """
        if msg.data:
            self.e_stop.set()
            self.drive(linear=0.0, angular=0.0)

    def run_loop(self):
        """
        Executes main driving logic
        """
        self.drive(0.0, 0.0)
        sleep(1)
        for _ in range(4):
            if not self.e_stop.is_set():
                print("driving forward")
                self.drive_forward(0.5)
            if not self.e_stop.is_set():
                print("turning left")
                self.turn_left()
        print("done with square run loop")

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

    def turn_left(self):
        """
        Executes 90 degree left turn
        """
        angular_vel = 0.3
        if not self.e_stop.is_set():
            self.drive(linear=0.0, angular=angular_vel)
            sleep(
                math.pi / angular_vel / 2
            )  # run for as long as takes to get 90 degrees at that angV
            self.drive(linear=0.0, angular=0.0)  # ensure stop

    def drive_forward(self, distance):
        """
        Drive straight for the specified distance

        Args:
            distance (_type_): the distance to drive forward (only positive values)
        """
        # CONSIDER: changing this implementation to have constants defined
        forward_vel = 0.1
        if not self.e_stop.is_set():
            self.drive(linear=forward_vel, angular=0.0)
        sleep(distance / forward_vel)
        self.drive(linear=0.0, angular=0.0)


def main(args=None):
    """
    Overall main sequence
    """
    rclpy.init(args=args)
    node = DrawSquare()
    rclpy.spin(node)
    node.destroy_node()  # get rid of once finished
    rclpy.shutdown()


if __name__ == "__main__":
    main()
