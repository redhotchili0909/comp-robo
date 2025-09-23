"""
Wall follower module.
"""

import math
from typing import Optional
from threading import Event
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from neato2_interfaces.msg import Bump
from visualization_msgs.msg import Marker


class WallFollowerNode(Node):
    """
    Autonomous wall follower using symmetric angle pairs to stay parallel.
    Includes physical bump-based escape behavior.
    """

    def __init__(self):
        super().__init__("WallFollowerNode")
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.run_loop)
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.subscription_lidar = self.create_subscription(
            LaserScan, "scan", self.process_scan, 10
        )
        self.create_subscription(Bump, "/bump", self._on_bump_msg, 10)
        
        self._enabled = Event()
        self.create_subscription(String, "fsm/state", self._on_fsm_state, 10)

        self.wall_marker_pub = self.create_publisher(Marker, "wall_marker", 10)

        # controller state
        self.angular_vel = 0.0
        self.wall_available = False
        self.mode = "FOLLOW"  # FOLLOW | BACKING | RECOVER_TURN
        self.state_until: Optional[rclpy.time.Time] = None
        self.last_bump_side = "both"  # "left" | "right" | "both"

        self.pair_offset_deg = 35.0
        self.k_parallel = 0.8
        self.follow_side = "auto"    # "left", "right", or "auto"
        self.max_angular_z = 1.0
        self.linear_speed = 0.1
        self.search_linear_speed = 0.08
        self.wall_max_range = 1.0

        self.back_speed = -0.2
        self.back_duration_s = 1.0
        self.recover_turn_speed = 0.6
        self.recover_turn_duration_s = 1.5
 
    def _on_fsm_state(self, msg: String):
        if msg.data == "WALL":
            if not self._enabled.is_set():
                self.get_logger().info("Activated by FSM (WALL)")
            self._enabled.set()
        else:
            if self._enabled.is_set():
                self.get_logger().info("Deactivated by FSM (leaving WALL)")
            self._enabled.clear()
            self._stop_robot()

            # clear wall markers
            marker = Marker()
            marker.action = Marker.DELETEALL
            self.wall_marker_pub.publish(marker)

    def _stop_robot(self):
        # immediately stop the robot
        vel = Twist()
        vel.linear.x = 0.0
        vel.angular.z = 0.0
        self.cmd_pub.publish(vel)

    def run_loop(self):
        # main control loop, runs at fixed interval

        if not self._enabled.is_set():
            self._stop_robot()
            return
        
        vel = Twist()
        now = self.get_clock().now()

        if self.mode == "BACKING":
            # still backing?
            if self.state_until and now < self.state_until:
                vel.linear.x = self.back_speed
                turn = self.recover_turn_speed * 0.5

                if self.last_bump_side == "both":
                    vel.angular.z = 0.0
                elif self.last_bump_side == "left":
                    vel.angular.z = -turn
                else:
                    vel.angular.z = turn
            else:
                # transition to recover turn
                self.mode = "RECOVER_TURN"
                self.state_until = now + Duration(seconds=float(self.recover_turn_duration_s))
                vel.linear.x = 0.0
                vel.angular.z = 0.0
        elif self.mode == "RECOVER_TURN":
            # still turning?
            if self.state_until and now < self.state_until:
                vel.linear.x = 0.0
                # choose turn direction based on last bump side
                if self.last_bump_side == "both":
                    #  default turn left when both were hit
                    vel.angular.z = self.recover_turn_speed
                elif self.last_bump_side == "left":
                    vel.angular.z = -self.recover_turn_speed
                elif self.last_bump_side == "right" :
                    vel.angular.z = self.recover_turn_speed
            else:
                # transition to follow mode
                self.mode = "FOLLOW"
                self.wall_available = False
                vel.linear.x = self.search_linear_speed
                vel.angular.z = 0.0
        else:  # FOLLOW
            if self.wall_available:
                vel.linear.x = self.linear_speed
                vel.angular.z = self.angular_vel
            else:
                # no reliable wall -> creep forward to find one
                vel.linear.x = self.search_linear_speed
                vel.angular.z = 0.0

        self.cmd_pub.publish(vel)

    def range_at_deg(self, scan: LaserScan, deg: float) -> float:
        """
        Return a valid range in the scan frame.
        Returns NaN if out of bounds/invalid.
        """
        rad = math.radians(deg)
        idx = int(round((rad - scan.angle_min) / scan.angle_increment))
        if idx < 0 or idx >= len(scan.ranges):
            return float("nan")
        r = scan.ranges[idx]
        if r == 0.0 or math.isnan(r) or math.isinf(r):
            return float("nan")
        return float(r)

    def _publish_wall_marker(self, scan: LaserScan, p1: Point, p2: Point, color):
        # helper to publish a wall marker for visualization
        marker = Marker()
        marker.header = scan.header
        marker.ns = "wall"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        marker.points = [p1, p2]
        self.wall_marker_pub.publish(marker)

    def process_scan(self, msg: LaserScan):
        # process incoming LIDAR scan to update wall-following control
        if not self._enabled.is_set():
            return
        off = float(self.pair_offset_deg)

        # left side (center ≈ +90° from forward)
        lf = self.range_at_deg(msg, 90.0 - off)  # front-left
        lb = self.range_at_deg(msg, 90.0 + off)  # back-left

        # right side (center ≈ -90° from forward)
        rf = self.range_at_deg(msg, -90.0 + off)  # front-right
        rb = self.range_at_deg(msg, -90.0 - off)  # back-right

        # if escaping, skip control updates
        if self.mode != "FOLLOW":
            return

        # check for NaN values
        left_pair_ok = not (math.isnan(lf) or math.isnan(lb))
        right_pair_ok = not (math.isnan(rf) or math.isnan(rb))

        # decide which side to use
        side = None
        pref = self.follow_side
        if pref == "left" and left_pair_ok:
            side = "left"
        elif pref == "right" and right_pair_ok:
            side = "right"
        else:
            if left_pair_ok:
                side = "left"
            elif right_pair_ok:
                side = "right"

        if side is None:
            self.wall_available = False
            self.angular_vel = 0.0
            return

        # compute angular correction
        if side == "left":
            err_parallel = lf - lb
            az = self.k_parallel * err_parallel
        else:
            err_parallel = rf - rb
            az = -self.k_parallel * err_parallel

        az = float(max(-self.max_angular_z, min(self.max_angular_z, az)))
        self.angular_vel = az
        self.wall_available = True

        # publish wall marker for visualization

        if side == "left":
            a1 = math.radians(90.0 - off)
            a2 = math.radians(90.0 + off)
            r1, r2 = lf, lb
            color = (0.0, 1.0, 0.0, 1.0)
        else:
            a1 = math.radians(-90.0 + off)
            a2 = math.radians(-90.0 - off)
            r1, r2 = rf, rb
            color = (0.0, 0.5, 1.0, 1.0)

        p1 = Point()
        p1.x = float(r1 * math.cos(a1))
        p1.y = float(r1 * math.sin(a1))
        p1.z = 0.0
        p2 = Point()
        p2.x = float(r2 * math.cos(a2))
        p2.y = float(r2 * math.sin(a2))
        p2.z = 0.0
        self._publish_wall_marker(msg, p1, p2, color)

    # bump handling
    def _on_bump_msg(self, msg):
        if self.mode != "FOLLOW" or not self._enabled.is_set():
            return
        left = bool(getattr(msg, "left_front")) or bool(getattr(msg, "left_side"))
        right = bool(getattr(msg, "right_front")) or bool(getattr(msg, "right_side"))
        if left or right:
            if left and not right:
                side = "left"
            elif right and not left:
                side = "right"
            else:
                side = "both"
            self._trigger_bump(side)

    def _trigger_bump(self, side: str):
        self.last_bump_side = side if side in ("left", "right", "both") else "both"
        self.mode = "BACKING"
        self.state_until = self.get_clock().now() + Duration(seconds=float(self.back_duration_s))
        self.get_logger().debug(f"Bump detected on {self.last_bump_side}; backing up.")


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()