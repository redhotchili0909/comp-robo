"""
Wall follower module.
"""

import math
from typing import Optional
from threading import Event
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

try:
    from neato2_interfaces.msg import Bump  # type: ignore
except ImportError:
    Bump = None

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
        if Bump is not None:
            self.create_subscription(Bump, "/bump", self._on_bump_msg, 10)
        
        self._enabled = Event()
        self._estop = Event()
        self.create_subscription(String, "fsm/state", self._on_fsm_state, 10)
        self.create_subscription(Bool, "estop", self._on_estop, 10)

        # Controller state
        self.angular_vel = 0.0
        self.wall_available = False
        self.mode = "FOLLOW"  # FOLLOW | BACKING | RECOVER_TURN
        self.state_until: Optional[rclpy.time.Time] = None
        self.last_bump_side = "both"  # "left" | "right" | "both"

        self.left_wall_mean = 100.0
        self.right_wall_mean = 100.0

        self.pair_offset_deg = 25.0
        self.k_parallel = 0.7
        self.follow_side = "auto"    # "left", "right", or "auto"
        self.max_angular_z = 0.7
        self.linear_speed = 0.3
        self.search_linear_speed = 0.08
        self.wall_max_range = 1.0

        self.back_speed = -0.3
        self.back_duration_s = 1.0
        self.recover_turn_speed = 0.6
        self.recover_turn_duration_s = 1.5

        self.declare_parameters(
            namespace="",
            parameters=[
                ("pair_offset_deg", self.pair_offset_deg),
                ("k_parallel", self.k_parallel),
                ("follow_side", self.follow_side),
                ("max_angular_z", self.max_angular_z),
                ("linear_speed", self.linear_speed),
                ("search_linear_speed", self.search_linear_speed),
                ("wall_max_range", self.wall_max_range),
                ("back_speed", self.back_speed),
                ("back_duration_s", self.back_duration_s),
                ("recover_turn_speed", self.recover_turn_speed),
                ("recover_turn_duration_s", self.recover_turn_duration_s),
            ],
        )
        self.pair_offset_deg = float(self.get_parameter("pair_offset_deg").value)
        self.k_parallel = float(self.get_parameter("k_parallel").value)
        self.follow_side = str(self.get_parameter("follow_side").value)
        self.max_angular_z = float(self.get_parameter("max_angular_z").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.search_linear_speed = float(self.get_parameter("search_linear_speed").value)
        self.wall_max_range = float(self.get_parameter("wall_max_range").value)
        self.back_speed = float(self.get_parameter("back_speed").value)
        self.back_duration_s = float(self.get_parameter("back_duration_s").value)
        self.recover_turn_speed = float(self.get_parameter("recover_turn_speed").value)
        self.recover_turn_duration_s = float(self.get_parameter("recover_turn_duration_s").value)

    def _on_estop(self, msg: Bool):
        if bool(msg.data):
            self._estop.set()
            self._publish_zero()
        else:
            self._estop.clear()

    def _on_fsm_state(self, msg: String):
        if msg.data == "WALL":
            if not self._enabled.is_set():
                self.get_logger().info("Activated by FSM (WALL)")
            self._enabled.set()
        else:
            if self._enabled.is_set():
                self.get_logger().info("Deactivated by FSM (leaving WALL)")
            self._enabled.clear()
            self._publish_zero()

    def _publish_zero(self):
        vel = Twist()
        vel.linear.x = 0.0
        vel.angular.z = 0.0
        self.cmd_pub.publish(vel)

    def run_loop(self):
        if (not self._enabled.is_set()) or self._estop.is_set():
            self._publish_zero()
            return
        vel = Twist()
        now = self.get_clock().now()

        if self.mode == "BACKING":
            if self.state_until and now < self.state_until:
                vel.linear.x = self.back_speed
                turn = self.recover_turn_speed * 0.5
                vel.angular.z = (-turn if self.last_bump_side == "left" else turn) if self.last_bump_side != "both" else 0.0
            else:
                # transition to recover turn
                self.mode = "RECOVER_TURN"
                self.state_until = now + Duration(seconds=float(self.recover_turn_duration_s))
                vel.linear.x = 0.0
                vel.angular.z = 0.0
        elif self.mode == "RECOVER_TURN":
            if self.state_until and now < self.state_until:
                vel.linear.x = 0.0
                vel.angular.z = (-self.recover_turn_speed if self.last_bump_side == "left" else self.recover_turn_speed) if self.last_bump_side != "both" else self.recover_turn_speed
            else:
                # back to follow mode
                self.mode = "FOLLOW"
                self.wall_available = False  # reacquire
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
        Return a valid range at a specific bearing in degrees in the scan frame.
        Returns NaN if out of bounds/invalid or beyond wall_max_range (not considered a wall).
        """
        rad = math.radians(deg)
        idx = int(round((rad - scan.angle_min) / scan.angle_increment))
        if idx < 0 or idx >= len(scan.ranges):
            return float("nan")
        r = scan.ranges[idx]
        if r == 0.0 or math.isinf(r) or math.isnan(r):
            return float("nan")
        if r > self.wall_max_range:
            return float("nan")
        return float(r)

    def process_scan(self, msg: LaserScan):
        if (not self._enabled.is_set()) or self._estop.is_set():
            return
        off = float(self.pair_offset_deg)

        # Left side (center ≈ +90° from forward)
        lf = self.range_at_deg(msg, 90.0 - off)  # front-left
        lb = self.range_at_deg(msg, 90.0 + off)  # back-left
        lc = self.range_at_deg(msg, 90.0)        # center-left

        # Right side (center ≈ -90° from forward)
        rf = self.range_at_deg(msg, -90.0 + off)  # front-right
        rb = self.range_at_deg(msg, -90.0 - off)  # back-right
        rc = self.range_at_deg(msg, -90.0)        # center-right

        # Keep values for side choice (fallback to large when invalid)
        self.left_wall_mean = lc if not math.isnan(lc) else 100.0
        self.right_wall_mean = rc if not math.isnan(rc) else 100.0

        # If escaping, skip control updates
        if self.mode != "FOLLOW":
            return

        # Validity checks for symmetric pairs
        left_pair_ok = not (math.isnan(lf) or math.isnan(lb))
        right_pair_ok = not (math.isnan(rf) or math.isnan(rb))

        # Decide which side to use
        side = None
        pref = (self.follow_side or "auto").lower()
        if pref == "left" and left_pair_ok:
            side = "left"
        elif pref == "right" and right_pair_ok:
            side = "right"
        else:
            if left_pair_ok and right_pair_ok:
                side = "left" if self.left_wall_mean <= self.right_wall_mean else "right"
            elif left_pair_ok:
                side = "left"
            elif right_pair_ok:
                side = "right"

        if side is None:
            self.wall_available = False
            self.angular_vel = 0.0
            return

        # Compute angular correction (parallel-only)
        if side == "left":
            err_parallel = lf - lb
            az = self.k_parallel * err_parallel
        else:
            err_parallel = rf - rb
            az = -self.k_parallel * err_parallel

        az = float(max(-self.max_angular_z, min(self.max_angular_z, az)))
        self.angular_vel = az
        self.wall_available = True

    # Bump handling
    def _on_bump_msg(self, msg):
        if self.mode != "FOLLOW" or (not self._enabled.is_set()) or self._estop.is_set():
            return
        left = bool(getattr(msg, "left_front", 0)) or bool(getattr(msg, "left_side", 0))
        right = bool(getattr(msg, "right_front", 0)) or bool(getattr(msg, "right_side", 0))
        if left or right:
            side = "left" if left and not right else ("right" if right and not left else "both")
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