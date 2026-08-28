import time
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Empty
from std_srvs.srv import Trigger

class HandshakeController(Node):
    def __init__(self):
        super().__init__("handshake_controller")

        # ==========================================================
        # Parameters
        # ==========================================================
        self.declare_parameter(
            "enable_motion",
            False
        )
        # Approach目标允许的工作空间
        self.declare_parameter(
            "workspace_x_min",
            0.05
        )
        self.declare_parameter(
            "workspace_x_max",
            0.90
        )
        self.declare_parameter(
            "workspace_y_min",
            -0.55
        )
        self.declare_parameter(
            "workspace_y_max",
            0.55
        )
        self.declare_parameter(
            "workspace_z_min",
            0.05
        )

        self.declare_parameter(
            "workspace_z_max",
            0.60
        )

        # 当前TCP到Approach一次最多允许移动多远
        self.declare_parameter(
            "max_move_distance_m",
            0.45
        )

        # 到位判定
        self.declare_parameter(
            "position_tolerance_m",
            0.015
        )

        # 最长等待时间
        self.declare_parameter(
            "motion_timeout_s",
            15.0
        )

        # ==========================================================
        # Read parameters
        # ==========================================================
        self.enable_motion = bool(
            self.get_parameter(
                "enable_motion"
            ).value
        )
        self.workspace_min = np.array(
            [
                float(
                    self.get_parameter(
                        "workspace_x_min"
                    ).value
                ),
                float(
                    self.get_parameter(
                        "workspace_y_min"
                    ).value
                ),
                float(
                    self.get_parameter(
                        "workspace_z_min"
                    ).value
                ),
            ],
            dtype=np.float64,
        )
        self.workspace_max = np.array(
            [
                float(
                    self.get_parameter(
                        "workspace_x_max"
                    ).value
                ),
                float(
                    self.get_parameter(
                        "workspace_y_max"
                    ).value
                ),
                float(
                    self.get_parameter(
                        "workspace_z_max"
                    ).value
                ),
            ],
            dtype=np.float64,
        )
        self.max_move_distance = float(
            self.get_parameter(
                "max_move_distance_m"
            ).value
        )
        self.position_tolerance = float(
            self.get_parameter(
                "position_tolerance_m"
            ).value
        )
        self.motion_timeout = float(
            self.get_parameter(
                "motion_timeout_s"
            ).value
        )

        # ==========================================================
        # Runtime state
        # ==========================================================
        self.approach_pose = None
        self.current_tcp_pose = None
        self.plan_ready = False
        self.command_sent = False
        self.motion_running = False
        self.motion_complete = False
        self.motion_start_time = 0.0

        # ==========================================================
        # Subscribers
        # ==========================================================
        self.create_subscription(
            PoseStamped,
            "/handshake/approach_pose",
            self.approach_callback,
            10
        )

        self.create_subscription(
            Bool,
            "/handshake/plan_ready",
            self.plan_ready_callback,
            10
        )

        self.create_subscription(
            PoseStamped,
            "/feedback/tcp_pose",
            self.tcp_callback,
            10
        )

        self.create_subscription(
            Empty,
            "/handshake/reset_event",
            self.reset_callback,
            10
        )

        # ==========================================================
        # Publisher
        # ==========================================================

        self.move_pub = self.create_publisher(
            PoseStamped,
            "/control/move_p",
            10
        )

        self.reached_pub = self.create_publisher(
            Bool,
            "/handshake/approach_reached",
            10
        )

        # ==========================================================
        # Manual start service
        # ==========================================================

        self.start_service = self.create_service(
            Trigger,
            "/handshake/start_approach",
            self.start_approach_callback
        )

        # ==========================================================
        # Monitor timer
        # ==========================================================

        self.timer = self.create_timer(
            0.1,
            self.monitor_motion
        )

        self.get_logger().info(
            "Handshake STEP 4 controller started"
        )

        if self.enable_motion:

            self.get_logger().warning(
                "REAL ROBOT MOTION ENABLED"
            )

        else:

            self.get_logger().warning(
                "DRY RUN MODE - ROBOT WILL NOT MOVE"
            )

    # ==============================================================
    # Inputs
    # ==============================================================

    def approach_callback(
        self,
        msg: PoseStamped
    ):

        if self.motion_running:
            return

        self.approach_pose = msg

    def plan_ready_callback(
        self,
        msg: Bool
    ):

        self.plan_ready = bool(
            msg.data
        )

    def tcp_callback(
        self,
        msg: PoseStamped
    ):

        self.current_tcp_pose = msg

    # ==============================================================
    # Helpers
    # ==============================================================

    def pose_position_array(
        self,
        msg: PoseStamped
    ):

        return np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64
        )

    def check_target_safety(self):

        if self.approach_pose is None:

            return (
                False,
                "No approach pose available"
            )

        if self.current_tcp_pose is None:

            return (
                False,
                "No TCP feedback available"
            )

        target = self.pose_position_array(
            self.approach_pose
        )

        current = self.pose_position_array(
            self.current_tcp_pose
        )

        # ----------------------------------------------------------
        # Finite check
        # ----------------------------------------------------------

        if not np.all(
            np.isfinite(target)
        ):

            return (
                False,
                "Approach target contains NaN/Inf"
            )

        # ----------------------------------------------------------
        # Workspace check
        # ----------------------------------------------------------

        if np.any(
            target < self.workspace_min
        ):

            return (
                False,
                (
                    "Approach below workspace minimum: "
                    f"{target}"
                )
            )

        if np.any(
            target > self.workspace_max
        ):

            return (
                False,
                (
                    "Approach above workspace maximum: "
                    f"{target}"
                )
            )

        # ----------------------------------------------------------
        # Move distance check
        # ----------------------------------------------------------

        distance = float(
            np.linalg.norm(
                target - current
            )
        )

        if (
            distance
            > self.max_move_distance
        ):

            return (
                False,
                (
                    "Approach is too far from current TCP: "
                    f"{distance:.3f} m > "
                    f"{self.max_move_distance:.3f} m"
                )
            )

        return (
            True,
            (
                f"Safety check passed. "
                f"Move distance = "
                f"{distance:.3f} m"
            )
        )

    # ==============================================================
    # Start approach
    # ==============================================================

    def start_approach_callback(
        self,
        request,
        response
    ):

        if self.motion_running:

            response.success = False
            response.message = (
                "Approach motion already running"
            )

            return response

        if self.motion_complete:

            response.success = False
            response.message = (
                "Approach already completed. "
                "Reset before starting again."
            )

            return response

        if not self.plan_ready:

            response.success = False
            response.message = (
                "Handshake plan is not ready"
            )

            return response

        safe, reason = (
            self.check_target_safety()
        )

        if not safe:

            self.get_logger().error(
                f"PLAN REJECTED: {reason}"
            )

            response.success = False
            response.message = reason

            return response

        target = self.pose_position_array(
            self.approach_pose
        )

        current = self.pose_position_array(
            self.current_tcp_pose
        )

        distance = float(
            np.linalg.norm(
                target - current
            )
        )

        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            "APPROACH MOTION REQUESTED"
        )

        self.get_logger().info(
            f"Current TCP = "
            f"[{current[0]:+.4f}, "
            f"{current[1]:+.4f}, "
            f"{current[2]:+.4f}] m"
        )

        self.get_logger().info(
            f"Approach = "
            f"[{target[0]:+.4f}, "
            f"{target[1]:+.4f}, "
            f"{target[2]:+.4f}] m"
        )

        self.get_logger().info(
            f"Move distance = "
            f"{distance * 100:.1f} cm"
        )

        # ----------------------------------------------------------
        # Dry run
        # ----------------------------------------------------------

        if not self.enable_motion:

            self.get_logger().warning(
                "DRY RUN: command NOT sent to robot"
            )

            response.success = True
            response.message = (
                "Safety check passed; "
                "dry-run only"
            )

            return response

        # ----------------------------------------------------------
        # REAL MOVE_P
        # ----------------------------------------------------------

        command = PoseStamped()

        command.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        command.header.frame_id = (
            "base_link"
        )

        command.pose = (
            self.approach_pose.pose
        )

        self.move_pub.publish(
            command
        )

        self.command_sent = True
        self.motion_running = True
        self.motion_start_time = (
            time.monotonic()
        )

        self.get_logger().warning(
            "MOVE_P command sent"
        )

        self.get_logger().info(
            "======================================"
        )

        response.success = True
        response.message = (
            "Approach motion started"
        )

        return response

    # ==============================================================
    # Monitor
    # ==============================================================

    def monitor_motion(self):

        if not self.motion_running:
            return

        if self.current_tcp_pose is None:
            return

        if self.approach_pose is None:
            return

        current = self.pose_position_array(
            self.current_tcp_pose
        )

        target = self.pose_position_array(
            self.approach_pose
        )

        error = float(
            np.linalg.norm(
                target - current
            )
        )

        # ----------------------------------------------------------
        # Reached
        # ----------------------------------------------------------

        if (
            error
            <= self.position_tolerance
        ):

            self.motion_running = False
            self.motion_complete = True

            reached = Bool()
            reached.data = True

            self.reached_pub.publish(
                reached
            )

            self.get_logger().info(
                "======================================"
            )

            self.get_logger().info(
                "APPROACH REACHED"
            )

            self.get_logger().info(
                f"Position error = "
                f"{error * 1000:.1f} mm"
            )

            self.get_logger().info(
                "STEP 4 COMPLETE"
            )

            self.get_logger().info(
                "======================================"
            )

            return

        # ----------------------------------------------------------
        # Timeout
        # ----------------------------------------------------------

        elapsed = (
            time.monotonic()
            -
            self.motion_start_time
        )

        if (
            elapsed
            > self.motion_timeout
        ):

            self.motion_running = False

            reached = Bool()
            reached.data = False

            self.reached_pub.publish(
                reached
            )

            self.get_logger().error(
                "APPROACH MOTION TIMEOUT"
            )

            self.get_logger().error(
                f"Remaining error: "
                f"{error * 1000:.1f} mm"
            )

    # ==============================================================
    # Reset
    # ==============================================================

    def reset_callback(
        self,
        msg: Empty
    ):

        self.approach_pose = None

        self.plan_ready = False

        self.command_sent = False
        self.motion_running = False
        self.motion_complete = False

        self.motion_start_time = 0.0

        reached = Bool()
        reached.data = False

        self.reached_pub.publish(
            reached
        )

        self.get_logger().info(
            "STEP 4 controller reset"
        )


def main(args=None):
    rclpy.init(args=args)
    node = HandshakeController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()