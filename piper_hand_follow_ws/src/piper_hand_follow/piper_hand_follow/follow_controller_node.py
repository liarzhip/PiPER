import time

import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def quaternion_to_rotation_matrix(q):
    """
    geometry_msgs Quaternion (x, y, z, w)
    -> 3x3 rotation matrix
    """

    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)

    norm = np.sqrt(
        x * x +
        y * y +
        z * z +
        w * w
    )

    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


class HandFollowController(Node):

    def __init__(self):
        super().__init__("hand_follow_controller")

        # ============================================================
        # Parameters
        # ============================================================

        # 是否真的控制机械臂
        self.declare_parameter(
            "enable_motion",
            False
        )

        # 坐标系
        self.declare_parameter(
            "base_frame",
            "base_link"
        )

        self.declare_parameter(
            "camera_frame",
            "camera_color_optical_frame"
        )

        # Topics
        self.declare_parameter(
            "hand_topic",
            "/hand/pose_camera"
        )

        self.declare_parameter(
            "tcp_feedback_topic",
            "/feedback/tcp_pose"
        )

        self.declare_parameter(
            "control_topic",
            "/control/move_p"
        )

        # ============================================================
        # Eye-in-Hand visual servo parameters
        # ============================================================

        # 希望手掌位于：
        #
        # x = 0
        # y = 0
        # z = desired_distance
        #
        # 即画面中央，距离相机 40 cm
        self.declare_parameter(
            "desired_distance_m",
            0.40
        )

        # 比例控制增益
        #
        # error(m) * kp -> velocity(m/s)
        self.declare_parameter(
            "kp",
            0.12
        )

        # XY 平面死区
        self.declare_parameter(
            "deadband_xy_m",
            0.008
        )

        # 深度方向死区
        self.declare_parameter(
            "deadband_z_m",
            0.012
        )

        # 控制频率
        self.declare_parameter(
            "control_rate_hz",
            6.0
        )

        # ============================================================
        # Smooth motion parameters
        # ============================================================

        # TCP 最大跟随速度
        self.declare_parameter(
            "max_speed_mps",
            0.035
        )

        # TCP 最大加速度
        self.declare_parameter(
            "max_accel_mps2",
            0.10
        )

        # 即使出现异常，也限制一次 MOVE_P 最大增量
        self.declare_parameter(
            "max_step_m",
            0.008
        )

        # ============================================================
        # Safety
        # ============================================================

        # 超过这么久没有新的手掌位置，则停止
        self.declare_parameter(
            "hand_timeout_s",
            0.35
        )

        # 合理手掌深度
        self.declare_parameter(
            "min_hand_depth_m",
            0.20
        )

        self.declare_parameter(
            "max_hand_depth_m",
            0.80
        )

        # PIPER-X 第一版保守工作空间
        self.declare_parameter(
            "workspace_x_min",
            0.20
        )

        self.declare_parameter(
            "workspace_x_max",
            0.50
        )

        self.declare_parameter(
            "workspace_y_min",
            -0.25
        )

        self.declare_parameter(
            "workspace_y_max",
            0.25
        )

        self.declare_parameter(
            "workspace_z_min",
            0.15
        )

        self.declare_parameter(
            "workspace_z_max",
            0.50
        )

        # ============================================================
        # Read parameters
        # ============================================================

        self.enable_motion = bool(
            self.get_parameter(
                "enable_motion"
            ).value
        )

        self.base_frame = str(
            self.get_parameter(
                "base_frame"
            ).value
        )

        self.camera_frame = str(
            self.get_parameter(
                "camera_frame"
            ).value
        )

        self.hand_topic = str(
            self.get_parameter(
                "hand_topic"
            ).value
        )

        self.tcp_feedback_topic = str(
            self.get_parameter(
                "tcp_feedback_topic"
            ).value
        )

        self.control_topic = str(
            self.get_parameter(
                "control_topic"
            ).value
        )

        self.desired_distance = float(
            self.get_parameter(
                "desired_distance_m"
            ).value
        )

        self.kp = float(
            self.get_parameter(
                "kp"
            ).value
        )

        self.deadband_xy = float(
            self.get_parameter(
                "deadband_xy_m"
            ).value
        )

        self.deadband_z = float(
            self.get_parameter(
                "deadband_z_m"
            ).value
        )

        self.control_rate = float(
            self.get_parameter(
                "control_rate_hz"
            ).value
        )

        self.max_speed = float(
            self.get_parameter(
                "max_speed_mps"
            ).value
        )

        self.max_accel = float(
            self.get_parameter(
                "max_accel_mps2"
            ).value
        )

        self.max_step = float(
            self.get_parameter(
                "max_step_m"
            ).value
        )

        self.hand_timeout = float(
            self.get_parameter(
                "hand_timeout_s"
            ).value
        )

        self.min_hand_depth = float(
            self.get_parameter(
                "min_hand_depth_m"
            ).value
        )

        self.max_hand_depth = float(
            self.get_parameter(
                "max_hand_depth_m"
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

        # ============================================================
        # TF
        # ============================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ============================================================
        # Runtime state
        # ============================================================

        self.hand_pose = None
        self.tcp_pose = None

        self.hand_last_time = 0.0

        # 当前控制器内部认为的 TCP 速度
        #
        # base_link 坐标系
        self.current_velocity_base = np.zeros(
            3,
            dtype=np.float64
        )

        self.last_control_time = time.monotonic()

        # ============================================================
        # ROS interfaces
        # ============================================================

        self.create_subscription(
            PoseStamped,
            self.hand_topic,
            self.hand_callback,
            10
        )

        self.create_subscription(
            PoseStamped,
            self.tcp_feedback_topic,
            self.tcp_callback,
            10
        )

        self.command_pub = self.create_publisher(
            PoseStamped,
            self.control_topic,
            10
        )

        # 即使 enable_motion=False
        # 也会发布计算出的目标，方便调试
        self.target_pub = self.create_publisher(
            PoseStamped,
            "/hand_follow/target_pose",
            10
        )

        self.timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_loop
        )

        # 减少日志刷屏
        self.last_log_time = 0.0

        # ============================================================
        # Startup information
        # ============================================================

        self.get_logger().info(
            "Eye-in-Hand hand follow controller started"
        )

        self.get_logger().info(
            f"Desired hand position: "
            f"[0.000, 0.000, {self.desired_distance:.3f}] m"
        )

        self.get_logger().info(
            f"Control rate: {self.control_rate:.1f} Hz"
        )

        self.get_logger().info(
            f"Kp: {self.kp:.3f}"
        )

        self.get_logger().info(
            f"Max speed: "
            f"{self.max_speed * 1000.0:.1f} mm/s"
        )

        self.get_logger().info(
            f"Max acceleration: "
            f"{self.max_accel:.3f} m/s^2"
        )

        if self.enable_motion:

            self.get_logger().warn(
                "====================================="
            )

            self.get_logger().warn(
                "REAL ROBOT MOTION ENABLED"
            )

            self.get_logger().warn(
                "====================================="
            )

        else:

            self.get_logger().warn(
                "DRY RUN MODE - ROBOT WILL NOT MOVE"
            )

    # ================================================================
    # Callbacks
    # ================================================================

    def hand_callback(self, msg: PoseStamped):

        self.hand_pose = msg

        self.hand_last_time = (
            time.monotonic()
        )

    def tcp_callback(self, msg: PoseStamped):

        self.tcp_pose = msg

    # ================================================================
    # Helper functions
    # ================================================================

    def reset_velocity(self):

        self.current_velocity_base[:] = 0.0

    def limit_vector_norm(
        self,
        vector,
        max_norm
    ):

        norm = float(
            np.linalg.norm(vector)
        )

        if norm <= max_norm:
            return vector

        if norm < 1e-9:
            return vector

        return (
            vector
            / norm
            * max_norm
        )

    # ================================================================
    # Main controller
    # ================================================================

    def control_loop(self):

        now = time.monotonic()

        dt = (
            now
            - self.last_control_time
        )

        self.last_control_time = now

        # 避免由于系统调度异常导致 dt 过大
        dt = float(
            np.clip(
                dt,
                0.01,
                0.30
            )
        )

        # ------------------------------------------------------------
        # 1. Check input
        # ------------------------------------------------------------

        if self.hand_pose is None:

            self.reset_velocity()
            return

        if self.tcp_pose is None:

            self.reset_velocity()
            return

        hand_age = (
            now
            - self.hand_last_time
        )

        if hand_age > self.hand_timeout:

            self.reset_velocity()
            return

        # ------------------------------------------------------------
        # 2. Hand position in camera optical frame
        # ------------------------------------------------------------

        hand = np.array(
            [
                self.hand_pose.pose.position.x,
                self.hand_pose.pose.position.y,
                self.hand_pose.pose.position.z,
            ],
            dtype=np.float64,
        )

        # ------------------------------------------------------------
        # 3. Depth safety
        # ------------------------------------------------------------

        if (
            hand[2] < self.min_hand_depth
            or
            hand[2] > self.max_hand_depth
        ):

            self.reset_velocity()

            return

        # ------------------------------------------------------------
        # 4. Eye-in-Hand visual error
        #
        # desired:
        #
        # x = 0
        # y = 0
        # z = desired_distance
        #
        # Camera motion follows the error direction.
        #
        # If the camera moves +X,
        # the hand appears toward -X in camera coordinates.
        #
        # Therefore moving camera +error drives error -> 0.
        # ------------------------------------------------------------

        error_camera = np.array(
            [
                hand[0],
                hand[1],
                hand[2]
                - self.desired_distance,
            ],
            dtype=np.float64,
        )

        # ------------------------------------------------------------
        # 5. Deadband
        # ------------------------------------------------------------

        if (
            abs(error_camera[0])
            < self.deadband_xy
        ):
            error_camera[0] = 0.0

        if (
            abs(error_camera[1])
            < self.deadband_xy
        ):
            error_camera[1] = 0.0

        if (
            abs(error_camera[2])
            < self.deadband_z
        ):
            error_camera[2] = 0.0

        # 已经非常接近目标
        if (
            np.linalg.norm(
                error_camera
            )
            < 1e-8
        ):

            self.reset_velocity()
            return

        # ------------------------------------------------------------
        # 6. P controller
        #
        # Error (m)
        # ->
        # desired camera velocity (m/s)
        # ------------------------------------------------------------

        desired_velocity_camera = (
            self.kp
            * error_camera
        )

        # ------------------------------------------------------------
        # 7. Speed limit
        # ------------------------------------------------------------

        desired_velocity_camera = (
            self.limit_vector_norm(
                desired_velocity_camera,
                self.max_speed
            )
        )

        # ------------------------------------------------------------
        # 8. Camera -> Base rotation
        #
        # 这里只转换“运动向量”，
        # 不需要 translation。
        # ------------------------------------------------------------

        try:

            tf = (
                self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.camera_frame,
                    Time(),
                    timeout=Duration(
                        seconds=0.05
                    )
                )
            )

        except TransformException as e:

            self.reset_velocity()

            if (
                now
                - self.last_log_time
                > 1.0
            ):

                self.get_logger().warning(
                    f"TF unavailable: {e}"
                )

                self.last_log_time = now

            return

        R_base_camera = (
            quaternion_to_rotation_matrix(
                tf.transform.rotation
            )
        )

        desired_velocity_base = (
            R_base_camera
            @ desired_velocity_camera
        )

        # ------------------------------------------------------------
        # 9. Acceleration limit
        #
        # 当前速度不能瞬间跳到目标速度。
        # ------------------------------------------------------------

        velocity_change = (
            desired_velocity_base
            - self.current_velocity_base
        )

        max_velocity_change = (
            self.max_accel
            * dt
        )

        velocity_change = (
            self.limit_vector_norm(
                velocity_change,
                max_velocity_change
            )
        )

        self.current_velocity_base += (
            velocity_change
        )

        # 再次保证速度绝不会超限
        self.current_velocity_base = (
            self.limit_vector_norm(
                self.current_velocity_base,
                self.max_speed
            )
        )

        # ------------------------------------------------------------
        # 10. Velocity -> position increment
        # ------------------------------------------------------------

        delta_base = (
            self.current_velocity_base
            * dt
        )

        # 双保险：每周期最大位置变化
        delta_base = (
            self.limit_vector_norm(
                delta_base,
                self.max_step
            )
        )

        # ------------------------------------------------------------
        # 11. Current TCP
        # ------------------------------------------------------------

        current_tcp = np.array(
            [
                self.tcp_pose.pose.position.x,
                self.tcp_pose.pose.position.y,
                self.tcp_pose.pose.position.z,
            ],
            dtype=np.float64,
        )

        # ------------------------------------------------------------
        # 12. New target
        # ------------------------------------------------------------

        target = (
            current_tcp
            + delta_base
        )

        # ------------------------------------------------------------
        # 13. Workspace protection
        # ------------------------------------------------------------

        if np.any(
            target
            < self.workspace_min
        ):

            self.reset_velocity()

            if (
                now
                - self.last_log_time
                > 1.0
            ):

                self.get_logger().warning(
                    "Target below workspace minimum: "
                    f"{target}"
                )

                self.last_log_time = now

            return

        if np.any(
            target
            > self.workspace_max
        ):

            self.reset_velocity()

            if (
                now
                - self.last_log_time
                > 1.0
            ):

                self.get_logger().warning(
                    "Target above workspace maximum: "
                    f"{target}"
                )

                self.last_log_time = now

            return

        # ------------------------------------------------------------
        # 14. Build PoseStamped
        # ------------------------------------------------------------

        target_msg = PoseStamped()

        target_msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        target_msg.header.frame_id = (
            self.base_frame
        )

        target_msg.pose.position.x = float(
            target[0]
        )

        target_msg.pose.position.y = float(
            target[1]
        )

        target_msg.pose.position.z = float(
            target[2]
        )

        # 第一版：
        #
        # 只改变 TCP 的 XYZ，
        # 不改变末端姿态。
        target_msg.pose.orientation = (
            self.tcp_pose.pose.orientation
        )

        # ------------------------------------------------------------
        # 15. Debug target
        # ------------------------------------------------------------

        self.target_pub.publish(
            target_msg
        )

        # ------------------------------------------------------------
        # 16. Real robot control
        # ------------------------------------------------------------

        if self.enable_motion:

            self.command_pub.publish(
                target_msg
            )

        # ------------------------------------------------------------
        # 17. Debug log
        # ------------------------------------------------------------

        if (
            now
            - self.last_log_time
            > 0.5
        ):

            speed_mm_s = (
                np.linalg.norm(
                    self.current_velocity_base
                )
                * 1000.0
            )

            self.get_logger().info(
                "hand_cam="
                f"[{hand[0]:+.3f}, "
                f"{hand[1]:+.3f}, "
                f"{hand[2]:+.3f}]  "
                "error="
                f"[{error_camera[0]:+.3f}, "
                f"{error_camera[1]:+.3f}, "
                f"{error_camera[2]:+.3f}]  "
                f"speed={speed_mm_s:.1f}mm/s  "
                "target="
                f"[{target[0]:+.3f}, "
                f"{target[1]:+.3f}, "
                f"{target[2]:+.3f}]"
            )

            self.last_log_time = now


def main(args=None):

    rclpy.init(args=args)

    node = HandFollowController()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.reset_velocity()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()