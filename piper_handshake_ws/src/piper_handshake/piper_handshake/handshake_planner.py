import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
)
from rclpy.time import Time
from std_msgs.msg import Bool, Empty
from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)


def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))

    if n < eps:
        return None

    return v / n


def quaternion_to_rotation_matrix(q):
    """
    geometry_msgs Quaternion (x, y, z, w)
    -> 3x3 rotation matrix
    """

    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)

    n = np.sqrt(
        x * x +
        y * y +
        z * z +
        w * w
    )

    if n < 1e-12:
        return None

    x /= n
    y /= n
    z /= n
    w /= n

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


def rotation_matrix_to_quaternion(R):
    """
    3x3 rotation matrix -> normalized quaternion [x, y, z, w]
    """

    R = np.asarray(
        R,
        dtype=np.float64,
    )

    trace = float(
        R[0, 0] +
        R[1, 1] +
        R[2, 2]
    )

    if trace > 0.0:

        s = (
            np.sqrt(trace + 1.0)
            * 2.0
        )

        qw = 0.25 * s
        qx = (
            R[2, 1] - R[1, 2]
        ) / s
        qy = (
            R[0, 2] - R[2, 0]
        ) / s
        qz = (
            R[1, 0] - R[0, 1]
        ) / s

    elif (
        R[0, 0] > R[1, 1]
        and
        R[0, 0] > R[2, 2]
    ):

        s = np.sqrt(
            1.0
            + R[0, 0]
            - R[1, 1]
            - R[2, 2]
        ) * 2.0

        qw = (
            R[2, 1] - R[1, 2]
        ) / s
        qx = 0.25 * s
        qy = (
            R[0, 1] + R[1, 0]
        ) / s
        qz = (
            R[0, 2] + R[2, 0]
        ) / s

    elif R[1, 1] > R[2, 2]:

        s = np.sqrt(
            1.0
            + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        ) * 2.0

        qw = (
            R[0, 2] - R[2, 0]
        ) / s
        qx = (
            R[0, 1] + R[1, 0]
        ) / s
        qy = 0.25 * s
        qz = (
            R[1, 2] + R[2, 1]
        ) / s

    else:

        s = np.sqrt(
            1.0
            + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        ) * 2.0

        qw = (
            R[1, 0] - R[0, 1]
        ) / s
        qx = (
            R[0, 2] + R[2, 0]
        ) / s
        qy = (
            R[1, 2] + R[2, 1]
        ) / s
        qz = 0.25 * s

    q = np.array(
        [qx, qy, qz, qw],
        dtype=np.float64,
    )

    n = float(np.linalg.norm(q))

    if n < 1e-12:
        return None

    return q / n


class HandshakePlanner(Node):

    def __init__(self):
        super().__init__("handshake_planner")

        # ==========================================================
        # Parameters
        # ==========================================================

        self.declare_parameter(
            "base_frame",
            "base_link"
        )

        self.declare_parameter(
            "ee_frame",
            "gripper_base"
        )

        # 距离手掌15 cm的预接近点
        self.declare_parameter(
            "approach_distance_m",
            0.15
        )

        # 最终第一版握手点仍保留5 cm间隙
        self.declare_parameter(
            "handshake_clearance_m",
            0.05
        )

        # 当 palm normal 几乎与 approach Z 平行时，
        # 投影会退化。这个阈值用于触发备用方向。
        self.declare_parameter(
            "orientation_projection_min_norm",
            0.10
        )

        # 对 gripper Y 轴的正负号进行选择：
        # +Y/-Y 对夹爪开合轴本质上是同一条轴线，
        # 因此选择更接近当前 gripper Y 的方向，
        # 可避免 MoveIt 无意义地多转约180度。
        self.declare_parameter(
            "prefer_nearest_y_axis_sign",
            True
        )

        self.base_frame = str(
            self.get_parameter(
                "base_frame"
            ).value
        )

        self.ee_frame = str(
            self.get_parameter(
                "ee_frame"
            ).value
        )

        self.approach_distance = float(
            self.get_parameter(
                "approach_distance_m"
            ).value
        )

        self.handshake_clearance = float(
            self.get_parameter(
                "handshake_clearance_m"
            ).value
        )

        self.projection_min_norm = float(
            self.get_parameter(
                "orientation_projection_min_norm"
            ).value
        )

        self.prefer_nearest_y_axis_sign = bool(
            self.get_parameter(
                "prefer_nearest_y_axis_sign"
            ).value
        )

        # ==========================================================
        # State
        # ==========================================================

        self.plan_generated = False
        self.approach_pose = None
        self.handshake_pose = None

        # ==========================================================
        # TF
        # ==========================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ==========================================================
        # QoS
        # ==========================================================

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ==========================================================
        # Input
        # ==========================================================

        self.create_subscription(
            PoseStamped,
            "/handshake/locked_pose_base",
            self.target_callback,
            10
        )

        self.create_subscription(
            Empty,
            "/handshake/reset_event",
            self.reset_callback,
            10
        )

        # ==========================================================
        # Output
        # ==========================================================

        self.approach_pub = self.create_publisher(
            PoseStamped,
            "/handshake/approach_pose",
            qos
        )

        self.handshake_pub = self.create_publisher(
            PoseStamped,
            "/handshake/handshake_pose",
            qos
        )

        self.ready_pub = self.create_publisher(
            Bool,
            "/handshake/plan_ready",
            qos
        )

        self.timer = self.create_timer(
            0.2,
            self.timer_callback
        )

        self.get_logger().info(
            "Handshake planner started - upgraded palm-guided orientation"
        )

        self.get_logger().info(
            f"Approach distance: "
            f"{self.approach_distance * 100:.1f} cm"
        )

        self.get_logger().info(
            f"Handshake clearance: "
            f"{self.handshake_clearance * 100:.1f} cm"
        )

    # ==============================================================
    # Generate handshake plan
    # ==============================================================

    def target_callback(
        self,
        msg: PoseStamped
    ):

        if self.plan_generated:
            return

        if msg.header.frame_id != self.base_frame:

            self.get_logger().error(
                f"Expected frame '{self.base_frame}', "
                f"but got '{msg.header.frame_id}'"
            )

            return

        # ----------------------------------------------------------
        # Get current gripper pose from TF
        # ----------------------------------------------------------

        try:

            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                Time(),
                timeout=Duration(
                    seconds=0.5
                )
            )

        except TransformException as e:

            self.get_logger().error(
                f"Cannot get "
                f"{self.base_frame} -> "
                f"{self.ee_frame}: {e}"
            )

            return

        current = np.array(
            [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ],
            dtype=np.float64
        )

        hand = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64
        )

        # ----------------------------------------------------------
        # Current gripper -> Hand
        #
        # NEW:
        # 这条方向直接定义未来 gripper_base 的蓝色 Z 轴。
        #
        # 已确认：
        #   gripper Z = 夹爪朝前
        #   gripper Y = 夹爪开合方向
        # ----------------------------------------------------------

        direction = (
            hand - current
        )

        distance = float(
            np.linalg.norm(direction)
        )

        if distance < 1e-6:

            self.get_logger().error(
                "Hand target and gripper position "
                "are nearly identical."
            )

            return

        if distance <= self.approach_distance:

            self.get_logger().error(
                "Hand is already closer than the "
                "approach distance."
            )

            return

        z_gripper = normalize(
            direction
        )

        if z_gripper is None:
            return

        # ----------------------------------------------------------
        # Read full palm orientation from locked_pose_base
        #
        # R_base_palm columns:
        #   [:,0] = palm X / palm width
        #   [:,1] = palm Y / wrist -> middle MCP
        #   [:,2] = palm Z / palm normal
        # ----------------------------------------------------------

        R_base_palm = (
            quaternion_to_rotation_matrix(
                msg.pose.orientation
            )
        )

        if R_base_palm is None:

            self.get_logger().error(
                "Locked palm quaternion is invalid."
            )

            return

        x_palm = R_base_palm[:, 0]
        y_palm = R_base_palm[:, 1]
        z_palm = R_base_palm[:, 2]

        # ----------------------------------------------------------
        # Desired gripper Y axis
        #
        # 用 palm normal 决定夹爪绕其前进Z轴的转角。
        #
        # 先把 palm normal 投影到垂直于 z_gripper 的平面。
        # 这样可以保证：
        #
        #   Z_gripper = 真正朝着手
        #   Y_gripper = 尽量符合手掌朝向
        #
        # 而不会因为手指方向稍微倾斜，
        # 强迫整个机械臂大幅翻腕。
        # ----------------------------------------------------------

        y_hint = z_palm

        y_projected = (
            y_hint
            -
            np.dot(
                y_hint,
                z_gripper
            )
            * z_gripper
        )

        y_projected_norm = float(
            np.linalg.norm(
                y_projected
            )
        )

        # ----------------------------------------------------------
        # Degenerate case fallback
        #
        # 如果 palm normal 几乎平行于接近方向，
        # 用 palm X（手掌宽度方向）作为备用线索。
        # ----------------------------------------------------------

        if (
            y_projected_norm
            < self.projection_min_norm
        ):

            self.get_logger().warning(
                "Palm normal is nearly parallel to "
                "approach direction; using palm X as "
                "orientation fallback."
            )

            y_projected = (
                x_palm
                -
                np.dot(
                    x_palm,
                    z_gripper
                )
                * z_gripper
            )

            y_projected_norm = float(
                np.linalg.norm(
                    y_projected
                )
            )

        # ----------------------------------------------------------
        # Last fallback: current gripper Y axis
        # ----------------------------------------------------------

        R_base_current_gripper = (
            quaternion_to_rotation_matrix(
                tf.transform.rotation
            )
        )

        if R_base_current_gripper is None:

            self.get_logger().error(
                "Current gripper quaternion is invalid."
            )

            return

        current_y = (
            R_base_current_gripper[:, 1]
        )

        if (
            y_projected_norm
            < self.projection_min_norm
        ):

            self.get_logger().warning(
                "Palm orientation projection is still "
                "degenerate; using current gripper Y axis."
            )

            y_projected = (
                current_y
                -
                np.dot(
                    current_y,
                    z_gripper
                )
                * z_gripper
            )

        y_gripper = normalize(
            y_projected
        )

        if y_gripper is None:

            self.get_logger().error(
                "Cannot construct gripper Y axis."
            )

            return

        # ----------------------------------------------------------
        # Choose +/- Y sign nearest to current gripper orientation.
        #
        # 对夹爪开合轴而言 +Y 与 -Y 是同一条物理轴线。
        # 选择更接近当前姿态的符号可以显著减少不必要翻转。
        # ----------------------------------------------------------

        if self.prefer_nearest_y_axis_sign:

            if (
                np.dot(
                    y_gripper,
                    current_y
                )
                < 0.0
            ):

                y_gripper = (
                    -y_gripper
                )

        # ----------------------------------------------------------
        # Construct a strict right-handed gripper frame:
        #
        # x = y × z
        # y = z × x
        #
        # Then X × Y = Z.
        # ----------------------------------------------------------

        x_gripper = normalize(
            np.cross(
                y_gripper,
                z_gripper
            )
        )

        if x_gripper is None:

            self.get_logger().error(
                "Cannot construct gripper X axis."
            )

            return

        y_gripper = normalize(
            np.cross(
                z_gripper,
                x_gripper
            )
        )

        if y_gripper is None:

            self.get_logger().error(
                "Cannot re-orthogonalize gripper Y axis."
            )

            return

        R_base_gripper = np.column_stack(
            (
                x_gripper,
                y_gripper,
                z_gripper,
            )
        )

        q_gripper = (
            rotation_matrix_to_quaternion(
                R_base_gripper
            )
        )

        if q_gripper is None:

            self.get_logger().error(
                "Failed to calculate gripper quaternion."
            )

            return

        # ----------------------------------------------------------
        # Generate positions along gripper forward Z axis
        #
        # Approach -----> Handshake -----> Hand
        #       Z_gripper direction
        # ----------------------------------------------------------

        approach = (
            hand
            -
            z_gripper
            * self.approach_distance
        )

        handshake = (
            hand
            -
            z_gripper
            * self.handshake_clearance
        )

        # ----------------------------------------------------------
        # Approach Pose
        # ----------------------------------------------------------

        approach_msg = PoseStamped()

        approach_msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        approach_msg.header.frame_id = (
            self.base_frame
        )

        approach_msg.pose.position.x = float(
            approach[0]
        )

        approach_msg.pose.position.y = float(
            approach[1]
        )

        approach_msg.pose.position.z = float(
            approach[2]
        )

        approach_msg.pose.orientation.x = float(
            q_gripper[0]
        )

        approach_msg.pose.orientation.y = float(
            q_gripper[1]
        )

        approach_msg.pose.orientation.z = float(
            q_gripper[2]
        )

        approach_msg.pose.orientation.w = float(
            q_gripper[3]
        )

        # ----------------------------------------------------------
        # Handshake Pose
        # ----------------------------------------------------------

        handshake_msg = PoseStamped()

        handshake_msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        handshake_msg.header.frame_id = (
            self.base_frame
        )

        handshake_msg.pose.position.x = float(
            handshake[0]
        )

        handshake_msg.pose.position.y = float(
            handshake[1]
        )

        handshake_msg.pose.position.z = float(
            handshake[2]
        )

        handshake_msg.pose.orientation.x = float(
            q_gripper[0]
        )

        handshake_msg.pose.orientation.y = float(
            q_gripper[1]
        )

        handshake_msg.pose.orientation.z = float(
            q_gripper[2]
        )

        handshake_msg.pose.orientation.w = float(
            q_gripper[3]
        )

        # ----------------------------------------------------------
        # Freeze plan
        # ----------------------------------------------------------

        self.approach_pose = approach_msg
        self.handshake_pose = handshake_msg
        self.plan_generated = True

        self.approach_pub.publish(
            approach_msg
        )

        self.handshake_pub.publish(
            handshake_msg
        )

        ready = Bool()
        ready.data = True

        self.ready_pub.publish(
            ready
        )

        # ----------------------------------------------------------
        # Debug information
        # ----------------------------------------------------------

        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            "HANDSHAKE 6D PLAN GENERATED"
        )

        self.get_logger().info(
            f"Current gripper = "
            f"[{current[0]:+.4f}, "
            f"{current[1]:+.4f}, "
            f"{current[2]:+.4f}] m"
        )

        self.get_logger().info(
            f"Locked hand = "
            f"[{hand[0]:+.4f}, "
            f"{hand[1]:+.4f}, "
            f"{hand[2]:+.4f}] m"
        )

        self.get_logger().info(
            f"Distance to hand = "
            f"{distance * 100:.1f} cm"
        )

        self.get_logger().info(
            f"Approach = "
            f"[{approach[0]:+.4f}, "
            f"{approach[1]:+.4f}, "
            f"{approach[2]:+.4f}] m"
        )

        self.get_logger().info(
            f"Handshake = "
            f"[{handshake[0]:+.4f}, "
            f"{handshake[1]:+.4f}, "
            f"{handshake[2]:+.4f}] m"
        )

        self.get_logger().info(
            f"Z_gripper(forward) = "
            f"[{z_gripper[0]:+.3f}, "
            f"{z_gripper[1]:+.3f}, "
            f"{z_gripper[2]:+.3f}]"
        )

        self.get_logger().info(
            f"Y_gripper(open/close) = "
            f"[{y_gripper[0]:+.3f}, "
            f"{y_gripper[1]:+.3f}, "
            f"{y_gripper[2]:+.3f}]"
        )

        self.get_logger().info(
            f"Target quaternion xyzw = "
            f"[{q_gripper[0]:+.4f}, "
            f"{q_gripper[1]:+.4f}, "
            f"{q_gripper[2]:+.4f}, "
            f"{q_gripper[3]:+.4f}]"
        )

        # Geometry sanity check
        approach_to_hand = float(
            np.linalg.norm(
                hand - approach
            )
        )

        handshake_to_hand = float(
            np.linalg.norm(
                hand - handshake
            )
        )

        self.get_logger().info(
            f"|Hand-Approach| = "
            f"{approach_to_hand * 100:.1f} cm"
        )

        self.get_logger().info(
            f"|Hand-Handshake| = "
            f"{handshake_to_hand * 100:.1f} cm"
        )

        self.get_logger().info(
            "NO ROBOT MOTION IN STEP 3"
        )

        self.get_logger().info(
            "======================================"
        )

    # ==============================================================
    # Republish frozen plan
    # ==============================================================

    def timer_callback(self):

        if not self.plan_generated:
            return

        now = (
            self.get_clock()
            .now()
            .to_msg()
        )

        self.approach_pose.header.stamp = now
        self.handshake_pose.header.stamp = now

        self.approach_pub.publish(
            self.approach_pose
        )

        self.handshake_pub.publish(
            self.handshake_pose
        )

        ready = Bool()
        ready.data = True

        self.ready_pub.publish(
            ready
        )

    # ==============================================================
    # Reset
    # ==============================================================

    def reset_callback(
        self,
        msg: Empty
    ):

        self.plan_generated = False
        self.approach_pose = None
        self.handshake_pose = None

        ready = Bool()
        ready.data = False

        self.ready_pub.publish(
            ready
        )

        self.get_logger().info(
            "Handshake plan cleared"
        )


def main(args=None):

    rclpy.init(args=args)

    node = HandshakePlanner()

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
