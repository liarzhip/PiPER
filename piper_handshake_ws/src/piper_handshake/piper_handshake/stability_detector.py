from collections import deque
import time

import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Empty
from std_srvs.srv import Trigger


def normalize_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))

    if n < 1e-12:
        return None

    return q / n


def average_quaternions(quaternions):
    """
    Simple sign-aligned quaternion average.

    q and -q represent the same rotation, so all samples are first aligned
    to the sign of the first quaternion before averaging.
    """
    if len(quaternions) == 0:
        return None

    reference = normalize_quaternion(quaternions[0])

    if reference is None:
        return None

    aligned = []

    for q in quaternions:
        qn = normalize_quaternion(q)

        if qn is None:
            return None

        if float(np.dot(qn, reference)) < 0.0:
            qn = -qn

        aligned.append(qn)

    mean_q = np.mean(
        np.asarray(aligned, dtype=np.float64),
        axis=0,
    )

    return normalize_quaternion(mean_q)


def quaternion_angle_deg(q1, q2):
    """
    Smallest angular distance between two unit quaternions in degrees.
    """
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)

    if q1 is None or q2 is None:
        return 180.0

    dot = abs(
        float(
            np.clip(
                np.dot(q1, q2),
                -1.0,
                1.0,
            )
        )
    )

    angle_rad = 2.0 * np.arccos(dot)

    return float(np.degrees(angle_rad))


class HandStabilityDetector(Node):

    def __init__(self):
        super().__init__("hand_stability_detector")

        # ==========================================================
        # Parameters
        # ==========================================================

        # STEP 3升级后，输入的是完整的掌心 6D Pose
        self.declare_parameter(
            "hand_topic",
            "/handshake/palm_pose_camera",
        )

        # 约30Hz视觉输入时，30帧约等于1秒
        self.declare_parameter(
            "window_size",
            30,
        )

        # 位置稳定阈值
        self.declare_parameter(
            "xy_std_threshold_m",
            0.008,
        )

        self.declare_parameter(
            "z_std_threshold_m",
            0.010,
        )

        # 姿态稳定阈值：
        # 窗口内任一样本相对平均姿态的最大夹角不能超过该值
        self.declare_parameter(
            "orientation_threshold_deg",
            8.0,
        )

        self.declare_parameter(
            "hand_timeout_s",
            0.40,
        )

        # ==========================================================
        # Read parameters
        # ==========================================================

        self.hand_topic = str(
            self.get_parameter("hand_topic").value
        )

        self.window_size = int(
            self.get_parameter("window_size").value
        )

        self.xy_threshold = float(
            self.get_parameter("xy_std_threshold_m").value
        )

        self.z_threshold = float(
            self.get_parameter("z_std_threshold_m").value
        )

        self.orientation_threshold_deg = float(
            self.get_parameter("orientation_threshold_deg").value
        )

        self.hand_timeout = float(
            self.get_parameter("hand_timeout_s").value
        )

        # ==========================================================
        # Runtime state
        # ==========================================================

        self.position_history = deque(
            maxlen=self.window_size
        )

        self.orientation_history = deque(
            maxlen=self.window_size
        )

        self.last_hand_time = 0.0

        # 一旦锁定后就不再修改
        self.locked = False
        self.locked_pose = None

        # ==========================================================
        # ROS interfaces
        # ==========================================================

        self.create_subscription(
            PoseStamped,
            self.hand_topic,
            self.hand_callback,
            10,
        )

        self.stable_pub = self.create_publisher(
            Bool,
            "/handshake/hand_stable",
            10,
        )

        self.locked_pose_pub = self.create_publisher(
            PoseStamped,
            "/handshake/locked_pose_camera",
            10,
        )

        self.reset_service = self.create_service(
            Trigger,
            "/handshake/reset",
            self.reset_callback,
        )

        self.reset_event_pub = self.create_publisher(
            Empty,
            "/handshake/reset_event",
            10,
        )

        self.timer = self.create_timer(
            0.1,
            self.timer_callback,
        )

        self.get_logger().info(
            "Hand 6D stability detector started"
        )

        self.get_logger().info(
            f"Input topic: {self.hand_topic}"
        )

        self.get_logger().info(
            f"Window: {self.window_size} samples"
        )

        self.get_logger().info(
            f"XY std threshold: "
            f"{self.xy_threshold * 1000.0:.1f} mm"
        )

        self.get_logger().info(
            f"Z std threshold: "
            f"{self.z_threshold * 1000.0:.1f} mm"
        )

        self.get_logger().info(
            f"Orientation threshold: "
            f"{self.orientation_threshold_deg:.1f} deg"
        )

    # ==============================================================
    # Hand input
    # ==============================================================

    def hand_callback(
        self,
        msg: PoseStamped,
    ):

        self.last_hand_time = time.monotonic()

        if self.locked:
            return

        # ----------------------------------------------------------
        # Position
        # ----------------------------------------------------------

        p = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64,
        )

        # ----------------------------------------------------------
        # Orientation
        # ----------------------------------------------------------

        q = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )

        q = normalize_quaternion(q)

        if q is None:
            self.get_logger().warning(
                "Received invalid zero quaternion; sample ignored."
            )
            return

        self.position_history.append(p)
        self.orientation_history.append(q)

        if (
            len(self.position_history) < self.window_size
            or
            len(self.orientation_history) < self.window_size
        ):
            return

        points = np.asarray(
            self.position_history,
            dtype=np.float64,
        )

        quaternions = list(
            self.orientation_history
        )

        # ----------------------------------------------------------
        # Position stability
        # ----------------------------------------------------------

        std = np.std(
            points,
            axis=0,
        )

        mean_position = np.mean(
            points,
            axis=0,
        )

        position_stable = (
            std[0] < self.xy_threshold
            and
            std[1] < self.xy_threshold
            and
            std[2] < self.z_threshold
        )

        # ----------------------------------------------------------
        # Orientation stability
        # ----------------------------------------------------------

        mean_q = average_quaternions(
            quaternions
        )

        if mean_q is None:
            return

        angular_errors_deg = [
            quaternion_angle_deg(
                q_sample,
                mean_q,
            )
            for q_sample in quaternions
        ]

        max_orientation_error_deg = float(
            max(angular_errors_deg)
        )

        orientation_stable = (
            max_orientation_error_deg
            < self.orientation_threshold_deg
        )

        stable = (
            position_stable
            and
            orientation_stable
        )

        stable_msg = Bool()
        stable_msg.data = bool(stable)

        self.stable_pub.publish(
            stable_msg
        )

        # ----------------------------------------------------------
        # Lock full 6D target
        # ----------------------------------------------------------

        if stable:

            self.locked = True

            locked_msg = PoseStamped()

            locked_msg.header.stamp = (
                self.get_clock()
                .now()
                .to_msg()
            )

            locked_msg.header.frame_id = (
                msg.header.frame_id
            )

            locked_msg.pose.position.x = float(
                mean_position[0]
            )

            locked_msg.pose.position.y = float(
                mean_position[1]
            )

            locked_msg.pose.position.z = float(
                mean_position[2]
            )

            locked_msg.pose.orientation.x = float(
                mean_q[0]
            )

            locked_msg.pose.orientation.y = float(
                mean_q[1]
            )

            locked_msg.pose.orientation.z = float(
                mean_q[2]
            )

            locked_msg.pose.orientation.w = float(
                mean_q[3]
            )

            self.locked_pose = locked_msg

            self.locked_pose_pub.publish(
                locked_msg
            )

            self.get_logger().info(
                "======================================"
            )

            self.get_logger().info(
                "HAND 6D STABLE - TARGET LOCKED"
            )

            self.get_logger().info(
                f"Camera XYZ = "
                f"[{mean_position[0]:+.4f}, "
                f"{mean_position[1]:+.4f}, "
                f"{mean_position[2]:+.4f}] m"
            )

            self.get_logger().info(
                f"Position STD = "
                f"[{std[0] * 1000.0:.2f}, "
                f"{std[1] * 1000.0:.2f}, "
                f"{std[2] * 1000.0:.2f}] mm"
            )

            self.get_logger().info(
                f"Camera quaternion xyzw = "
                f"[{mean_q[0]:+.4f}, "
                f"{mean_q[1]:+.4f}, "
                f"{mean_q[2]:+.4f}, "
                f"{mean_q[3]:+.4f}]"
            )

            self.get_logger().info(
                f"Max orientation deviation = "
                f"{max_orientation_error_deg:.2f} deg"
            )

            self.get_logger().info(
                "======================================"
            )

    # ==============================================================
    # Timeout
    # ==============================================================

    def timer_callback(self):

        if self.locked:
            return

        if self.last_hand_time == 0.0:
            return

        age = (
            time.monotonic()
            - self.last_hand_time
        )

        if age > self.hand_timeout:

            if (
                len(self.position_history) > 0
                or
                len(self.orientation_history) > 0
            ):

                self.get_logger().warning(
                    "Hand lost - 6D stability history cleared"
                )

                self.position_history.clear()
                self.orientation_history.clear()

            stable_msg = Bool()
            stable_msg.data = False

            self.stable_pub.publish(
                stable_msg
            )

    # ==============================================================
    # Reset
    # ==============================================================

    def reset_callback(
        self,
        request,
        response,
    ):

        self.position_history.clear()
        self.orientation_history.clear()

        self.locked = False
        self.locked_pose = None

        response.success = True
        response.message = (
            "Handshake 6D target reset"
        )

        self.get_logger().info(
            "Handshake 6D target reset"
        )

        reset_msg = Empty()

        self.reset_event_pub.publish(
            reset_msg
        )

        return response


def main(args=None):

    rclpy.init(args=args)

    node = HandStabilityDetector()

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
