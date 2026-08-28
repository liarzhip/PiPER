import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    DurabilityPolicy,
    ReliabilityPolicy,
)
from rclpy.time import Time
from std_msgs.msg import Empty
from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)


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
        x * x
        + y * y
        + z * z
        + w * w
    )

    if norm < 1e-12:
        raise ValueError(
            "Quaternion norm is zero."
        )

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


def rotation_matrix_to_quaternion(R):
    """
    3x3 rotation matrix -> normalized quaternion [x, y, z, w]
    """

    R = np.asarray(
        R,
        dtype=np.float64,
    )

    trace = float(
        R[0, 0]
        + R[1, 1]
        + R[2, 2]
    )

    if trace > 0.0:

        s = (
            np.sqrt(trace + 1.0)
            * 2.0
        )

        qw = 0.25 * s
        qx = (
            R[2, 1]
            - R[1, 2]
        ) / s
        qy = (
            R[0, 2]
            - R[2, 0]
        ) / s
        qz = (
            R[1, 0]
            - R[0, 1]
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
            R[2, 1]
            - R[1, 2]
        ) / s
        qx = 0.25 * s
        qy = (
            R[0, 1]
            + R[1, 0]
        ) / s
        qz = (
            R[0, 2]
            + R[2, 0]
        ) / s

    elif R[1, 1] > R[2, 2]:

        s = np.sqrt(
            1.0
            + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        ) * 2.0

        qw = (
            R[0, 2]
            - R[2, 0]
        ) / s
        qx = (
            R[0, 1]
            + R[1, 0]
        ) / s
        qy = 0.25 * s
        qz = (
            R[1, 2]
            + R[2, 1]
        ) / s

    else:

        s = np.sqrt(
            1.0
            + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        ) * 2.0

        qw = (
            R[1, 0]
            - R[0, 1]
        ) / s
        qx = (
            R[0, 2]
            + R[2, 0]
        ) / s
        qy = (
            R[1, 2]
            + R[2, 1]
        ) / s
        qz = 0.25 * s

    q = np.array(
        [qx, qy, qz, qw],
        dtype=np.float64,
    )

    q_norm = float(
        np.linalg.norm(q)
    )

    if q_norm < 1e-12:
        raise ValueError(
            "Failed to build valid quaternion."
        )

    return q / q_norm


class LockedPoseTransformer(Node):

    def __init__(self):
        super().__init__(
            "locked_pose_transformer"
        )

        # ==========================================================
        # Parameters
        # ==========================================================

        self.declare_parameter(
            "base_frame",
            "base_link",
        )

        self.declare_parameter(
            "input_topic",
            "/handshake/locked_pose_camera",
        )

        self.declare_parameter(
            "output_topic",
            "/handshake/locked_pose_base",
        )

        self.base_frame = str(
            self.get_parameter(
                "base_frame"
            ).value
        )

        input_topic = str(
            self.get_parameter(
                "input_topic"
            ).value
        )

        output_topic = str(
            self.get_parameter(
                "output_topic"
            ).value
        )

        # ==========================================================
        # TF
        # ==========================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        # ==========================================================
        # State
        # ==========================================================

        self.locked = False
        self.locked_pose_base = None

        # ==========================================================
        # ROS interfaces
        # ==========================================================

        self.create_subscription(
            PoseStamped,
            input_topic,
            self.pose_callback,
            10,
        )

        # 后启动 ros2 topic echo 也能拿到最后一个冻结目标
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            output_topic,
            qos,
        )

        self.create_subscription(
            Empty,
            "/handshake/reset_event",
            self.reset_callback,
            10,
        )

        # 周期重发冻结后的 Base Pose
        self.timer = self.create_timer(
            0.2,
            self.timer_callback,
        )

        self.get_logger().info(
            "Locked 6D pose transformer started"
        )

        self.get_logger().info(
            f"Target frame: {self.base_frame}"
        )

    # ==============================================================
    # Transform full Pose once
    # ==============================================================

    def pose_callback(
        self,
        msg: PoseStamped,
    ):

        # 已经冻结之后不再更新
        if self.locked:
            return

        source_frame = (
            msg.header.frame_id
        )

        if not source_frame:

            self.get_logger().error(
                "Input pose has empty frame_id"
            )

            return

        try:

            # base_link <- camera_color_optical_frame
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                Time(),
                timeout=Duration(
                    seconds=0.5
                ),
            )

        except TransformException as e:

            self.get_logger().error(
                f"Cannot transform "
                f"{source_frame} -> "
                f"{self.base_frame}: {e}"
            )

            return

        # ==========================================================
        # Position: Camera -> Base
        # ==========================================================

        p_camera = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64,
        )

        translation = np.array(
            [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ],
            dtype=np.float64,
        )

        try:

            R_base_camera = (
                quaternion_to_rotation_matrix(
                    tf.transform.rotation
                )
            )

            R_camera_palm = (
                quaternion_to_rotation_matrix(
                    msg.pose.orientation
                )
            )

        except ValueError as e:

            self.get_logger().error(
                f"Invalid quaternion: {e}"
            )

            return

        p_base = (
            R_base_camera
            @ p_camera
            + translation
        )

        # ==========================================================
        # Orientation: Camera Palm -> Base Palm
        #
        # R_base_palm =
        #     R_base_camera * R_camera_palm
        # ==========================================================

        R_base_palm = (
            R_base_camera
            @ R_camera_palm
        )

        try:

            q_base_palm = (
                rotation_matrix_to_quaternion(
                    R_base_palm
                )
            )

        except ValueError as e:

            self.get_logger().error(
                f"Failed to convert palm orientation: {e}"
            )

            return

        # ==========================================================
        # Freeze full 6D Pose
        # ==========================================================

        result = PoseStamped()

        result.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        result.header.frame_id = (
            self.base_frame
        )

        result.pose.position.x = float(
            p_base[0]
        )

        result.pose.position.y = float(
            p_base[1]
        )

        result.pose.position.z = float(
            p_base[2]
        )

        result.pose.orientation.x = float(
            q_base_palm[0]
        )

        result.pose.orientation.y = float(
            q_base_palm[1]
        )

        result.pose.orientation.z = float(
            q_base_palm[2]
        )

        result.pose.orientation.w = float(
            q_base_palm[3]
        )

        self.locked_pose_base = result
        self.locked = True

        self.pose_pub.publish(
            result
        )

        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            "HAND 6D TARGET TRANSFORMED AND FROZEN"
        )

        self.get_logger().info(
            f"Camera XYZ = "
            f"[{p_camera[0]:+.4f}, "
            f"{p_camera[1]:+.4f}, "
            f"{p_camera[2]:+.4f}] m"
        )

        self.get_logger().info(
            f"Base XYZ = "
            f"[{p_base[0]:+.4f}, "
            f"{p_base[1]:+.4f}, "
            f"{p_base[2]:+.4f}] m"
        )

        self.get_logger().info(
            f"Base palm quaternion xyzw = "
            f"[{q_base_palm[0]:+.4f}, "
            f"{q_base_palm[1]:+.4f}, "
            f"{q_base_palm[2]:+.4f}, "
            f"{q_base_palm[3]:+.4f}]"
        )

        self.get_logger().info(
            "======================================"
        )

    # ==============================================================
    # Keep publishing frozen target
    # ==============================================================

    def timer_callback(self):

        if (
            self.locked
            and
            self.locked_pose_base
            is not None
        ):

            self.locked_pose_base.header.stamp = (
                self.get_clock()
                .now()
                .to_msg()
            )

            self.pose_pub.publish(
                self.locked_pose_base
            )

    # ==============================================================
    # Reset
    # ==============================================================

    def reset_callback(
        self,
        msg: Empty,
    ):

        self.locked = False
        self.locked_pose_base = None

        self.get_logger().info(
            "Base 6D target cleared"
        )


def main(args=None):

    rclpy.init(args=args)

    node = LockedPoseTransformer()

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
