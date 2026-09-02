import math
import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


def quat_normalize(q, eps=1e-12):
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < eps:
        return None
    return q / n


def quat_multiply(q1, q2):
    """
    Hamilton product.
    q = [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_to_rotation_matrix(q):
    q = quat_normalize(q)
    if q is None:
        return None

    x, y, z, w = q

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [
                1.0 - 2.0 * (yy + zz),
                2.0 * (xy - wz),
                2.0 * (xz + wy),
            ],
            [
                2.0 * (xy + wz),
                1.0 - 2.0 * (xx + zz),
                2.0 * (yz - wx),
            ],
            [
                2.0 * (xz - wy),
                2.0 * (yz + wx),
                1.0 - 2.0 * (xx + yy),
            ],
        ],
        dtype=np.float64,
    )


class TargetTransformerNode(Node):
    """
    STEP 3:
      /perception/earbud_case_pose_camera
          -> /targets/earbud_case_pose_base_live

      /perception/palm_pose_camera
          -> /targets/palm_pose_base_live

    The node also broadcasts two live TF frames:
      base_link -> earbud_case_live
      base_link -> palm_live

    Important:
    - This node does NOT freeze targets.
    - This node does NOT command robot motion.
    - The eye-in-hand transform
          gripper_base -> camera_color_optical_frame
      must already exist in TF.
    """

    def __init__(self):
        super().__init__("target_transformer_node")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("case_child_frame", "earbud_case_live")
        self.declare_parameter("palm_child_frame", "palm_live")
        self.declare_parameter("tf_timeout_sec", 0.15)

        self.base_frame = str(
            self.get_parameter("base_frame").value
        )
        self.case_child_frame = str(
            self.get_parameter("case_child_frame").value
        )
        self.palm_child_frame = str(
            self.get_parameter("palm_child_frame").value
        )
        self.tf_timeout_sec = float(
            self.get_parameter("tf_timeout_sec").value
        )

        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=10.0)
        )
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )
        self.tf_broadcaster = TransformBroadcaster(
            self
        )

        self.case_pub = self.create_publisher(
            PoseStamped,
            "/targets/earbud_case_pose_base_live",
            10,
        )
        self.palm_pub = self.create_publisher(
            PoseStamped,
            "/targets/palm_pose_base_live",
            10,
        )

        self.case_sub = self.create_subscription(
            PoseStamped,
            "/perception/earbud_case_pose_camera",
            self.case_callback,
            10,
        )
        self.palm_sub = self.create_subscription(
            PoseStamped,
            "/perception/palm_pose_camera",
            self.palm_callback,
            10,
        )

        self.last_tf_warning_ns = 0

        self.get_logger().info(
            "STEP 3 target transformer started."
        )
        self.get_logger().info(
            f"Target frame: {self.base_frame}"
        )
        self.get_logger().info(
            "TF policy: STOP-AND-LOOK / latest TF."
        )
        self.get_logger().info(
            "Live poses may be ignored during robot motion; lock only after stop."
        )

    def _warn_throttled(self, text):
        now_ns = self.get_clock().now().nanoseconds

        if (
            now_ns - self.last_tf_warning_ns
            >= 2_000_000_000
        ):
            self.get_logger().warning(text)
            self.last_tf_warning_ns = now_ns

    def _lookup_transform(self, msg):
        """
        Stop-and-Look policy.

        This project does NOT use visual measurements while the robot is
        moving. Targets are only allowed to be locked after the arm reaches
        a stationary observation pose.

        Therefore we intentionally use the latest available TF rather than
        requesting an exact image timestamp:

            base_link <- camera_color_optical_frame @ latest

        During robot motion the live Base pose may be temporarily inaccurate,
        but it is not allowed to be locked/used for a new target. Once the
        robot has stopped, latest TF is exactly the transform we want.
        """
        source_frame = str(msg.header.frame_id).strip()

        if not source_frame:
            self._warn_throttled(
                "Incoming PoseStamped has empty frame_id."
            )
            return None

        timeout = Duration(
            seconds=self.tf_timeout_sec
        )

        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                Time(),  # latest available TF
                timeout=timeout,
            )

        except TransformException as exc:
            self._warn_throttled(
                f"Latest TF {self.base_frame} <- "
                f"{source_frame} unavailable: {exc}"
            )
            return None

    def _transform_pose(self, msg):
        transform = self._lookup_transform(
            msg
        )

        if transform is None:
            return None

        t = transform.transform.translation
        r = transform.transform.rotation

        p_camera = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64,
        )

        q_base_camera = quat_normalize(
            [
                r.x,
                r.y,
                r.z,
                r.w,
            ]
        )

        q_camera_target = quat_normalize(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )

        if (
            q_base_camera is None
            or
            q_camera_target is None
        ):
            self._warn_throttled(
                "Invalid quaternion received."
            )
            return None

        R_base_camera = (
            quat_to_rotation_matrix(
                q_base_camera
            )
        )

        p_base = (
            R_base_camera @ p_camera
            + np.array(
                [
                    t.x,
                    t.y,
                    t.z,
                ],
                dtype=np.float64,
            )
        )

        q_base_target = quat_normalize(
            quat_multiply(
                q_base_camera,
                q_camera_target,
            )
        )

        if q_base_target is None:
            return None

        out = PoseStamped()
        out.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )
        out.header.frame_id = (
            self.base_frame
        )

        out.pose.position.x = float(
            p_base[0]
        )
        out.pose.position.y = float(
            p_base[1]
        )
        out.pose.position.z = float(
            p_base[2]
        )

        out.pose.orientation.x = float(
            q_base_target[0]
        )
        out.pose.orientation.y = float(
            q_base_target[1]
        )
        out.pose.orientation.z = float(
            q_base_target[2]
        )
        out.pose.orientation.w = float(
            q_base_target[3]
        )

        return out

    def _broadcast_pose_as_tf(
        self,
        pose_msg,
        child_frame,
    ):
        tf_msg = TransformStamped()

        tf_msg.header.stamp = (
            pose_msg.header.stamp
        )
        tf_msg.header.frame_id = (
            self.base_frame
        )
        tf_msg.child_frame_id = (
            child_frame
        )

        tf_msg.transform.translation.x = (
            pose_msg.pose.position.x
        )
        tf_msg.transform.translation.y = (
            pose_msg.pose.position.y
        )
        tf_msg.transform.translation.z = (
            pose_msg.pose.position.z
        )

        tf_msg.transform.rotation = (
            pose_msg.pose.orientation
        )

        self.tf_broadcaster.sendTransform(
            tf_msg
        )

    def case_callback(self, msg):
        out = self._transform_pose(
            msg
        )

        if out is None:
            return

        self.case_pub.publish(
            out
        )

        self._broadcast_pose_as_tf(
            out,
            self.case_child_frame,
        )

    def palm_callback(self, msg):
        out = self._transform_pose(
            msg
        )

        if out is None:
            return

        self.palm_pub.publish(
            out
        )

        self._broadcast_pose_as_tf(
            out,
            self.palm_child_frame,
        )


def main(args=None):
    rclpy.init(args=args)

    node = TargetTransformerNode()

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
