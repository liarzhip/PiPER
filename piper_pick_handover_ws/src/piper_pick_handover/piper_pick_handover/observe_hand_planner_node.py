#!/usr/bin/env python3
import math
import threading

import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster


def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def quat_to_matrix(q):
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return None
    x /= n
    y /= n
    z /= n
    w /= n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def matrix_to_quat(R):
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2,1] - R[1,2]) / s
        qy = (R[0,2] - R[2,0]) / s
        qz = (R[1,0] - R[0,1]) / s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
        qw = (R[2,1] - R[1,2]) / s
        qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s
        qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
        qw = (R[0,2] - R[2,0]) / s
        qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s
        qz = (R[1,2] + R[2,1]) / s
    else:
        s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
        qw = (R[1,0] - R[0,1]) / s
        qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s
        qz = 0.25 * s

    q = normalize(np.array([qx, qy, qz, qw], dtype=np.float64))
    if q is None:
        return None
    return q


def orientation_with_z_toward(R_reference, z_target):
    """
    Point gripper +Z at the target while preserving the previous gripper +Y
    direction as much as possible. This avoids an arbitrary wrist roll.
    """
    z = normalize(z_target)
    if z is None:
        return None

    y_ref = np.asarray(R_reference[:, 1], dtype=np.float64)
    y = y_ref - float(np.dot(y_ref, z)) * z
    y = normalize(y)

    if y is None:
        y_fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(y_fallback, z))) > 0.95:
            y_fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y = y_fallback - float(np.dot(y_fallback, z)) * z
        y = normalize(y)

    if y is None:
        return None

    x = normalize(np.cross(y, z))
    if x is None:
        return None

    y = normalize(np.cross(z, x))
    if y is None:
        return None

    return np.column_stack((x, y, z))


class ObserveHandPlannerNode(Node):
    """
    STEP 7E face-hand target generator.

    After Lift:
      - KEEP gripper_base position at the Lift target
      - ROTATE gripper_base so its +Z axis points toward Palm Initial

    Then /moveit/execute_observe_hand moves to this pose, stops,
    arms Palm Final and restarts perception.

    This exactly implements:
      Lift -> gripper +Z faces hand -> stop -> re-detect Palm Final.
    """

    def __init__(self):
        super().__init__("observe_hand_planner_node")
        self.cb_group = ReentrantCallbackGroup()
        self.lock = threading.Lock()

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("minimum_hand_distance_m", 0.08)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.minimum_hand_distance = float(
            self.get_parameter("minimum_hand_distance_m").value
        )

        self.palm_initial = None
        self.lift_pose = None
        self.last_pose = None

        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            "/handover/observe_hand_pose",
            latched,
        )
        self.ready_pub = self.create_publisher(
            Bool,
            "/handover/observe_hand_ready",
            latched,
        )

        self.create_subscription(
            PoseStamped,
            "/targets/palm_initial_pose_base",
            self._palm_cb,
            latched,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            PoseStamped,
            "/pick/lift_pose",
            self._lift_cb,
            latched,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/handover/recompute_observe_hand",
            self._recompute_cb,
            callback_group=self.cb_group,
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self._publish_ready(False)

        self.get_logger().info(
            "Face-Hand planner started: Lift position is preserved; "
            "gripper +Z points toward Palm Initial."
        )

    def _publish_ready(self, value):
        msg = Bool()
        msg.data = bool(value)
        self.ready_pub.publish(msg)

    def _palm_cb(self, msg):
        with self.lock:
            self.palm_initial = msg
        self._try_generate()

    def _lift_cb(self, msg):
        with self.lock:
            self.lift_pose = msg
        self._try_generate()

    def _generate(self):
        with self.lock:
            palm = self.palm_initial
            lift = self.lift_pose

        if palm is None or lift is None:
            raise RuntimeError(
                "Palm Initial and Lift target are both required."
            )

        p_lift = np.array([
            lift.pose.position.x,
            lift.pose.position.y,
            lift.pose.position.z,
        ], dtype=np.float64)

        p_palm = np.array([
            palm.pose.position.x,
            palm.pose.position.y,
            palm.pose.position.z,
        ], dtype=np.float64)

        to_hand = p_palm - p_lift
        distance = float(np.linalg.norm(to_hand))

        if distance < self.minimum_hand_distance:
            raise RuntimeError(
                f"Palm Initial is too close to Lift pose: {distance:.3f} m"
            )

        R_lift = quat_to_matrix([
            lift.pose.orientation.x,
            lift.pose.orientation.y,
            lift.pose.orientation.z,
            lift.pose.orientation.w,
        ])
        if R_lift is None:
            raise RuntimeError("Invalid Lift orientation quaternion.")

        R_face = orientation_with_z_toward(
            R_lift,
            to_hand,
        )
        if R_face is None:
            raise RuntimeError(
                "Cannot construct gripper orientation toward Palm Initial."
            )

        q = matrix_to_quat(R_face)
        if q is None:
            raise RuntimeError("Cannot compute face-hand quaternion.")

        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.base_frame

        # Pure reorientation at Lift.
        out.pose.position = lift.pose.position

        out.pose.orientation.x = float(q[0])
        out.pose.orientation.y = float(q[1])
        out.pose.orientation.z = float(q[2])
        out.pose.orientation.w = float(q[3])

        return out, distance

    def _broadcast(self, pose):
        tf_msg = TransformStamped()
        tf_msg.header = pose.header
        tf_msg.child_frame_id = "face_hand_target"
        tf_msg.transform.translation.x = pose.pose.position.x
        tf_msg.transform.translation.y = pose.pose.position.y
        tf_msg.transform.translation.z = pose.pose.position.z
        tf_msg.transform.rotation = pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf_msg)

    def _try_generate(self):
        try:
            pose, distance = self._generate()
        except Exception:
            self._publish_ready(False)
            return

        self.last_pose = pose
        self.pose_pub.publish(pose)
        self._broadcast(pose)
        self._publish_ready(True)

        self.get_logger().info(
            "Face-Hand target ready | "
            f"Palm distance={distance:.3f} m | "
            "Lift XYZ unchanged | gripper +Z -> Palm Initial."
        )

    def _recompute_cb(self, request, response):
        del request

        try:
            pose, distance = self._generate()
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self._publish_ready(False)
            return response

        self.last_pose = pose
        self.pose_pub.publish(pose)
        self._broadcast(pose)
        self._publish_ready(True)

        response.success = True
        response.message = (
            "FACE HAND TARGET OK | "
            f"Palm distance={distance:.3f} m | "
            "gripper +Z points toward Palm Initial."
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ObserveHandPlannerNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
