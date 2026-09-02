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
from std_msgs.msg import Bool, String
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
    return q


def orientation_with_z_toward(R_reference, z_target):
    z = normalize(z_target)
    if z is None:
        return None

    y_ref = np.asarray(R_reference[:, 1], dtype=np.float64)
    y = y_ref - float(np.dot(y_ref, z)) * z
    y = normalize(y)

    if y is None:
        y = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(y, z))) > 0.95:
            y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y = normalize(y - float(np.dot(y, z)) * z)

    if y is None:
        return None

    x = normalize(np.cross(y, z))
    if x is None:
        return None

    y = normalize(np.cross(z, x))
    if y is None:
        return None

    return np.column_stack((x, y, z))


class HandoverPlannerNode(Node):
    """
    Palm Final -> Handover Approach / Final target generator.

    Important geometry:
      gripper_base +Z points toward the hand.
      The held earbud-case center is approximated as:
          gripper_base + object_center_offset_from_gripper_m * (+Z)

    Therefore the FINAL gripper_base stays farther from the palm than the
    desired object-center standoff.

    Palm orientation is NOT used; only the freshly re-detected Palm Final
    position is used for approach direction.
    """

    def __init__(self):
        super().__init__("handover_planner_node")
        self.cb_group = ReentrantCallbackGroup()
        self.lock = threading.Lock()

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter(
            "object_center_offset_from_gripper_m",
            0.138,
        )
        self.declare_parameter(
            "approach_object_standoff_m",
            0.20,
        )
        self.declare_parameter(
            "final_object_standoff_m",
            0.10,
        )
        self.declare_parameter(
            "minimum_palm_distance_m",
            0.12,
        )

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.object_offset = float(
            self.get_parameter(
                "object_center_offset_from_gripper_m"
            ).value
        )
        self.approach_standoff = float(
            self.get_parameter(
                "approach_object_standoff_m"
            ).value
        )
        self.final_standoff = float(
            self.get_parameter(
                "final_object_standoff_m"
            ).value
        )
        self.minimum_palm_distance = float(
            self.get_parameter(
                "minimum_palm_distance_m"
            ).value
        )

        if self.final_standoff <= 0.0:
            raise ValueError("final_object_standoff_m must be > 0")
        if self.approach_standoff <= self.final_standoff:
            raise ValueError(
                "approach_object_standoff_m must be greater than "
                "final_object_standoff_m"
            )

        self.face_pose = None
        self.palm_final = None
        self.palm_final_locked = False

        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.approach_pub = self.create_publisher(
            PoseStamped,
            "/handover/approach_pose",
            latched,
        )
        self.final_pub = self.create_publisher(
            PoseStamped,
            "/handover/final_pose",
            latched,
        )
        self.ready_pub = self.create_publisher(
            Bool,
            "/handover/plan_ready",
            latched,
        )

        self.status_pub = self.create_publisher(
            String,
            "/handover/status",
            latched,
        )

        self.create_subscription(
            PoseStamped,
            "/handover/observe_hand_pose",
            self._face_cb,
            latched,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            PoseStamped,
            "/targets/palm_final_pose_base",
            self._palm_final_cb,
            latched,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            Bool,
            "/targets/palm_final_locked",
            self._locked_cb,
            latched,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/handover/recompute_plan",
            self._recompute_cb,
            callback_group=self.cb_group,
        )

        # Clear only the cached Palm Final / handover readiness.
        # Keep face_pose because it is the already-generated observe-hand
        # reference pose and is still needed when a fresh Palm Final arrives.
        self.create_service(
            Trigger,
            "/handover/reset_plan",
            self._reset_plan_cb,
            callback_group=self.cb_group,
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self._publish_ready(False)
        self._publish_status(
            "waiting for /handover/observe_hand_pose and "
            "/targets/palm_final_pose_base"
        )

        self.get_logger().info(
            "Handover planner started | "
            f"object offset={self.object_offset:.3f} m | "
            f"approach object standoff={self.approach_standoff:.3f} m | "
            f"final object standoff={self.final_standoff:.3f} m."
        )

    def _publish_ready(self, value):
        msg = Bool()
        msg.data = bool(value)
        self.ready_pub.publish(msg)


    def _publish_status(self, text):
        msg = String()
        msg.data = str(text)
        self.status_pub.publish(msg)

    def _face_cb(self, msg):
        with self.lock:
            self.face_pose = msg
        self._try_generate()

    def _palm_final_cb(self, msg):
        with self.lock:
            self.palm_final = msg
        self._try_generate()

    def _locked_cb(self, msg):
        # Keep this status for diagnostics, but do NOT make it a hard
        # prerequisite for target generation.
        #
        # /targets/palm_final_pose_base is already the LOCKED Palm Final
        # output of target_lock_node.  In practice the Bool status topic may
        # be volatile / event-driven, so a node started later can miss the
        # last True sample even though the latched Palm Final pose is valid.
        with self.lock:
            self.palm_final_locked = bool(msg.data)

        self._try_generate()

    def _generate(self):
        with self.lock:
            face = self.face_pose
            palm = self.palm_final
            locked = self.palm_final_locked

        # IMPORTANT:
        # /targets/palm_final_pose_base is itself the locked/frozen Palm Final
        # target.  Therefore receiving that pose is the authoritative
        # condition for handover target generation.
        #
        # Do not hard-block on /targets/palm_final_locked here because that
        # Bool can be event-driven/volatile and may not replay to a planner
        # node that starts after the lock event.
        if face is None and palm is None:
            raise RuntimeError(
                "Missing both /handover/observe_hand_pose and "
                "/targets/palm_final_pose_base."
            )
        if face is None:
            raise RuntimeError(
                "Missing /handover/observe_hand_pose."
            )
        if palm is None:
            raise RuntimeError(
                "Missing /targets/palm_final_pose_base."
            )

        p_ref = np.array([
            face.pose.position.x,
            face.pose.position.y,
            face.pose.position.z,
        ], dtype=np.float64)

        p_palm = np.array([
            palm.pose.position.x,
            palm.pose.position.y,
            palm.pose.position.z,
        ], dtype=np.float64)

        to_palm = p_palm - p_ref
        distance = float(np.linalg.norm(to_palm))

        if distance < self.minimum_palm_distance:
            raise RuntimeError(
                f"Palm Final is too close to current reference pose: "
                f"{distance:.3f} m"
            )

        z = normalize(to_palm)
        if z is None:
            raise RuntimeError("Cannot compute direction to Palm Final.")

        R_ref = quat_to_matrix([
            face.pose.orientation.x,
            face.pose.orientation.y,
            face.pose.orientation.z,
            face.pose.orientation.w,
        ])
        if R_ref is None:
            raise RuntimeError("Invalid Face-Hand orientation.")

        R = orientation_with_z_toward(
            R_ref,
            z,
        )
        if R is None:
            raise RuntimeError("Cannot construct handover orientation.")

        q = matrix_to_quat(R)
        if q is None:
            raise RuntimeError("Cannot compute handover quaternion.")

        # Object center = gripper_base + object_offset * +Z.
        # Keep object center away from the palm by configured standoff.
        approach_gripper_distance = (
            self.object_offset
            + self.approach_standoff
        )
        final_gripper_distance = (
            self.object_offset
            + self.final_standoff
        )

        p_approach = (
            p_palm
            - approach_gripper_distance * z
        )
        p_final = (
            p_palm
            - final_gripper_distance * z
        )

        approach = PoseStamped()
        approach.header.stamp = self.get_clock().now().to_msg()
        approach.header.frame_id = self.base_frame
        approach.pose.position.x = float(p_approach[0])
        approach.pose.position.y = float(p_approach[1])
        approach.pose.position.z = float(p_approach[2])
        approach.pose.orientation.x = float(q[0])
        approach.pose.orientation.y = float(q[1])
        approach.pose.orientation.z = float(q[2])
        approach.pose.orientation.w = float(q[3])

        final = PoseStamped()
        final.header = approach.header
        final.pose.position.x = float(p_final[0])
        final.pose.position.y = float(p_final[1])
        final.pose.position.z = float(p_final[2])
        final.pose.orientation = approach.pose.orientation

        return approach, final, distance

    def _broadcast(self, pose, child):
        tf_msg = TransformStamped()
        tf_msg.header = pose.header
        tf_msg.child_frame_id = child
        tf_msg.transform.translation.x = pose.pose.position.x
        tf_msg.transform.translation.y = pose.pose.position.y
        tf_msg.transform.translation.z = pose.pose.position.z
        tf_msg.transform.rotation = pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf_msg)

    def _try_generate(self):
        try:
            approach, final, distance = self._generate()
        except Exception as exc:
            self._publish_ready(False)
            self._publish_status(
                "NOT READY: " + str(exc)
            )
            return

        self.approach_pub.publish(approach)
        self.final_pub.publish(final)
        self._broadcast(approach, "handover_approach_target")
        self._broadcast(final, "handover_final_target")
        self._publish_ready(True)
        self._publish_status(
            "READY | "
            f"Palm Final distance={distance:.3f} m | "
            f"approach standoff={self.approach_standoff:.3f} m | "
            f"final standoff={self.final_standoff:.3f} m"
        )

        self.get_logger().info(
            "Handover targets ready | "
            f"Palm Final distance={distance:.3f} m | "
            "gripper +Z -> Palm Final."
        )

    def _reset_plan_cb(self, request, response):
        del request

        with self.lock:
            # Do NOT clear face_pose here.
            # The observe-hand pose is still the correct reference after Lift.
            # Only old Palm Final data must be invalidated before re-observation.
            self.palm_final = None
            self.palm_final_locked = False

        self._publish_ready(False)
        self._publish_status(
            "RESET | waiting for fresh /targets/palm_final_pose_base"
        )

        response.success = True
        response.message = (
            "Handover plan reset; cached Palm Final cleared. "
            "Waiting for fresh Palm Final."
        )

        self.get_logger().info(response.message)
        return response

    def _recompute_cb(self, request, response):
        del request

        try:
            approach, final, distance = self._generate()
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self._publish_ready(False)
            self._publish_status(
                "NOT READY: " + str(exc)
            )
            return response

        self.approach_pub.publish(approach)
        self.final_pub.publish(final)
        self._broadcast(approach, "handover_approach_target")
        self._broadcast(final, "handover_final_target")
        self._publish_ready(True)
        self._publish_status(
            "READY | "
            f"Palm Final distance={distance:.3f} m | "
            f"approach standoff={self.approach_standoff:.3f} m | "
            f"final standoff={self.final_standoff:.3f} m"
        )

        response.success = True
        response.message = (
            "HANDOVER PLAN TARGETS OK | "
            f"Palm Final distance={distance:.3f} m | "
            f"approach object standoff={self.approach_standoff:.3f} m | "
            f"final object standoff={self.final_standoff:.3f} m."
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = HandoverPlannerNode()
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
