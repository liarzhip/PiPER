#!/usr/bin/env python3
"""
Fresh Palm Final -> Handover Approach / Final target generator.

New architecture:
- No /handover/observe_hand_pose.
- No Palm Initial.
- After Lift, the arm first returns to MECHANICAL HOME while holding the object.
- Palm Final is then detected from this stable pose.
- This planner reads the CURRENT real base_link -> gripper_base TF and uses
  it as the handover reference pose.

Geometry:
    gripper_base +Z points toward Palm Final.

    held object center ~= gripper_base
                          + object_center_offset_from_gripper_m * (+Z)

Palm orientation is intentionally ignored.
"""

import math
import threading

import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_ros import (
    Buffer,
    TransformBroadcaster,
    TransformException,
    TransformListener,
)


def normalize(v, eps=1e-9):
    v = np.asarray(
        v,
        dtype=np.float64,
    )

    n = float(
        np.linalg.norm(v)
    )

    if n < eps:
        return None

    return v / n


def quat_to_matrix(q):
    x, y, z, w = [
        float(v)
        for v in q
    ]

    n = math.sqrt(
        x*x + y*y + z*z + w*w
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
                1 - 2*(y*y + z*z),
                2*(x*y - z*w),
                2*(x*z + y*w),
            ],
            [
                2*(x*y + z*w),
                1 - 2*(x*x + z*z),
                2*(y*z - x*w),
            ],
            [
                2*(x*z - y*w),
                2*(y*z + x*w),
                1 - 2*(x*x + y*y),
            ],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(R):
    R = np.asarray(
        R,
        dtype=np.float64,
    )

    tr = float(
        np.trace(R)
    )

    if tr > 0.0:
        s = math.sqrt(
            tr + 1.0
        ) * 2.0

        qw = 0.25 * s
        qx = (
            R[2, 1]
            -
            R[1, 2]
        ) / s
        qy = (
            R[0, 2]
            -
            R[2, 0]
        ) / s
        qz = (
            R[1, 0]
            -
            R[0, 1]
        ) / s

    elif (
        R[0, 0] > R[1, 1]
        and
        R[0, 0] > R[2, 2]
    ):
        s = math.sqrt(
            1.0
            + R[0, 0]
            - R[1, 1]
            - R[2, 2]
        ) * 2.0

        qw = (
            R[2, 1]
            -
            R[1, 2]
        ) / s
        qx = 0.25 * s
        qy = (
            R[0, 1]
            +
            R[1, 0]
        ) / s
        qz = (
            R[0, 2]
            +
            R[2, 0]
        ) / s

    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(
            1.0
            + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        ) * 2.0

        qw = (
            R[0, 2]
            -
            R[2, 0]
        ) / s
        qx = (
            R[0, 1]
            +
            R[1, 0]
        ) / s
        qy = 0.25 * s
        qz = (
            R[1, 2]
            +
            R[2, 1]
        ) / s

    else:
        s = math.sqrt(
            1.0
            + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        ) * 2.0

        qw = (
            R[1, 0]
            -
            R[0, 1]
        ) / s
        qx = (
            R[0, 2]
            +
            R[2, 0]
        ) / s
        qy = (
            R[1, 2]
            +
            R[2, 1]
        ) / s
        qz = 0.25 * s

    return normalize(
        np.array(
            [
                qx,
                qy,
                qz,
                qw,
            ],
            dtype=np.float64,
        )
    )


def orientation_with_z_toward(
    R_reference,
    z_target,
):
    """
    Point gripper +Z toward the target while preserving the current
    gripper +Y direction as much as possible, avoiding arbitrary wrist roll.
    """

    z = normalize(
        z_target
    )

    if z is None:
        return None

    y_ref = np.asarray(
        R_reference[:, 1],
        dtype=np.float64,
    )

    y = (
        y_ref
        -
        float(
            np.dot(
                y_ref,
                z,
            )
        )
        * z
    )

    y = normalize(y)

    if y is None:
        y = np.array(
            [
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )

        if abs(
            float(
                np.dot(
                    y,
                    z,
                )
            )
        ) > 0.95:
            y = np.array(
                [
                    0.0,
                    1.0,
                    0.0,
                ],
                dtype=np.float64,
            )

        y = normalize(
            y
            -
            float(
                np.dot(
                    y,
                    z,
                )
            )
            * z
        )

    if y is None:
        return None

    x = normalize(
        np.cross(
            y,
            z,
        )
    )

    if x is None:
        return None

    y = normalize(
        np.cross(
            z,
            x,
        )
    )

    if y is None:
        return None

    return np.column_stack(
        (
            x,
            y,
            z,
        )
    )


class HandoverPlannerNode(Node):

    def __init__(self):
        super().__init__(
            "handover_planner_node"
        )

        self.cb_group = (
            ReentrantCallbackGroup()
        )
        self.lock = threading.Lock()

        # ----------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------
        self.declare_parameter(
            "base_frame",
            "base_link",
        )
        self.declare_parameter(
            "gripper_frame",
            "gripper_base",
        )
        self.declare_parameter(
            "tf_timeout_sec",
            0.5,
        )

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

        self.base_frame = str(
            self.get_parameter(
                "base_frame"
            ).value
        )
        self.gripper_frame = str(
            self.get_parameter(
                "gripper_frame"
            ).value
        )
        self.tf_timeout = float(
            self.get_parameter(
                "tf_timeout_sec"
            ).value
        )

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
            raise ValueError(
                "final_object_standoff_m must be > 0"
            )

        if (
            self.approach_standoff
            <=
            self.final_standoff
        ):
            raise ValueError(
                "approach_object_standoff_m must be greater than "
                "final_object_standoff_m"
            )

        # ----------------------------------------------------------
        # State
        # ----------------------------------------------------------
        self.palm_final = None
        self.palm_final_locked = False

        # ----------------------------------------------------------
        # TF
        # ----------------------------------------------------------
        self.tf_buffer = Buffer(
            cache_time=Duration(
                seconds=10.0
            )
        )

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.tf_broadcaster = (
            TransformBroadcaster(self)
        )

        # ----------------------------------------------------------
        # QoS
        # ----------------------------------------------------------
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ----------------------------------------------------------
        # Outputs
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # Inputs
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # Services
        # ----------------------------------------------------------
        self.create_service(
            Trigger,
            "/handover/recompute_plan",
            self._recompute_cb,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/handover/reset_plan",
            self._reset_plan_cb,
            callback_group=self.cb_group,
        )

        self._publish_ready(False)
        self._publish_status(
            "waiting for fresh /targets/palm_final_pose_base at MECHANICAL HOME"
        )

        self.get_logger().info(
            "Handover planner started."
        )
        self.get_logger().info(
            f"Reference pose = CURRENT TF "
            f"{self.base_frame} -> {self.gripper_frame}"
        )
        self.get_logger().info(
            "No Palm Initial / observe-hand pose is used."
        )
        self.get_logger().info(
            f"object offset={self.object_offset:.3f} m | "
            f"approach object standoff={self.approach_standoff:.3f} m | "
            f"final object standoff={self.final_standoff:.3f} m."
        )

    # ==============================================================
    # Helpers
    # ==============================================================

    def _publish_ready(
        self,
        value,
    ):
        msg = Bool()
        msg.data = bool(value)
        self.ready_pub.publish(msg)

    def _publish_status(
        self,
        text,
    ):
        msg = String()
        msg.data = str(text)
        self.status_pub.publish(msg)

    def _current_gripper_reference(self):
        """
        Read current REAL base_link -> gripper_base TF.

        This is expected to be called after the arm has returned to MECHANICAL HOME
        while still holding the earbud case.
        """

        try:
            tf_msg = (
                self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.gripper_frame,
                    Time(),
                    timeout=Duration(
                        seconds=self.tf_timeout
                    ),
                )
            )

        except TransformException as exc:
            raise RuntimeError(
                f"Cannot read current TF "
                f"{self.base_frame} <- {self.gripper_frame}: {exc}"
            ) from exc

        p_ref = np.array(
            [
                tf_msg.transform.translation.x,
                tf_msg.transform.translation.y,
                tf_msg.transform.translation.z,
            ],
            dtype=np.float64,
        )

        R_ref = quat_to_matrix(
            [
                tf_msg.transform.rotation.x,
                tf_msg.transform.rotation.y,
                tf_msg.transform.rotation.z,
                tf_msg.transform.rotation.w,
            ]
        )

        if R_ref is None:
            raise RuntimeError(
                "Invalid current gripper TF quaternion."
            )

        return (
            p_ref,
            R_ref,
        )

    # ==============================================================
    # Inputs
    # ==============================================================

    def _palm_final_cb(
        self,
        msg,
    ):
        with self.lock:
            self.palm_final = msg

        self._try_generate()

    def _locked_cb(
        self,
        msg,
    ):
        with self.lock:
            self.palm_final_locked = bool(
                msg.data
            )

        if not msg.data:
            self._publish_ready(False)

        self._try_generate()

    # ==============================================================
    # Geometry
    # ==============================================================

    def _generate(self):
        with self.lock:
            palm = self.palm_final
            locked = self.palm_final_locked

        # New architecture deliberately requires the fresh lock Boolean.
        # Both pose and Bool are transient-local in target_lock_node.
        if not locked:
            raise RuntimeError(
                "Palm Final is not freshly locked."
            )

        if palm is None:
            raise RuntimeError(
                "Missing /targets/palm_final_pose_base."
            )

        (
            p_ref,
            R_ref,
        ) = self._current_gripper_reference()

        p_palm = np.array(
            [
                palm.pose.position.x,
                palm.pose.position.y,
                palm.pose.position.z,
            ],
            dtype=np.float64,
        )

        to_palm = (
            p_palm
            -
            p_ref
        )

        distance = float(
            np.linalg.norm(
                to_palm
            )
        )

        if distance < self.minimum_palm_distance:
            raise RuntimeError(
                "Palm Final is too close to current MECHANICAL HOME gripper pose: "
                f"{distance:.3f} m"
            )

        z = normalize(
            to_palm
        )

        if z is None:
            raise RuntimeError(
                "Cannot compute direction to Palm Final."
            )

        R = orientation_with_z_toward(
            R_ref,
            z,
        )

        if R is None:
            raise RuntimeError(
                "Cannot construct handover orientation."
            )

        q = matrix_to_quat(R)

        if q is None:
            raise RuntimeError(
                "Cannot compute handover quaternion."
            )

        # Object center = gripper_base + object_offset * +Z.
        approach_gripper_distance = (
            self.object_offset
            +
            self.approach_standoff
        )

        final_gripper_distance = (
            self.object_offset
            +
            self.final_standoff
        )

        p_approach = (
            p_palm
            -
            approach_gripper_distance
            * z
        )

        p_final = (
            p_palm
            -
            final_gripper_distance
            * z
        )

        approach = PoseStamped()
        approach.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )
        approach.header.frame_id = (
            self.base_frame
        )

        approach.pose.position.x = float(
            p_approach[0]
        )
        approach.pose.position.y = float(
            p_approach[1]
        )
        approach.pose.position.z = float(
            p_approach[2]
        )

        approach.pose.orientation.x = float(
            q[0]
        )
        approach.pose.orientation.y = float(
            q[1]
        )
        approach.pose.orientation.z = float(
            q[2]
        )
        approach.pose.orientation.w = float(
            q[3]
        )

        final = PoseStamped()
        final.header = approach.header

        final.pose.position.x = float(
            p_final[0]
        )
        final.pose.position.y = float(
            p_final[1]
        )
        final.pose.position.z = float(
            p_final[2]
        )
        final.pose.orientation = (
            approach.pose.orientation
        )

        return (
            approach,
            final,
            distance,
            p_ref,
        )

    def _broadcast(
        self,
        pose,
        child,
    ):
        tf_msg = TransformStamped()
        tf_msg.header = pose.header
        tf_msg.child_frame_id = child

        tf_msg.transform.translation.x = (
            pose.pose.position.x
        )
        tf_msg.transform.translation.y = (
            pose.pose.position.y
        )
        tf_msg.transform.translation.z = (
            pose.pose.position.z
        )

        tf_msg.transform.rotation = (
            pose.pose.orientation
        )

        self.tf_broadcaster.sendTransform(
            tf_msg
        )

    def _publish_plan(
        self,
        approach,
        final,
        distance,
        p_ref,
    ):
        self.approach_pub.publish(
            approach
        )
        self.final_pub.publish(
            final
        )

        self._broadcast(
            approach,
            "handover_approach_target",
        )
        self._broadcast(
            final,
            "handover_final_target",
        )

        self._publish_ready(True)

        self._publish_status(
            "READY | "
            f"Palm Final distance={distance:.3f} m | "
            f"HOME gripper ref=[{p_ref[0]:+.3f}, "
            f"{p_ref[1]:+.3f}, {p_ref[2]:+.3f}] m | "
            f"approach standoff={self.approach_standoff:.3f} m | "
            f"final standoff={self.final_standoff:.3f} m"
        )

        self.get_logger().info(
            "Handover targets ready | "
            f"Palm Final distance={distance:.3f} m | "
            "reference=CURRENT MECHANICAL HOME gripper TF | "
            "gripper +Z -> Palm Final."
        )

    def _try_generate(self):
        try:
            (
                approach,
                final,
                distance,
                p_ref,
            ) = self._generate()

        except Exception as exc:
            self._publish_ready(False)
            self._publish_status(
                "NOT READY: "
                + str(exc)
            )
            return

        self._publish_plan(
            approach,
            final,
            distance,
            p_ref,
        )

    # ==============================================================
    # Services
    # ==============================================================

    def _reset_plan_cb(
        self,
        request,
        response,
    ):
        del request

        with self.lock:
            self.palm_final = None
            self.palm_final_locked = False

        self._publish_ready(False)

        self._publish_status(
            "RESET | waiting for fresh Palm Final at MECHANICAL HOME"
        )

        response.success = True
        response.message = (
            "Handover plan reset; cached Palm Final cleared. "
            "Waiting for fresh Palm Final at MECHANICAL HOME."
        )

        self.get_logger().info(
            response.message
        )

        return response

    def _recompute_cb(
        self,
        request,
        response,
    ):
        del request

        try:
            (
                approach,
                final,
                distance,
                p_ref,
            ) = self._generate()

        except Exception as exc:
            response.success = False
            response.message = str(exc)

            self._publish_ready(False)
            self._publish_status(
                "NOT READY: "
                + str(exc)
            )

            return response

        self._publish_plan(
            approach,
            final,
            distance,
            p_ref,
        )

        response.success = True
        response.message = (
            "HANDOVER PLAN TARGETS OK | "
            "reference=current MECHANICAL HOME gripper TF | "
            f"Palm Final distance={distance:.3f} m | "
            f"approach object standoff={self.approach_standoff:.3f} m | "
            f"final object standoff={self.final_standoff:.3f} m."
        )

        return response


def main(args=None):
    rclpy.init(args=args)

    node = HandoverPlannerNode()

    executor = MultiThreadedExecutor(
        num_threads=3
    )
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
