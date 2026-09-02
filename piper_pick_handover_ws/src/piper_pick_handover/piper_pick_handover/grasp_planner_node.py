import math

import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import (
    Buffer,
    TransformBroadcaster,
    TransformException,
    TransformListener,
)


def normalize(v, eps=1e-10):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def quat_normalize(q, eps=1e-12):
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < eps:
        return None
    return q / n


def quat_to_rotation_matrix(q):
    q = quat_normalize(q)
    if q is None:
        return None

    x, y, z, w = q

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
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(
            1.0 + R[0, 0] - R[1, 1] - R[2, 2]
        ) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(
            1.0 + R[1, 1] - R[0, 0] - R[2, 2]
        ) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s

    else:
        s = math.sqrt(
            1.0 + R[2, 2] - R[0, 0] - R[1, 1]
        ) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return quat_normalize(
        [qx, qy, qz, qw]
    )


def quaternion_distance_deg(q1, q2):
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)

    if q1 is None or q2 is None:
        return float("inf")

    d = abs(float(np.dot(q1, q2)))
    d = max(-1.0, min(1.0, d))

    return math.degrees(
        2.0 * math.acos(d)
    )


class GraspPlannerNode(Node):
    """
    STEP 5 - geometry-only grasp planner.

    Input:
        /targets/earbud_case_pose_base

    Output:
        /pick/pregrasp_pose
        /pick/grasp_pose
        /pick/lift_pose
        /pick/grasp_plan_ready

    Live RViz TF frames:
        pick_pregrasp_target
        pick_grasp_target
        pick_lift_target

    Coordinate convention used here:
        case X : long axis
        case Y : short axis
        case Z : visual surface normal (diagnostic only for grasp tilt)

        gripper_base +Z : forward / approach direction
        gripper_base +Y : finger opening / closing direction

    Tabletop grasp rule:
        gripper +Z  -> exactly base_link -Z (vertical downward)
        gripper +Y  -> projected case short/long axis in the base_link XY plane

    The visual case orientation is used only for in-plane yaw / long-short-axis
    alignment. Visual roll/pitch is deliberately ignored because an eye-in-hand
    oblique view can bias the estimated case surface normal.

    This makes the two fingers clamp across the short dimension of the
    earbud case while approaching from above.

    IMPORTANT:
        This node publishes only target poses.
        It does NOT call MoveIt and does NOT move the robot.
    """

    def __init__(self):
        super().__init__("grasp_planner_node")

        # ==========================================================
        # Parameters
        # ==========================================================

        self.declare_parameter(
            "base_frame",
            "base_link",
        )
        self.declare_parameter(
            "gripper_frame",
            "gripper_base",
        )

        # "short" means gripper Y (opening direction) is aligned to
        # the case short axis. "long" is available for experiments.
        self.declare_parameter(
            "grip_across_axis",
            "short",
        )

        # Approximate forward distance from gripper_base to the
        # useful finger grasp plane. This MUST be tuned in RViz.
        self.declare_parameter(
            "gripper_base_to_grasp_center_m",
            0.138,
        )

        # Positive means the desired contact center is below the
        # segmented top surface, i.e. fingers descend onto the sides.
        self.declare_parameter(
            "grasp_depth_below_surface_m",
            0.012,
        )

        self.declare_parameter(
            "pregrasp_offset_m",
            0.100,
        )
        self.declare_parameter(
            "lift_height_m",
            0.100,
        )

        # Legacy compatibility parameter.
        # Kept so existing grasp.yaml files remain valid, but the visual
        # case normal is no longer used to tilt the gripper. Tabletop grasp
        # always uses base_link +Z as the trusted table normal.
        self.declare_parameter(
            "min_surface_up_dot",
            0.50,
        )

        # Choose the 180-deg symmetric grasp orientation that is
        # closest to the current wrist orientation.
        self.declare_parameter(
            "choose_nearest_wrist_solution",
            True,
        )

        self.declare_parameter(
            "tf_timeout_sec",
            0.15,
        )

        # ==========================================================
        # Read parameters
        # ==========================================================

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

        self.grip_across_axis = str(
            self.get_parameter(
                "grip_across_axis"
            ).value
        ).strip().lower()

        self.tool_forward_length = float(
            self.get_parameter(
                "gripper_base_to_grasp_center_m"
            ).value
        )
        self.grasp_depth = float(
            self.get_parameter(
                "grasp_depth_below_surface_m"
            ).value
        )
        self.pregrasp_offset = float(
            self.get_parameter(
                "pregrasp_offset_m"
            ).value
        )
        self.lift_height = float(
            self.get_parameter(
                "lift_height_m"
            ).value
        )

        self.min_surface_up_dot = float(
            self.get_parameter(
                "min_surface_up_dot"
            ).value
        )
        self.choose_nearest = bool(
            self.get_parameter(
                "choose_nearest_wrist_solution"
            ).value
        )
        self.tf_timeout_sec = float(
            self.get_parameter(
                "tf_timeout_sec"
            ).value
        )

        if self.grip_across_axis not in (
            "short",
            "long",
        ):
            raise ValueError(
                "grip_across_axis must be 'short' or 'long'"
            )

        # ==========================================================
        # TF
        # ==========================================================

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

        # ==========================================================
        # QoS for locked targets / generated plan
        # ==========================================================

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ==========================================================
        # ROS interfaces
        # ==========================================================

        self.pregrasp_pub = self.create_publisher(
            PoseStamped,
            "/pick/pregrasp_pose",
            latched_qos,
        )
        self.grasp_pub = self.create_publisher(
            PoseStamped,
            "/pick/grasp_pose",
            latched_qos,
        )
        self.lift_pub = self.create_publisher(
            PoseStamped,
            "/pick/lift_pose",
            latched_qos,
        )
        self.ready_pub = self.create_publisher(
            Bool,
            "/pick/grasp_plan_ready",
            latched_qos,
        )

        self.case_sub = self.create_subscription(
            PoseStamped,
            "/targets/earbud_case_pose_base",
            self.case_callback,
            latched_qos,
        )

        # target_lock_node publishes false on reset.
        self.case_locked_sub = self.create_subscription(
            Bool,
            "/targets/case_locked",
            self.case_locked_callback,
            latched_qos,
        )

        # Cached targets for TF visualization.
        self.pregrasp_pose = None
        self.grasp_pose = None
        self.lift_pose = None
        self.plan_ready = False

        self.tf_timer = self.create_timer(
            0.10,
            self.broadcast_target_frames,
        )

        self.publish_ready(False)

        self.get_logger().info(
            "STEP 5 grasp planner started."
        )
        self.get_logger().info(
            "Gripper mapping: +Z=approach, +Y=opening direction."
        )
        self.get_logger().info(
            f"Grip across case axis: {self.grip_across_axis}"
        )
        self.get_logger().info(
            "RViz target generation ONLY. NO ROBOT MOTION."
        )

    # ==============================================================
    # TF / Pose helpers
    # ==============================================================

    def get_current_gripper_quaternion(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.gripper_frame,
                Time(),
                timeout=Duration(
                    seconds=self.tf_timeout_sec
                ),
            )

        except TransformException as exc:
            self.get_logger().warning(
                "Current gripper TF unavailable; "
                "using default grasp symmetry. "
                f"Reason: {exc}"
            )
            return None

        r = tf_msg.transform.rotation

        return quat_normalize(
            [
                r.x,
                r.y,
                r.z,
                r.w,
            ]
        )

    def make_pose(
        self,
        xyz,
        q,
    ):
        msg = PoseStamped()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )
        msg.header.frame_id = (
            self.base_frame
        )

        msg.pose.position.x = float(
            xyz[0]
        )
        msg.pose.position.y = float(
            xyz[1]
        )
        msg.pose.position.z = float(
            xyz[2]
        )

        msg.pose.orientation.x = float(
            q[0]
        )
        msg.pose.orientation.y = float(
            q[1]
        )
        msg.pose.orientation.z = float(
            q[2]
        )
        msg.pose.orientation.w = float(
            q[3]
        )

        return msg

    def pose_to_tf(
        self,
        pose,
        child_frame,
    ):
        tf_msg = TransformStamped()

        tf_msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )
        tf_msg.header.frame_id = (
            self.base_frame
        )
        tf_msg.child_frame_id = (
            child_frame
        )

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

        return tf_msg

    def publish_ready(
        self,
        value,
    ):
        msg = Bool()
        msg.data = bool(value)
        self.ready_pub.publish(msg)

    def clear_plan(self):
        self.pregrasp_pose = None
        self.grasp_pose = None
        self.lift_pose = None
        self.plan_ready = False
        self.publish_ready(False)

    # ==============================================================
    # Grasp geometry
    # ==============================================================

    def build_gripper_orientation(
        self,
        case_rotation,
    ):
        """
        Build a tabletop-constrained grasp orientation.

        The earbud-case pose still provides the in-plane long-axis direction,
        but its visual surface normal is NOT used for roll/pitch.

        Returns:
            q_base_gripper,
            surface_up,
            gripper_z,
            chosen_clamp_axis
        """

        # Trusted tabletop normal in base_link.
        base_up = np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )

        case_x = normalize(
            case_rotation[:, 0]
        )
        case_z_visual = normalize(
            case_rotation[:, 2]
        )

        if case_x is None:
            self.get_logger().error(
                "Invalid case long axis."
            )
            return None

        # ----------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT use the visually estimated case normal to tilt the
        # gripper. The eye-in-hand camera can see the earbud case from
        # an oblique angle, which biases the estimated surface normal.
        #
        # For this task the object is known to rest on a horizontal
        # tabletop, so base_link +Z is the trusted surface normal.
        # ----------------------------------------------------------
        surface_up = base_up.copy()

        if case_z_visual is not None:
            normal_dot = abs(
                float(
                    np.dot(
                        case_z_visual,
                        base_up,
                    )
                )
            )
            normal_dot = max(
                0.0,
                min(
                    1.0,
                    normal_dot,
                ),
            )
            visual_tilt_deg = math.degrees(
                math.acos(
                    normal_dot
                )
            )

            self.get_logger().info(
                "Visual case-normal tilt from table normal: "
                f"{visual_tilt_deg:.2f} deg "
                "(ignored for gripper roll/pitch)."
            )

        # ----------------------------------------------------------
        # Preserve only the useful in-plane yaw information.
        #
        # Project the visually estimated case long axis onto the
        # horizontal base_link XY plane.
        # ----------------------------------------------------------
        long_axis = normalize(
            case_x
            -
            np.dot(
                case_x,
                base_up,
            )
            * base_up
        )

        if long_axis is None:
            self.get_logger().error(
                "Case long-axis projection onto tabletop is degenerate."
            )
            return None

        short_axis = normalize(
            np.cross(
                surface_up,
                long_axis,
            )
        )

        if short_axis is None:
            self.get_logger().error(
                "Failed to construct tabletop case short axis."
            )
            return None

        # gripper +Z is the approach direction.
        # Force it to be EXACTLY vertical downward in base_link.
        gripper_z = np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float64,
        )

        if self.grip_across_axis == "short":
            gripper_y = short_axis
        else:
            gripper_y = long_axis

        # Force exact orthonormal right-handed basis.
        gripper_y = normalize(
            gripper_y
            -
            np.dot(
                gripper_y,
                gripper_z,
            )
            * gripper_z
        )

        if gripper_y is None:
            return None

        gripper_x = normalize(
            np.cross(
                gripper_y,
                gripper_z,
            )
        )

        if gripper_x is None:
            return None

        gripper_y = normalize(
            np.cross(
                gripper_z,
                gripper_x,
            )
        )

        if gripper_y is None:
            return None

        # Candidate A
        R_a = np.column_stack(
            (
                gripper_x,
                gripper_y,
                gripper_z,
            )
        )
        q_a = rotation_matrix_to_quaternion(
            R_a
        )

        # Candidate B = same physical clamp line, 180 deg around tool Z.
        R_b = np.column_stack(
            (
                -gripper_x,
                -gripper_y,
                gripper_z,
            )
        )
        q_b = rotation_matrix_to_quaternion(
            R_b
        )

        if q_a is None or q_b is None:
            return None

        chosen_q = q_a

        if self.choose_nearest:
            q_current = (
                self.get_current_gripper_quaternion()
            )

            if q_current is not None:
                da = quaternion_distance_deg(
                    q_current,
                    q_a,
                )
                db = quaternion_distance_deg(
                    q_current,
                    q_b,
                )

                if db < da:
                    chosen_q = q_b
                    gripper_y = -gripper_y
                    gripper_x = -gripper_x

                self.get_logger().info(
                    "Grasp symmetry candidates | "
                    f"A={da:.1f} deg, "
                    f"B={db:.1f} deg -> "
                    f"{'B' if db < da else 'A'}"
                )

        return (
            chosen_q,
            surface_up,
            gripper_z,
            gripper_y,
        )

    def case_callback(
        self,
        msg,
    ):
        if str(
            msg.header.frame_id
        ) != self.base_frame:
            self.get_logger().error(
                "Locked case frame mismatch: "
                f"'{msg.header.frame_id}' "
                f"!= '{self.base_frame}'"
            )
            self.clear_plan()
            return

        case_position = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64,
        )

        q_case = quat_normalize(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )

        if q_case is None:
            self.get_logger().error(
                "Invalid locked case quaternion."
            )
            self.clear_plan()
            return

        R_case = quat_to_rotation_matrix(
            q_case
        )

        orientation = (
            self.build_gripper_orientation(
                R_case
            )
        )

        if orientation is None:
            self.clear_plan()
            return

        (
            q_gripper,
            surface_up,
            gripper_z,
            gripper_y,
        ) = orientation

        # ----------------------------------------------------------
        # Position planning
        #
        # case_position is the segmented TOP-SURFACE center.
        #
        # 1) Move the desired grasp center slightly below that surface
        #    so the fingers can contact the side walls.
        #
        # 2) gripper_base is behind the grasp center by a fixed distance
        #    opposite +Z_gripper.
        # ----------------------------------------------------------

        grasp_center = (
            case_position
            -
            self.grasp_depth
            * surface_up
        )

        grasp_base_position = (
            grasp_center
            -
            self.tool_forward_length
            * gripper_z
        )

        # Back away along -Z_gripper (upward for top grasp).
        pregrasp_position = (
            grasp_base_position
            -
            self.pregrasp_offset
            * gripper_z
        )

        # After grasp, lift vertically in base_link.
        lift_position = (
            grasp_base_position
            +
            np.array(
                [
                    0.0,
                    0.0,
                    self.lift_height,
                ],
                dtype=np.float64,
            )
        )

        self.pregrasp_pose = self.make_pose(
            pregrasp_position,
            q_gripper,
        )
        self.grasp_pose = self.make_pose(
            grasp_base_position,
            q_gripper,
        )
        self.lift_pose = self.make_pose(
            lift_position,
            q_gripper,
        )

        self.pregrasp_pub.publish(
            self.pregrasp_pose
        )
        self.grasp_pub.publish(
            self.grasp_pose
        )
        self.lift_pub.publish(
            self.lift_pose
        )

        self.plan_ready = True
        self.publish_ready(True)

        self.get_logger().info(
            "GRASP GEOMETRY READY"
        )
        self.get_logger().info(
            "Case top center  = "
            f"[{case_position[0]:+.4f}, "
            f"{case_position[1]:+.4f}, "
            f"{case_position[2]:+.4f}]"
        )
        self.get_logger().info(
            "PreGrasp         = "
            f"[{pregrasp_position[0]:+.4f}, "
            f"{pregrasp_position[1]:+.4f}, "
            f"{pregrasp_position[2]:+.4f}]"
        )
        self.get_logger().info(
            "Grasp gripperBase= "
            f"[{grasp_base_position[0]:+.4f}, "
            f"{grasp_base_position[1]:+.4f}, "
            f"{grasp_base_position[2]:+.4f}]"
        )
        self.get_logger().info(
            "Lift             = "
            f"[{lift_position[0]:+.4f}, "
            f"{lift_position[1]:+.4f}, "
            f"{lift_position[2]:+.4f}]"
        )
        self.get_logger().info(
            "Target mapping: "
            "gripper +Z -> base_link -Z (vertical tabletop approach); "
            f"gripper +Y -> projected case {self.grip_across_axis} axis"
        )

    def case_locked_callback(
        self,
        msg,
    ):
        if not msg.data:
            self.clear_plan()
            self.get_logger().info(
                "Case lock cleared -> grasp plan cleared."
            )

    # ==============================================================
    # RViz TF targets
    # ==============================================================

    def broadcast_target_frames(self):
        if not self.plan_ready:
            return

        transforms = [
            self.pose_to_tf(
                self.pregrasp_pose,
                "pick_pregrasp_target",
            ),
            self.pose_to_tf(
                self.grasp_pose,
                "pick_grasp_target",
            ),
            self.pose_to_tf(
                self.lift_pose,
                "pick_lift_target",
            ),
        ]

        self.tf_broadcaster.sendTransform(
            transforms
        )


def main(args=None):
    rclpy.init(args=args)

    node = GraspPlannerNode()

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
