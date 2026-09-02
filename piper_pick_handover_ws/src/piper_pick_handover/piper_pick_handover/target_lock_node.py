import math
from collections import deque

import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


def normalize_quaternion(q, eps=1e-12):
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < eps:
        return None
    return q / n


def pose_to_arrays(msg):
    p = np.array(
        [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ],
        dtype=np.float64,
    )

    q = normalize_quaternion(
        [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
    )

    return p, q


def quaternion_average(quaternions):
    """
    Average unit quaternions after resolving q / -q sign ambiguity.
    Sufficient here because a stable window contains only small rotations.
    """
    if not quaternions:
        return None

    q0 = normalize_quaternion(quaternions[0])
    if q0 is None:
        return None

    aligned = []

    for q in quaternions:
        q = normalize_quaternion(q)

        if q is None:
            return None

        if float(np.dot(q, q0)) < 0.0:
            q = -q

        aligned.append(q)

    mean_q = np.mean(
        np.stack(aligned),
        axis=0,
    )

    return normalize_quaternion(mean_q)


def quaternion_angle_deg(q1, q2):
    """
    Smallest orientation difference between two quaternions, degrees.
    """
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)

    if q1 is None or q2 is None:
        return float("inf")

    dot = abs(
        float(
            np.dot(q1, q2)
        )
    )

    dot = max(
        -1.0,
        min(1.0, dot),
    )

    return math.degrees(
        2.0 * math.acos(dot)
    )


class PoseWindow:
    def __init__(self, size):
        self.size = int(size)
        self.samples = deque(
            maxlen=self.size
        )

    def clear(self):
        self.samples.clear()

    def append(self, msg):
        p, q = pose_to_arrays(msg)

        if q is None:
            return False

        self.samples.append(
            (
                p,
                q,
                msg,
            )
        )

        return True

    def full(self):
        return (
            len(self.samples)
            >= self.size
        )

    def analyze(self):
        """
        Returns:
            stable statistics + averaged PoseStamped source data
        """
        if not self.full():
            return None

        positions = np.stack(
            [
                sample[0]
                for sample in self.samples
            ]
        )

        quaternions = [
            sample[1]
            for sample in self.samples
        ]

        mean_p = np.mean(
            positions,
            axis=0,
        )

        std_xyz = np.std(
            positions,
            axis=0,
        )

        mean_q = quaternion_average(
            quaternions
        )

        if mean_q is None:
            return None

        angle_errors = np.array(
            [
                quaternion_angle_deg(
                    q,
                    mean_q,
                )
                for q in quaternions
            ],
            dtype=np.float64,
        )

        # Use RMS orientation variation.
        angle_rms_deg = float(
            np.sqrt(
                np.mean(
                    angle_errors ** 2
                )
            )
        )

        newest_msg = (
            self.samples[-1][2]
        )

        return {
            "mean_position": mean_p,
            "mean_quaternion": mean_q,
            "std_xyz": std_xyz,
            "max_position_std": float(
                np.max(std_xyz)
            ),
            "angle_rms_deg": (
                angle_rms_deg
            ),
            "newest_msg": newest_msg,
        }


class TargetLockNode(Node):
    """
    STEP 4 target locking.

    Input LIVE targets:
      /targets/earbud_case_pose_base_live
      /targets/palm_pose_base_live

    Output LOCKED targets:
      /targets/earbud_case_pose_base
      /targets/palm_initial_pose_base
      /targets/palm_final_pose_base

    Startup:
      - earbud case is automatically locked once stable.
      - palm_initial is automatically locked once stable.

    Later:
      After the robot has grasped/lifted the case and reached the
      observe-hand pose, call:

        /targets/arm_palm_final

      Then this node discards old palm samples and waits for a NEW,
      stable palm measurement before locking palm_final.

    No robot motion occurs here.
    """

    def __init__(self):
        super().__init__(
            "target_lock_node"
        )

        # ==========================================================
        # Parameters
        # ==========================================================

        self.declare_parameter(
            "base_frame",
            "base_link",
        )

        self.declare_parameter(
            "window_size",
            20,
        )

        self.declare_parameter(
            "case_position_std_threshold_m",
            0.006,
        )
        self.declare_parameter(
            "case_orientation_rms_threshold_deg",
            5.0,
        )

        self.declare_parameter(
            "palm_position_std_threshold_m",
            0.010,
        )
        self.declare_parameter(
            "palm_orientation_rms_threshold_deg",
            10.0,
        )

        self.declare_parameter(
            "min_case_z_m",
            -0.10,
        )
        self.declare_parameter(
            "max_case_z_m",
            1.20,
        )
        self.declare_parameter(
            "min_palm_z_m",
            -0.10,
        )
        self.declare_parameter(
            "max_palm_z_m",
            1.50,
        )

        # ==========================================================
        # Read parameters
        # ==========================================================

        self.base_frame = str(
            self.get_parameter(
                "base_frame"
            ).value
        )

        self.window_size = int(
            self.get_parameter(
                "window_size"
            ).value
        )

        self.case_pos_std_threshold = float(
            self.get_parameter(
                "case_position_std_threshold_m"
            ).value
        )
        self.case_angle_threshold = float(
            self.get_parameter(
                "case_orientation_rms_threshold_deg"
            ).value
        )

        self.palm_pos_std_threshold = float(
            self.get_parameter(
                "palm_position_std_threshold_m"
            ).value
        )
        self.palm_angle_threshold = float(
            self.get_parameter(
                "palm_orientation_rms_threshold_deg"
            ).value
        )

        self.min_case_z = float(
            self.get_parameter(
                "min_case_z_m"
            ).value
        )
        self.max_case_z = float(
            self.get_parameter(
                "max_case_z_m"
            ).value
        )

        self.min_palm_z = float(
            self.get_parameter(
                "min_palm_z_m"
            ).value
        )
        self.max_palm_z = float(
            self.get_parameter(
                "max_palm_z_m"
            ).value
        )

        # ==========================================================
        # State
        # ==========================================================

        self.case_window = PoseWindow(
            self.window_size
        )
        self.palm_window = PoseWindow(
            self.window_size
        )

        self.case_locked = False
        self.palm_initial_locked = False
        self.palm_final_locked = False

        # Palm final must be explicitly armed AFTER observe-hand.
        self.palm_final_armed = False

        self.locked_case_pose = None
        self.locked_palm_initial_pose = None
        self.locked_palm_final_pose = None

        # ==========================================================
        # QoS
        #
        # Transient-local means a later planner can still receive the
        # most recently locked target after it starts.
        # ==========================================================

        locked_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ==========================================================
        # Publishers
        # ==========================================================

        self.case_locked_pose_pub = (
            self.create_publisher(
                PoseStamped,
                "/targets/earbud_case_pose_base",
                locked_qos,
            )
        )

        self.palm_initial_pose_pub = (
            self.create_publisher(
                PoseStamped,
                "/targets/palm_initial_pose_base",
                locked_qos,
            )
        )

        self.palm_final_pose_pub = (
            self.create_publisher(
                PoseStamped,
                "/targets/palm_final_pose_base",
                locked_qos,
            )
        )

        self.case_locked_pub = (
            self.create_publisher(
                Bool,
                "/targets/case_locked",
                locked_qos,
            )
        )

        self.palm_initial_locked_pub = (
            self.create_publisher(
                Bool,
                "/targets/palm_initial_locked",
                locked_qos,
            )
        )

        self.palm_final_locked_pub = (
            self.create_publisher(
                Bool,
                "/targets/palm_final_locked",
                locked_qos,
            )
        )

        self.start_targets_ready_pub = (
            self.create_publisher(
                Bool,
                "/targets/start_targets_ready",
                locked_qos,
            )
        )

        # ==========================================================
        # Subscribers
        # ==========================================================

        self.case_sub = (
            self.create_subscription(
                PoseStamped,
                "/targets/earbud_case_pose_base_live",
                self.case_callback,
                10,
            )
        )

        self.palm_sub = (
            self.create_subscription(
                PoseStamped,
                "/targets/palm_pose_base_live",
                self.palm_callback,
                10,
            )
        )

        # ==========================================================
        # Services
        # ==========================================================

        self.arm_palm_final_srv = (
            self.create_service(
                Trigger,
                "/targets/arm_palm_final",
                self.arm_palm_final_callback,
            )
        )

        self.reset_srv = (
            self.create_service(
                Trigger,
                "/targets/reset",
                self.reset_callback,
            )
        )

        self.publish_states()

        self.get_logger().info(
            "STEP 4 target lock node started."
        )

        self.get_logger().info(
            "Waiting for stable CASE + PALM INITIAL..."
        )

        self.get_logger().info(
            "NO robot motion."
        )

    # ==============================================================
    # Helpers
    # ==============================================================

    def valid_frame(self, msg):
        return (
            str(msg.header.frame_id)
            ==
            self.base_frame
        )

    @staticmethod
    def copy_locked_pose(
        source_msg,
        xyz,
        q,
        base_frame,
    ):
        out = PoseStamped()

        out.header.stamp = (
            source_msg.header.stamp
        )
        out.header.frame_id = (
            base_frame
        )

        out.pose.position.x = float(
            xyz[0]
        )
        out.pose.position.y = float(
            xyz[1]
        )
        out.pose.position.z = float(
            xyz[2]
        )

        out.pose.orientation.x = float(
            q[0]
        )
        out.pose.orientation.y = float(
            q[1]
        )
        out.pose.orientation.z = float(
            q[2]
        )
        out.pose.orientation.w = float(
            q[3]
        )

        return out

    @staticmethod
    def z_in_range(
        xyz,
        min_z,
        max_z,
    ):
        return (
            min_z
            <= float(xyz[2])
            <= max_z
        )

    def is_case_stable(
        self,
        stats,
    ):
        return (
            stats["max_position_std"]
            <= self.case_pos_std_threshold
            and
            stats["angle_rms_deg"]
            <= self.case_angle_threshold
        )

    def is_palm_stable(
        self,
        stats,
    ):
        return (
            stats["max_position_std"]
            <= self.palm_pos_std_threshold
            and
            stats["angle_rms_deg"]
            <= self.palm_angle_threshold
        )

    def publish_states(self):
        msg = Bool()

        msg.data = bool(
            self.case_locked
        )
        self.case_locked_pub.publish(
            msg
        )

        msg = Bool()
        msg.data = bool(
            self.palm_initial_locked
        )
        self.palm_initial_locked_pub.publish(
            msg
        )

        msg = Bool()
        msg.data = bool(
            self.palm_final_locked
        )
        self.palm_final_locked_pub.publish(
            msg
        )

        msg = Bool()
        msg.data = bool(
            self.case_locked
            and
            self.palm_initial_locked
        )
        self.start_targets_ready_pub.publish(
            msg
        )

    # ==============================================================
    # Case
    # ==============================================================

    def case_callback(
        self,
        msg,
    ):
        if self.case_locked:
            return

        if not self.valid_frame(
            msg
        ):
            self.get_logger().warning(
                "Ignoring case pose: "
                f"frame_id='{msg.header.frame_id}', "
                f"expected '{self.base_frame}'."
            )
            return

        if not self.case_window.append(
            msg
        ):
            return

        stats = (
            self.case_window.analyze()
        )

        if stats is None:
            return

        if not self.is_case_stable(
            stats
        ):
            return

        xyz = stats[
            "mean_position"
        ]
        q = stats[
            "mean_quaternion"
        ]

        if not self.z_in_range(
            xyz,
            self.min_case_z,
            self.max_case_z,
        ):
            self.get_logger().warning(
                "Stable case rejected: "
                f"Z={xyz[2]:.3f} m outside "
                f"[{self.min_case_z:.3f}, "
                f"{self.max_case_z:.3f}]"
            )
            self.case_window.clear()
            return

        self.locked_case_pose = (
            self.copy_locked_pose(
                stats["newest_msg"],
                xyz,
                q,
                self.base_frame,
            )
        )

        self.case_locked = True

        self.case_locked_pose_pub.publish(
            self.locked_case_pose
        )

        self.publish_states()

        self.get_logger().info(
            "CASE LOCKED | "
            f"XYZ=[{xyz[0]:+.4f}, "
            f"{xyz[1]:+.4f}, "
            f"{xyz[2]:+.4f}] m | "
            f"pos_std_max="
            f"{stats['max_position_std'] * 1000.0:.1f} mm | "
            f"ori_rms="
            f"{stats['angle_rms_deg']:.2f} deg"
        )

    # ==============================================================
    # Palm
    # ==============================================================

    def palm_callback(
        self,
        msg,
    ):
        # Before Palm Initial is locked:
        #   collect only initial-hand samples.
        #
        # After Palm Initial:
        #   ignore all palm data until /targets/arm_palm_final.
        #
        # After arm_palm_final:
        #   collect a fresh window and lock Palm Final.
        if (
            self.palm_initial_locked
            and
            not self.palm_final_armed
        ):
            return

        if self.palm_final_locked:
            return

        if not self.valid_frame(
            msg
        ):
            self.get_logger().warning(
                "Ignoring palm pose: "
                f"frame_id='{msg.header.frame_id}', "
                f"expected '{self.base_frame}'."
            )
            return

        if not self.palm_window.append(
            msg
        ):
            return

        stats = (
            self.palm_window.analyze()
        )

        if stats is None:
            return

        if not self.is_palm_stable(
            stats
        ):
            return

        xyz = stats[
            "mean_position"
        ]
        q = stats[
            "mean_quaternion"
        ]

        if not self.z_in_range(
            xyz,
            self.min_palm_z,
            self.max_palm_z,
        ):
            self.get_logger().warning(
                "Stable palm rejected: "
                f"Z={xyz[2]:.3f} m outside "
                f"[{self.min_palm_z:.3f}, "
                f"{self.max_palm_z:.3f}]"
            )
            self.palm_window.clear()
            return

        locked_pose = (
            self.copy_locked_pose(
                stats["newest_msg"],
                xyz,
                q,
                self.base_frame,
            )
        )

        if not self.palm_initial_locked:
            self.locked_palm_initial_pose = (
                locked_pose
            )

            self.palm_initial_locked = True

            self.palm_initial_pose_pub.publish(
                self.locked_palm_initial_pose
            )

            # Stop accumulating palm frames until final phase is armed.
            self.palm_window.clear()

            self.get_logger().info(
                "PALM INITIAL LOCKED | "
                f"XYZ=[{xyz[0]:+.4f}, "
                f"{xyz[1]:+.4f}, "
                f"{xyz[2]:+.4f}] m | "
                f"pos_std_max="
                f"{stats['max_position_std'] * 1000.0:.1f} mm | "
                f"ori_rms="
                f"{stats['angle_rms_deg']:.2f} deg"
            )

        elif self.palm_final_armed:
            self.locked_palm_final_pose = (
                locked_pose
            )

            self.palm_final_locked = True
            self.palm_final_armed = False

            self.palm_final_pose_pub.publish(
                self.locked_palm_final_pose
            )

            self.palm_window.clear()

            self.get_logger().info(
                "PALM FINAL LOCKED | "
                f"XYZ=[{xyz[0]:+.4f}, "
                f"{xyz[1]:+.4f}, "
                f"{xyz[2]:+.4f}] m | "
                f"pos_std_max="
                f"{stats['max_position_std'] * 1000.0:.1f} mm | "
                f"ori_rms="
                f"{stats['angle_rms_deg']:.2f} deg"
            )

        self.publish_states()

    # ==============================================================
    # Services
    # ==============================================================

    def arm_palm_final_callback(
        self,
        request,
        response,
    ):
        del request

        if not self.palm_initial_locked:
            response.success = False
            response.message = (
                "Palm Initial is not locked yet."
            )
            return response

        # A second call is allowed; it simply reacquires final palm.
        self.palm_final_locked = False
        self.locked_palm_final_pose = None

        self.palm_window.clear()
        self.palm_final_armed = True

        self.publish_states()

        response.success = True
        response.message = (
            "Palm Final armed. "
            "Old palm samples cleared; "
            "waiting for a fresh stable palm."
        )

        self.get_logger().info(
            "PALM FINAL ARMED | "
            "fresh palm window started."
        )

        return response

    def reset_callback(
        self,
        request,
        response,
    ):
        del request

        self.case_window.clear()
        self.palm_window.clear()

        self.case_locked = False
        self.palm_initial_locked = False
        self.palm_final_locked = False
        self.palm_final_armed = False

        self.locked_case_pose = None
        self.locked_palm_initial_pose = None
        self.locked_palm_final_pose = None

        self.publish_states()

        response.success = True
        response.message = (
            "All target locks cleared. "
            "Waiting for new stable Case + Palm Initial."
        )

        self.get_logger().info(
            "TARGET LOCK RESET"
        )

        return response


def main(args=None):
    rclpy.init(args=args)

    node = TargetLockNode()

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
