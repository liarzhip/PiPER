import math
import threading
import time

import numpy as np
import rclpy

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    MoveItErrorCodes,
    JointConstraint,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
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
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, SetBool
from tf2_ros import Buffer, TransformException, TransformListener


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


def pose_to_matrix(pose):
    q = quat_normalize(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
    )

    if q is None:
        return None

    R = quat_to_rotation_matrix(q)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [
        pose.position.x,
        pose.position.y,
        pose.position.z,
    ]
    return T


def transform_to_matrix(transform):
    q = quat_normalize(
        [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
    )

    if q is None:
        return None

    R = quat_to_rotation_matrix(q)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ]
    return T


def matrix_to_pose(T):
    pose = Pose()

    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])

    q = rotation_matrix_to_quaternion(
        T[:3, :3]
    )

    if q is None:
        return None

    pose.orientation.x = float(q[0])
    pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2])
    pose.orientation.w = float(q[3])

    return pose


class MoveItExecutorNode(Node):
    """
    STEP 6: MoveIt plan/visualize bridge.

    Input targets generated in STEP 5:
      /pick/pregrasp_pose
      /pick/grasp_pose
      /pick/lift_pose

    Services:
      /moveit/plan_pregrasp
      /moveit/plan_grasp
      /moveit/plan_lift
      /moveit/execute_last_plan
      /moveit/clear_plan

    STEP 6 default:
      allow_execute = False

    Therefore this node can PLAN and publish the trajectory to RViz,
    but refuses physical execution until explicitly enabled later.

    The target poses describe gripper_base, while the MoveIt 'arm'
    group normally solves IK for link6. This node converts:

        desired(base -> gripper_base)
                 ↓
        fixed TF(link6 -> gripper_base)
                 ↓
        desired(base -> link6)

    and sends the resulting link6 constraint to MoveIt.
    """

    def __init__(self):
        super().__init__(
            "moveit_executor_node"
        )

        self.cb_group = ReentrantCallbackGroup()
        self.operation_lock = threading.Lock()

        # ==========================================================
        # Parameters
        # ==========================================================

        self.declare_parameter(
            "base_frame",
            "base_link",
        )
        self.declare_parameter(
            "planning_group",
            "arm",
        )

        # STEP 5 publishes desired gripper_base poses.
        self.declare_parameter(
            "target_tool_frame",
            "gripper_base",
        )

        # The arm SRDF chain tip is link6.
        self.declare_parameter(
            "planning_link",
            "link6",
        )

        self.declare_parameter(
            "move_group_action",
            "/move_action",
        )
        self.declare_parameter(
            "execute_action",
            "/execute_trajectory",
        )

        self.declare_parameter(
            "allowed_planning_time",
            5.0,
        )
        self.declare_parameter(
            "num_planning_attempts",
            5,
        )

        self.declare_parameter(
            "velocity_scaling",
            0.15,
        )
        self.declare_parameter(
            "acceleration_scaling",
            0.10,
        )

        self.declare_parameter(
            "position_tolerance_m",
            0.006,
        )
        self.declare_parameter(
            "orientation_tolerance_rad",
            0.08,
        )

        self.declare_parameter(
            "planner_id",
            "",
        )
        self.declare_parameter(
            "pipeline_id",
            "",
        )

        self.declare_parameter(
            "action_server_timeout_sec",
            5.0,
        )
        self.declare_parameter(
            "plan_result_timeout_sec",
            20.0,
        )
        self.declare_parameter(
            "execute_result_timeout_sec",
            60.0,
        )
        self.declare_parameter(
            "tf_timeout_sec",
            0.5,
        )

        # CRITICAL for STEP 6.
        self.declare_parameter(
            "allow_execute",
            False,
        )

        # STEP 7B:
        # Dedicated, narrowly-scoped permission for ONLY PreGrasp.
        # Generic /moveit/execute_last_plan remains controlled by
        # allow_execute and should stay false.
        self.declare_parameter(
            "allow_pregrasp_execute",
            False,
        )

        self.declare_parameter(
            "require_table_ready",
            True,
        )
        self.declare_parameter(
            "require_grasp_plan_ready",
            True,
        )
        self.declare_parameter(
            "require_start_targets_ready",
            True,
        )

        self.declare_parameter(
            "stop_perception_before_motion",
            True,
        )
        self.declare_parameter(
            "perception_stop_service",
            "/perception/stop_observation",
        )

        self.declare_parameter(
            "post_execute_position_tolerance_m",
            0.020,
        )
        self.declare_parameter(
            "post_execute_orientation_tolerance_rad",
            0.18,
        )

        # After ExecuteTrajectory reports SUCCESS, the real PIPER feedback/TF
        # can still lag slightly behind the controller result.  Do not verify
        # only once immediately.  Instead, poll the REAL gripper_base TF for a
        # short settling window and accept only when the configured pose
        # tolerance is actually reached.
        self.declare_parameter(
            "post_execute_verify_timeout_sec",
            2.0,
        )
        self.declare_parameter(
            "post_execute_verify_period_sec",
            0.05,
        )

        # STEP 7C: controlled Cartesian descent from PreGrasp -> Grasp.
        self.declare_parameter(
            "allow_grasp_preview_execute",
            False,
        )
        self.declare_parameter(
            "compute_cartesian_path_service",
            "/compute_cartesian_path",
        )
        self.declare_parameter(
            "cartesian_max_step_m",
            0.005,
        )
        self.declare_parameter(
            "cartesian_jump_threshold",
            0.0,
        )
        self.declare_parameter(
            "cartesian_min_fraction",
            0.99,
        )

        # PreGrasp must already have been reached before descent.
        self.declare_parameter(
            "pregrasp_required_position_tolerance_m",
            0.030,
        )
        self.declare_parameter(
            "pregrasp_required_orientation_tolerance_rad",
            0.22,
        )

        # ----------------------------------------------------------
        # STEP 7C gripper control THROUGH MoveIt.
        #
        # IMPORTANT:
        # When MoveIt/ros2_control is running, it continuously owns
        # /control/joint_states.  The project must NOT publish a second
        # direct gripper command source to that topic.
        #
        # AgileX MoveIt provides planning group "gripper"; therefore
        # gripper opening is planned/executed through MoveGroup exactly
        # like the arm trajectory.
        # ----------------------------------------------------------

        self.declare_parameter(
            "gripper_planning_group",
            "gripper",
        )
        self.declare_parameter(
            "gripper_joint_name",
            "gripper",
        )
        self.declare_parameter(
            "gripper_open_width_m",
            0.070,
        )
        self.declare_parameter(
            "gripper_close_width_m",
            0.000,
        )
        self.declare_parameter(
            "gripper_position_tolerance_m",
            0.006,
        )
        self.declare_parameter(
            "gripper_goal_tolerance_m",
            0.002,
        )
        self.declare_parameter(
            "gripper_velocity_scaling",
            0.30,
        )
        self.declare_parameter(
            "gripper_acceleration_scaling",
            0.20,
        )

        # Separate execution gate for the gripper.
        # Keep false for the first dry-run / planning test.
        self.declare_parameter(
            "allow_gripper_execute",
            False,
        )

        # The official AgileX auto Gate monitors the ARM controller action.
        # A pure MoveIt "gripper" trajectory may therefore complete inside
        # ros2_control while the physical /control/* gate is still closed.
        #
        # For gripper-only execution, this node automatically opens the same
        # /control_enable service immediately before ExecuteTrajectory and
        # closes it again in a finally block.
        self.declare_parameter(
            "gripper_auto_gate",
            True,
        )
        self.declare_parameter(
            "control_gate_service",
            "/control_enable",
        )
        self.declare_parameter(
            "control_gate_timeout_sec",
            5.0,
        )

        # ----------------------------------------------------------
        # STEP 7D: real object grasp -> Lift -> Observe Hand
        # ----------------------------------------------------------

        # Distinct from empty-gripper CLOSE verification.
        self.declare_parameter(
            "allow_object_grasp_execute",
            False,
        )
        self.declare_parameter(
            "object_grasp_min_width_m",
            0.008,
        )
        self.declare_parameter(
            "object_grasp_max_width_m",
            0.065,
        )
        self.declare_parameter(
            "object_grasp_min_closure_m",
            0.005,
        )
        self.declare_parameter(
            "object_grasp_settle_sec",
            0.35,
        )

        self.declare_parameter(
            "allow_lift_execute",
            False,
        )
        self.declare_parameter(
            "require_object_grasped_for_lift",
            True,
        )
        self.declare_parameter(
            "lift_required_position_tolerance_m",
            0.030,
        )
        self.declare_parameter(
            "lift_required_orientation_tolerance_rad",
            0.22,
        )

        # Give the real PIPER/controller state time to fully settle after
        # Cartesian Lift before planning the Observe-Hand motion.  This avoids
        # MoveIt's "Invalid Trajectory: start point deviates from current robot
        # state" execution rejection when Observe-Hand is planned immediately
        # after Lift.
        self.declare_parameter(
            "post_lift_settle_sec",
            1.0,
        )

        self.declare_parameter(
            "allow_observe_hand_execute",
            False,
        )
        self.declare_parameter(
            "require_lift_before_observe",
            True,
        )
        self.declare_parameter(
            "observe_required_position_tolerance_m",
            0.035,
        )
        self.declare_parameter(
            "observe_required_orientation_tolerance_rad",
            0.25,
        )
        self.declare_parameter(
            "observe_settle_sec",
            0.50,
        )

        self.declare_parameter(
            "perception_start_service",
            "/perception/start_observation",
        )
        self.declare_parameter(
            "arm_palm_final_service",
            "/targets/arm_palm_final",
        )

        # ----------------------------------------------------------
        # STEP 7E: Palm Final -> Handover Approach -> Handover Final
        # ----------------------------------------------------------
        self.declare_parameter(
            "allow_handover_approach_execute",
            False,
        )
        self.declare_parameter(
            "allow_handover_final_execute",
            False,
        )
        self.declare_parameter(
            "handover_approach_required_position_tolerance_m",
            0.035,
        )
        self.declare_parameter(
            "handover_approach_required_orientation_tolerance_rad",
            0.25,
        )
        self.declare_parameter(
            "handover_final_required_position_tolerance_m",
            0.030,
        )
        self.declare_parameter(
            "handover_final_required_orientation_tolerance_rad",
            0.22,
        )

        # ==========================================================
        # Read parameters
        # ==========================================================

        self.base_frame = str(
            self.get_parameter(
                "base_frame"
            ).value
        )
        self.planning_group = str(
            self.get_parameter(
                "planning_group"
            ).value
        )
        self.target_tool_frame = str(
            self.get_parameter(
                "target_tool_frame"
            ).value
        )
        self.planning_link = str(
            self.get_parameter(
                "planning_link"
            ).value
        )

        self.move_group_action = str(
            self.get_parameter(
                "move_group_action"
            ).value
        )
        self.execute_action = str(
            self.get_parameter(
                "execute_action"
            ).value
        )

        self.allowed_planning_time = float(
            self.get_parameter(
                "allowed_planning_time"
            ).value
        )
        self.num_planning_attempts = int(
            self.get_parameter(
                "num_planning_attempts"
            ).value
        )

        self.velocity_scaling = float(
            self.get_parameter(
                "velocity_scaling"
            ).value
        )
        self.acceleration_scaling = float(
            self.get_parameter(
                "acceleration_scaling"
            ).value
        )

        self.position_tolerance = float(
            self.get_parameter(
                "position_tolerance_m"
            ).value
        )
        self.orientation_tolerance = float(
            self.get_parameter(
                "orientation_tolerance_rad"
            ).value
        )

        self.planner_id = str(
            self.get_parameter(
                "planner_id"
            ).value
        )
        self.pipeline_id = str(
            self.get_parameter(
                "pipeline_id"
            ).value
        )

        self.action_server_timeout = float(
            self.get_parameter(
                "action_server_timeout_sec"
            ).value
        )
        self.plan_result_timeout = float(
            self.get_parameter(
                "plan_result_timeout_sec"
            ).value
        )
        self.execute_result_timeout = float(
            self.get_parameter(
                "execute_result_timeout_sec"
            ).value
        )
        self.tf_timeout = float(
            self.get_parameter(
                "tf_timeout_sec"
            ).value
        )

        self.allow_execute = bool(
            self.get_parameter(
                "allow_execute"
            ).value
        )

        self.allow_pregrasp_execute = bool(
            self.get_parameter(
                "allow_pregrasp_execute"
            ).value
        )

        self.require_table_ready = bool(
            self.get_parameter(
                "require_table_ready"
            ).value
        )
        self.require_grasp_plan_ready = bool(
            self.get_parameter(
                "require_grasp_plan_ready"
            ).value
        )
        self.require_start_targets_ready = bool(
            self.get_parameter(
                "require_start_targets_ready"
            ).value
        )

        self.stop_perception_before_motion = bool(
            self.get_parameter(
                "stop_perception_before_motion"
            ).value
        )
        self.perception_stop_service = str(
            self.get_parameter(
                "perception_stop_service"
            ).value
        )

        self.post_execute_position_tolerance = float(
            self.get_parameter(
                "post_execute_position_tolerance_m"
            ).value
        )
        self.post_execute_orientation_tolerance = float(
            self.get_parameter(
                "post_execute_orientation_tolerance_rad"
            ).value
        )
        self.post_execute_verify_timeout = float(
            self.get_parameter(
                "post_execute_verify_timeout_sec"
            ).value
        )
        self.post_execute_verify_period = float(
            self.get_parameter(
                "post_execute_verify_period_sec"
            ).value
        )

        self.allow_grasp_preview_execute = bool(
            self.get_parameter(
                "allow_grasp_preview_execute"
            ).value
        )
        self.compute_cartesian_path_service = str(
            self.get_parameter(
                "compute_cartesian_path_service"
            ).value
        )
        self.cartesian_max_step = float(
            self.get_parameter(
                "cartesian_max_step_m"
            ).value
        )
        self.cartesian_jump_threshold = float(
            self.get_parameter(
                "cartesian_jump_threshold"
            ).value
        )
        self.cartesian_min_fraction = float(
            self.get_parameter(
                "cartesian_min_fraction"
            ).value
        )

        self.pregrasp_required_position_tolerance = float(
            self.get_parameter(
                "pregrasp_required_position_tolerance_m"
            ).value
        )
        self.pregrasp_required_orientation_tolerance = float(
            self.get_parameter(
                "pregrasp_required_orientation_tolerance_rad"
            ).value
        )

        self.gripper_planning_group = str(
            self.get_parameter(
                "gripper_planning_group"
            ).value
        )
        self.gripper_joint_name = str(
            self.get_parameter(
                "gripper_joint_name"
            ).value
        )
        self.gripper_open_width = float(
            self.get_parameter(
                "gripper_open_width_m"
            ).value
        )
        self.gripper_close_width = float(
            self.get_parameter(
                "gripper_close_width_m"
            ).value
        )
        self.gripper_position_tolerance = float(
            self.get_parameter(
                "gripper_position_tolerance_m"
            ).value
        )
        self.gripper_goal_tolerance = float(
            self.get_parameter(
                "gripper_goal_tolerance_m"
            ).value
        )
        self.gripper_velocity_scaling = float(
            self.get_parameter(
                "gripper_velocity_scaling"
            ).value
        )
        self.gripper_acceleration_scaling = float(
            self.get_parameter(
                "gripper_acceleration_scaling"
            ).value
        )
        self.allow_gripper_execute = bool(
            self.get_parameter(
                "allow_gripper_execute"
            ).value
        )

        self.gripper_auto_gate = bool(
            self.get_parameter(
                "gripper_auto_gate"
            ).value
        )
        self.control_gate_service = str(
            self.get_parameter(
                "control_gate_service"
            ).value
        )
        self.control_gate_timeout = float(
            self.get_parameter(
                "control_gate_timeout_sec"
            ).value
        )

        self.allow_object_grasp_execute = bool(
            self.get_parameter(
                "allow_object_grasp_execute"
            ).value
        )
        self.object_grasp_min_width = float(
            self.get_parameter(
                "object_grasp_min_width_m"
            ).value
        )
        self.object_grasp_max_width = float(
            self.get_parameter(
                "object_grasp_max_width_m"
            ).value
        )
        self.object_grasp_min_closure = float(
            self.get_parameter(
                "object_grasp_min_closure_m"
            ).value
        )
        self.object_grasp_settle_sec = float(
            self.get_parameter(
                "object_grasp_settle_sec"
            ).value
        )

        self.allow_lift_execute = bool(
            self.get_parameter(
                "allow_lift_execute"
            ).value
        )
        self.require_object_grasped_for_lift = bool(
            self.get_parameter(
                "require_object_grasped_for_lift"
            ).value
        )
        self.lift_required_position_tolerance = float(
            self.get_parameter(
                "lift_required_position_tolerance_m"
            ).value
        )
        self.lift_required_orientation_tolerance = float(
            self.get_parameter(
                "lift_required_orientation_tolerance_rad"
            ).value
        )

        self.post_lift_settle_sec = float(
            self.get_parameter(
                "post_lift_settle_sec"
            ).value
        )

        self.allow_observe_hand_execute = bool(
            self.get_parameter(
                "allow_observe_hand_execute"
            ).value
        )
        self.require_lift_before_observe = bool(
            self.get_parameter(
                "require_lift_before_observe"
            ).value
        )
        self.observe_required_position_tolerance = float(
            self.get_parameter(
                "observe_required_position_tolerance_m"
            ).value
        )
        self.observe_required_orientation_tolerance = float(
            self.get_parameter(
                "observe_required_orientation_tolerance_rad"
            ).value
        )
        self.observe_settle_sec = float(
            self.get_parameter(
                "observe_settle_sec"
            ).value
        )

        self.perception_start_service = str(
            self.get_parameter(
                "perception_start_service"
            ).value
        )
        self.arm_palm_final_service = str(
            self.get_parameter(
                "arm_palm_final_service"
            ).value
        )

        self.allow_handover_approach_execute = bool(
            self.get_parameter(
                "allow_handover_approach_execute"
            ).value
        )
        self.allow_handover_final_execute = bool(
            self.get_parameter(
                "allow_handover_final_execute"
            ).value
        )
        self.handover_approach_required_position_tolerance = float(
            self.get_parameter(
                "handover_approach_required_position_tolerance_m"
            ).value
        )
        self.handover_approach_required_orientation_tolerance = float(
            self.get_parameter(
                "handover_approach_required_orientation_tolerance_rad"
            ).value
        )
        self.handover_final_required_position_tolerance = float(
            self.get_parameter(
                "handover_final_required_position_tolerance_m"
            ).value
        )
        self.handover_final_required_orientation_tolerance = float(
            self.get_parameter(
                "handover_final_required_orientation_tolerance_rad"
            ).value
        )

        if not (0.0 <= self.gripper_open_width <= 0.1):
            raise ValueError(
                "gripper_open_width_m must be within [0.0, 0.1]"
            )

        if not (0.0 <= self.gripper_close_width <= 0.1):
            raise ValueError(
                "gripper_close_width_m must be within [0.0, 0.1]"
            )

        if self.gripper_close_width >= self.gripper_open_width:
            raise ValueError(
                "gripper_close_width_m must be smaller than "
                "gripper_open_width_m"
            )

        if not (
            0.0
            <= self.object_grasp_min_width
            < self.object_grasp_max_width
            <= 0.1
        ):
            raise ValueError(
                "object grasp width range must satisfy "
                "0 <= min < max <= 0.1 m"
            )

        # ==========================================================
        # TF
        # ==========================================================

        self.tf_buffer = Buffer(
            cache_time=Duration(
                seconds=10.0
            )
        )
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        # ==========================================================
        # QoS
        # ==========================================================

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ==========================================================
        # Target subscriptions
        # ==========================================================

        self.targets = {
            "pregrasp": None,
            "grasp": None,
            "lift": None,
            "observe_hand": None,
            "handover_approach": None,
            "handover_final": None,
        }

        self.create_subscription(
            PoseStamped,
            "/pick/pregrasp_pose",
            lambda msg: self._target_cb(
                "pregrasp", msg
            ),
            latched_qos,
            callback_group=self.cb_group,
        )

        self.create_subscription(
            PoseStamped,
            "/pick/grasp_pose",
            lambda msg: self._target_cb(
                "grasp", msg
            ),
            latched_qos,
            callback_group=self.cb_group,
        )

        self.create_subscription(
            PoseStamped,
            "/pick/lift_pose",
            lambda msg: self._target_cb(
                "lift", msg
            ),
            latched_qos,
            callback_group=self.cb_group,
        )

        self.create_subscription(
            PoseStamped,
            "/handover/observe_hand_pose",
            lambda msg: self._target_cb(
                "observe_hand", msg
            ),
            latched_qos,
            callback_group=self.cb_group,
        )

        self.create_subscription(
            PoseStamped,
            "/handover/approach_pose",
            lambda msg: self._target_cb(
                "handover_approach", msg
            ),
            latched_qos,
            callback_group=self.cb_group,
        )

        self.create_subscription(
            PoseStamped,
            "/handover/final_pose",
            lambda msg: self._target_cb(
                "handover_final", msg
            ),
            latched_qos,
            callback_group=self.cb_group,
        )

        self.palm_final_locked = False
        self.create_subscription(
            Bool,
            "/targets/palm_final_locked",
            self._palm_final_locked_cb,
            latched_qos,
            callback_group=self.cb_group,
        )

        # ==========================================================
        # STEP 7B execution interlocks
        # ==========================================================

        self.table_ready = False
        self.grasp_plan_ready = False
        self.start_targets_ready = False

        self.create_subscription(
            Bool,
            "/scene/table_ready",
            self._table_ready_cb,
            10,
            callback_group=self.cb_group,
        )

        self.create_subscription(
            Bool,
            "/pick/grasp_plan_ready",
            self._grasp_plan_ready_cb,
            latched_qos,
            callback_group=self.cb_group,
        )

        self.create_subscription(
            Bool,
            "/targets/start_targets_ready",
            self._start_targets_ready_cb,
            latched_qos,
            callback_group=self.cb_group,
        )

        # ==========================================================
        # MoveIt action clients
        # ==========================================================

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            self.move_group_action,
            callback_group=self.cb_group,
        )

        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            self.execute_action,
            callback_group=self.cb_group,
        )

        self.control_gate_client = self.create_client(
            SetBool,
            self.control_gate_service,
            callback_group=self.cb_group,
        )

        self.perception_stop_client = self.create_client(
            Trigger,
            self.perception_stop_service,
            callback_group=self.cb_group,
        )

        self.perception_start_client = self.create_client(
            Trigger,
            self.perception_start_service,
            callback_group=self.cb_group,
        )

        self.arm_palm_final_client = self.create_client(
            Trigger,
            self.arm_palm_final_service,
            callback_group=self.cb_group,
        )

        self.cartesian_path_client = self.create_client(
            GetCartesianPath,
            self.compute_cartesian_path_service,
            callback_group=self.cb_group,
        )

        # Real hardware feedback.  We read the externally exposed
        # "gripper" joint from AgileX /feedback/joint_states.
        self.current_gripper_width = None
        self.gripper_feedback_lock = threading.Lock()

        self.create_subscription(
            JointState,
            "/feedback/joint_states",
            self._joint_feedback_cb,
            20,
            callback_group=self.cb_group,
        )

        # ==========================================================
        # RViz / status
        # ==========================================================

        self.display_pub = self.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            10,
        )

        self.plan_valid_pub = self.create_publisher(
            Bool,
            "/moveit/last_plan_valid",
            latched_qos,
        )

        self.plan_target_pub = self.create_publisher(
            String,
            "/moveit/last_plan_target",
            latched_qos,
        )

        self.object_grasped = False
        self.object_grasped_pub = self.create_publisher(
            Bool,
            "/pick/object_grasped",
            latched_qos,
        )
        self._publish_object_grasped(False)

        # ==========================================================
        # Services
        # ==========================================================

        self.create_service(
            Trigger,
            "/moveit/plan_pregrasp",
            lambda req, res: self._plan_service(
                "pregrasp",
                req,
                res,
            ),
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/plan_grasp",
            lambda req, res: self._plan_service(
                "grasp",
                req,
                res,
            ),
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/plan_lift",
            lambda req, res: self._plan_service(
                "lift",
                req,
                res,
            ),
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/execute_last_plan",
            self._execute_service,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/execute_pregrasp",
            self._execute_pregrasp_service,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/execute_grasp_preview",
            self._execute_grasp_preview_service,
            callback_group=self.cb_group,
        )

        # STEP 7C diagnostic services.
        # These also go through MoveIt; they NEVER publish directly to
        # /control/joint_states.
        self.create_service(
            Trigger,
            "/moveit/plan_gripper_open",
            self._plan_gripper_open_service,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/execute_gripper_open",
            self._execute_gripper_open_service,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/plan_gripper_close",
            self._plan_gripper_close_service,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/execute_gripper_close",
            self._execute_gripper_close_service,
            callback_group=self.cb_group,
        )

        # STEP 7D: object-aware grasp. Unlike empty-close, success does
        # NOT require the fingers to reach 0.000 m.
        self.create_service(
            Trigger,
            "/moveit/execute_gripper_grasp",
            self._execute_gripper_grasp_service,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/execute_lift",
            self._execute_lift_service,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/plan_observe_hand",
            lambda req, res: self._plan_service(
                "observe_hand",
                req,
                res,
            ),
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/execute_observe_hand",
            self._execute_observe_hand_service,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/plan_handover_approach",
            lambda req, res: self._plan_service(
                "handover_approach",
                req,
                res,
            ),
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/execute_handover_approach",
            self._execute_handover_approach_service,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/plan_handover_final",
            self._plan_handover_final_service,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/execute_handover_final",
            self._execute_handover_final_service,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/moveit/clear_plan",
            self._clear_service,
            callback_group=self.cb_group,
        )

        # ==========================================================
        # Last plan
        # ==========================================================

        self.last_trajectory = None
        self.last_trajectory_start = None
        self.last_plan_target = ""

        self._publish_plan_state(
            False,
            "",
        )

        self.get_logger().info(
            "STEP 6 MoveIt executor started."
        )
        self.get_logger().info(
            f"Planning group = {self.planning_group}"
        )
        self.get_logger().info(
            f"Target tool = {self.target_tool_frame}"
        )
        self.get_logger().info(
            f"MoveIt planning link = {self.planning_link}"
        )
        self.get_logger().info(
            f"allow_execute = {self.allow_execute}"
        )
        self.get_logger().info(
            "STEP 7B dedicated PreGrasp execution = "
            f"{self.allow_pregrasp_execute}"
        )

        if not self.allow_execute:
            self.get_logger().warning(
                "Generic execute_last_plan remains DISABLED "
                "(allow_execute=false)."
            )

        if not self.allow_pregrasp_execute:
            self.get_logger().warning(
                "Dedicated /moveit/execute_pregrasp is currently BLOCKED "
                "(allow_pregrasp_execute=false)."
            )

        self.get_logger().info(
            "STEP 7C dedicated Grasp-preview execution = "
            f"{self.allow_grasp_preview_execute}"
        )

        if not self.allow_grasp_preview_execute:
            self.get_logger().warning(
                "Dedicated /moveit/execute_grasp_preview is BLOCKED "
                "(allow_grasp_preview_execute=false)."
            )

        self.get_logger().info(
            "STEP 7C gripper control source = MoveIt group "
            f"'{self.gripper_planning_group}' "
            f"(joint='{self.gripper_joint_name}')"
        )
        self.get_logger().info(
            "NO direct project publisher to /control/joint_states."
        )
        self.get_logger().info(
            "Gripper open width = "
            f"{self.gripper_open_width:.3f} m"
        )
        self.get_logger().info(
            "Gripper close width = "
            f"{self.gripper_close_width:.3f} m"
        )

        if not self.allow_gripper_execute:
            self.get_logger().warning(
                "MoveIt gripper execution is BLOCKED "
                "(allow_gripper_execute=false). "
                "Planning / RViz preview is still available."
            )

        if self.gripper_auto_gate:
            self.get_logger().info(
                "Gripper-only MoveIt execution uses automatic project Gate: "
                f"{self.control_gate_service} OPEN -> execute -> CLOSE."
            )

        self.get_logger().info(
            "STEP 7D object-grasp execution = "
            f"{self.allow_object_grasp_execute}"
        )
        self.get_logger().info(
            "STEP 7D Lift execution = "
            f"{self.allow_lift_execute}"
        )
        self.get_logger().info(
            "STEP 7D Observe-Hand execution = "
            f"{self.allow_observe_hand_execute}"
        )

        self.get_logger().info(
            "STEP 7E Handover-Approach execution = "
            f"{self.allow_handover_approach_execute}"
        )
        self.get_logger().info(
            "STEP 7E Handover-Final execution = "
            f"{self.allow_handover_final_execute}"
        )

    # ==============================================================
    # Helpers
    # ==============================================================

    def _table_ready_cb(
        self,
        msg,
    ):
        self.table_ready = bool(
            msg.data
        )

    def _grasp_plan_ready_cb(
        self,
        msg,
    ):
        self.grasp_plan_ready = bool(
            msg.data
        )

    def _start_targets_ready_cb(
        self,
        msg,
    ):
        self.start_targets_ready = bool(
            msg.data
        )

    def _target_cb(
        self,
        name,
        msg,
    ):
        self.targets[name] = msg

    def _palm_final_locked_cb(
        self,
        msg,
    ):
        self.palm_final_locked = bool(
            msg.data
        )

    def _joint_feedback_cb(
        self,
        msg,
    ):
        try:
            index = list(
                msg.name
            ).index(
                self.gripper_joint_name
            )
        except ValueError:
            return

        if index >= len(msg.position):
            return

        with self.gripper_feedback_lock:
            self.current_gripper_width = float(
                msg.position[index]
            )

    def _get_gripper_width(self):
        with self.gripper_feedback_lock:
            return self.current_gripper_width


    def _publish_object_grasped(
        self,
        value,
    ):
        self.object_grasped = bool(value)

        msg = Bool()
        msg.data = self.object_grasped
        self.object_grasped_pub.publish(
            msg
        )

    def _call_trigger_service(
        self,
        client,
        service_name,
    ):
        if not client.wait_for_service(
            timeout_sec=self.action_server_timeout
        ):
            return (
                False,
                f"service unavailable: {service_name}",
            )

        result = self._wait_future(
            client.call_async(
                Trigger.Request()
            ),
            self.action_server_timeout,
        )

        if result is None:
            return (
                False,
                f"service timed out: {service_name}",
            )

        return (
            bool(result.success),
            str(result.message),
        )

    def _verify_gripper_width(
        self,
        target_width,
    ):
        current = self._get_gripper_width()

        if current is None:
            return (
                False,
                "no gripper feedback in /feedback/joint_states",
            )

        error = abs(
            float(current)
            - float(target_width)
        )

        ok = (
            error
            <= self.gripper_position_tolerance
        )

        return (
            ok,
            (
                f"target={target_width:.4f} m, "
                f"feedback={current:.4f} m, "
                f"error={error * 1000.0:.1f} mm"
            ),
        )

    def _check_pregrasp_interlocks(self):
        problems = []

        if self.targets.get("pregrasp") is None:
            problems.append(
                "no pregrasp target"
            )

        if (
            self.require_table_ready
            and
            not self.table_ready
        ):
            problems.append(
                "table_ready=false"
            )

        if (
            self.require_grasp_plan_ready
            and
            not self.grasp_plan_ready
        ):
            problems.append(
                "grasp_plan_ready=false"
            )

        if (
            self.require_start_targets_ready
            and
            not self.start_targets_ready
        ):
            problems.append(
                "start_targets_ready=false"
            )

        return problems

    def _stop_perception_for_motion(self):
        if not self.stop_perception_before_motion:
            return (
                True,
                "perception stop disabled by config"
            )

        if not self.perception_stop_client.wait_for_service(
            timeout_sec=self.action_server_timeout
        ):
            return (
                False,
                "perception stop service unavailable"
            )

        future = self.perception_stop_client.call_async(
            Trigger.Request()
        )

        result = self._wait_future(
            future,
            self.action_server_timeout,
        )

        if result is None:
            return (
                False,
                "perception stop service timed out"
            )

        if not result.success:
            return (
                False,
                "perception stop service rejected request: "
                f"{result.message}"
            )

        return (
            True,
            result.message
        )

    def _set_control_gate(
        self,
        enabled,
        reason,
    ):
        if not self.control_gate_client.wait_for_service(
            timeout_sec=self.control_gate_timeout
        ):
            return (
                False,
                "control gate service unavailable: "
                f"{self.control_gate_service}",
            )

        request = SetBool.Request()
        request.data = bool(enabled)

        future = self.control_gate_client.call_async(
            request
        )

        result = self._wait_future(
            future,
            self.control_gate_timeout,
        )

        if result is None:
            return (
                False,
                "control gate service timed out",
            )

        if not result.success:
            return (
                False,
                "control gate service rejected request: "
                f"{result.message}",
            )

        state = "OPEN" if enabled else "CLOSED"
        self.get_logger().info(
            f"CONTROL GATE {state} | {reason}"
        )

        return (
            True,
            str(result.message),
        )

    def _execute_trajectory_internal(
        self,
        trajectory,
        label,
    ):
        if not self.execute_client.wait_for_server(
            timeout_sec=self.action_server_timeout
        ):
            return (
                False,
                "ExecuteTrajectory action "
                f"'{self.execute_action}' is unavailable."
            )

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory

        self.get_logger().warning(
            f"EXECUTING REAL ROBOT TRAJECTORY: {label}"
        )

        send_future = (
            self.execute_client.send_goal_async(
                goal
            )
        )

        goal_handle = self._wait_future(
            send_future,
            self.action_server_timeout,
        )

        if (
            goal_handle is None
            or
            not goal_handle.accepted
        ):
            return (
                False,
                "ExecuteTrajectory goal rejected."
            )

        result_future = (
            goal_handle.get_result_async()
        )

        wrapped_result = self._wait_future(
            result_future,
            self.execute_result_timeout,
        )

        if wrapped_result is None:
            return (
                False,
                "Trajectory execution timed out."
            )

        result = wrapped_result.result
        error_code = int(
            result.error_code.val
        )

        if error_code != MoveItErrorCodes.SUCCESS:
            return (
                False,
                "Trajectory execution failed: "
                f"MoveIt error_code={error_code}"
            )

        return (
            True,
            "trajectory execution succeeded"
        )

    def _current_tool_pose(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.target_tool_frame,
                Time(),
                timeout=Duration(
                    seconds=self.tf_timeout
                ),
            )
        except TransformException:
            return None

        p = np.array(
            [
                tf_msg.transform.translation.x,
                tf_msg.transform.translation.y,
                tf_msg.transform.translation.z,
            ],
            dtype=np.float64,
        )

        q = quat_normalize(
            [
                tf_msg.transform.rotation.x,
                tf_msg.transform.rotation.y,
                tf_msg.transform.rotation.z,
                tf_msg.transform.rotation.w,
            ]
        )

        if q is None:
            return None

        return (
            p,
            q,
        )

    def _verify_tool_target(
        self,
        target,
        position_tolerance=None,
        orientation_tolerance=None,
    ):
        current = self._current_tool_pose()

        if current is None:
            return (
                False,
                "cannot read current tool TF after execution"
            )

        current_p, current_q = current

        target_p = np.array(
            [
                target.pose.position.x,
                target.pose.position.y,
                target.pose.position.z,
            ],
            dtype=np.float64,
        )

        target_q = quat_normalize(
            [
                target.pose.orientation.x,
                target.pose.orientation.y,
                target.pose.orientation.z,
                target.pose.orientation.w,
            ]
        )

        if target_q is None:
            return (
                False,
                "invalid target quaternion"
            )

        position_error = float(
            np.linalg.norm(
                current_p
                - target_p
            )
        )

        dot = abs(
            float(
                np.dot(
                    current_q,
                    target_q,
                )
            )
        )
        dot = max(
            -1.0,
            min(
                1.0,
                dot,
            ),
        )

        orientation_error = float(
            2.0
            * math.acos(dot)
        )

        if position_tolerance is None:
            position_tolerance = (
                self.post_execute_position_tolerance
            )

        if orientation_tolerance is None:
            orientation_tolerance = (
                self.post_execute_orientation_tolerance
            )

        ok = (
            position_error
            <= float(position_tolerance)
            and
            orientation_error
            <= float(orientation_tolerance)
        )

        return (
            ok,
            (
                f"position_error={position_error * 1000.0:.1f} mm, "
                f"orientation_error={math.degrees(orientation_error):.1f} deg"
            ),
        )

    def _wait_for_tool_target(
        self,
        target,
        position_tolerance=None,
        orientation_tolerance=None,
        timeout_sec=None,
        period_sec=None,
    ):
        """
        Verify the REAL base_link -> target_tool_frame pose repeatedly.

        ExecuteTrajectory SUCCESS means the controller action has completed,
        but real hardware feedback / TF may still take a short time to settle.
        This helper avoids a false task abort caused by one immediate TF sample.

        It does NOT relax the configured pose tolerance.  The robot must still
        enter the requested tolerance before this function returns success.
        """

        if timeout_sec is None:
            timeout_sec = self.post_execute_verify_timeout

        if period_sec is None:
            period_sec = self.post_execute_verify_period

        timeout_sec = max(
            0.0,
            float(timeout_sec),
        )
        period_sec = max(
            0.01,
            float(period_sec),
        )

        deadline = (
            time.monotonic()
            + timeout_sec
        )

        attempts = 0
        last_detail = "no TF sample"

        while rclpy.ok():
            attempts += 1

            (
                ok,
                detail,
            ) = self._verify_tool_target(
                target,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            )

            last_detail = detail

            if ok:
                return (
                    True,
                    (
                        detail
                        + f" | settled_after={attempts} samples"
                    ),
                )

            if time.monotonic() >= deadline:
                break

            time.sleep(period_sec)

        return (
            False,
            (
                last_detail
                + f" | settle_timeout={timeout_sec:.2f}s"
                + f" | samples={attempts}"
            ),
        )

    def _publish_plan_state(
        self,
        valid,
        target_name,
    ):
        msg = Bool()
        msg.data = bool(valid)
        self.plan_valid_pub.publish(msg)

        text = String()
        text.data = str(target_name)
        self.plan_target_pub.publish(text)

    def _clear_last_plan(self):
        self.last_trajectory = None
        self.last_trajectory_start = None
        self.last_plan_target = ""

        self._publish_plan_state(
            False,
            "",
        )

    @staticmethod
    def _wait_future(
        future,
        timeout_sec,
    ):
        event = threading.Event()

        future.add_done_callback(
            lambda _future: event.set()
        )

        if not event.wait(
            timeout=max(
                0.01,
                float(timeout_sec),
            )
        ):
            return None

        try:
            return future.result()
        except Exception:
            return None

    def _desired_tool_to_planning_link(
        self,
        tool_pose_msg,
    ):
        """
        Convert desired base->target_tool_frame into
        desired base->planning_link.
        """
        if (
            str(tool_pose_msg.header.frame_id)
            != self.base_frame
        ):
            raise RuntimeError(
                "Target pose frame is "
                f"'{tool_pose_msg.header.frame_id}', "
                f"expected '{self.base_frame}'."
            )

        T_base_tool = pose_to_matrix(
            tool_pose_msg.pose
        )

        if T_base_tool is None:
            raise RuntimeError(
                "Invalid target quaternion."
            )

        if (
            self.planning_link
            ==
            self.target_tool_frame
        ):
            planning_pose = (
                matrix_to_pose(
                    T_base_tool
                )
            )

            if planning_pose is None:
                raise RuntimeError(
                    "Failed to convert target pose."
                )

            return planning_pose

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.planning_link,
                self.target_tool_frame,
                Time(),
                timeout=Duration(
                    seconds=self.tf_timeout
                ),
            )

        except TransformException as exc:
            raise RuntimeError(
                "Cannot get fixed tool transform "
                f"{self.planning_link} <- "
                f"{self.target_tool_frame}: {exc}"
            ) from exc

        # lookup_transform(planning_link, tool_frame)
        # gives T_planning_tool.
        T_planning_tool = transform_to_matrix(
            tf_msg.transform
        )

        if T_planning_tool is None:
            raise RuntimeError(
                "Invalid tool TF quaternion."
            )

        # T_base_tool =
        #   T_base_planning * T_planning_tool
        #
        # therefore:
        # T_base_planning =
        #   T_base_tool * inv(T_planning_tool)
        T_base_planning = (
            T_base_tool
            @ np.linalg.inv(
                T_planning_tool
            )
        )

        planning_pose = matrix_to_pose(
            T_base_planning
        )

        if planning_pose is None:
            raise RuntimeError(
                "Failed to compute planning-link pose."
            )

        return planning_pose

    def _make_goal_constraints(
        self,
        planning_pose,
    ):
        constraint = Constraints()
        constraint.name = (
            "pick_handover_pose_goal"
        )

        # ----------------------------------------------------------
        # Position constraint
        # ----------------------------------------------------------

        pc = PositionConstraint()
        pc.header.frame_id = (
            self.base_frame
        )
        pc.link_name = (
            self.planning_link
        )
        pc.weight = 1.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [
            float(
                self.position_tolerance
            )
        ]

        region_pose = Pose()
        region_pose.position = (
            planning_pose.position
        )
        region_pose.orientation.w = 1.0

        pc.constraint_region.primitives = [
            sphere
        ]
        pc.constraint_region.primitive_poses = [
            region_pose
        ]

        # ----------------------------------------------------------
        # Orientation constraint
        # ----------------------------------------------------------

        oc = OrientationConstraint()
        oc.header.frame_id = (
            self.base_frame
        )
        oc.link_name = (
            self.planning_link
        )
        oc.orientation = (
            planning_pose.orientation
        )
        oc.absolute_x_axis_tolerance = (
            self.orientation_tolerance
        )
        oc.absolute_y_axis_tolerance = (
            self.orientation_tolerance
        )
        oc.absolute_z_axis_tolerance = (
            self.orientation_tolerance
        )
        oc.weight = 1.0

        constraint.position_constraints = [
            pc
        ]
        constraint.orientation_constraints = [
            oc
        ]

        return constraint

    def _build_move_group_goal(
        self,
        planning_pose,
    ):
        goal = MoveGroup.Goal()

        request = goal.request

        request.group_name = (
            self.planning_group
        )
        request.num_planning_attempts = (
            self.num_planning_attempts
        )
        request.allowed_planning_time = (
            self.allowed_planning_time
        )

        request.max_velocity_scaling_factor = (
            self.velocity_scaling
        )
        request.max_acceleration_scaling_factor = (
            self.acceleration_scaling
        )

        # Let MoveIt use its latest monitored real-arm state.
        request.start_state.is_diff = True

        request.goal_constraints = [
            self._make_goal_constraints(
                planning_pose
            )
        ]

        if self.planner_id:
            request.planner_id = (
                self.planner_id
            )

        if (
            self.pipeline_id
            and
            hasattr(
                request,
                "pipeline_id",
            )
        ):
            request.pipeline_id = (
                self.pipeline_id
            )

        options = goal.planning_options
        options.plan_only = True
        options.look_around = False
        options.replan = False

        return goal

    # ==============================================================
    # Planning
    # ==============================================================

    def _plan_target_internal(
        self,
        target_name,
    ):
        target = self.targets.get(
            target_name
        )

        if target is None:
            return (
                False,
                f"No '{target_name}' target received yet.",
                None,
            )

        self._clear_last_plan()

        try:
            planning_pose = (
                self._desired_tool_to_planning_link(
                    target
                )
            )

        except Exception as exc:
            return (
                False,
                str(exc),
                None,
            )

        if not self.move_group_client.wait_for_server(
            timeout_sec=self.action_server_timeout
        ):
            return (
                False,
                "MoveGroup action "
                f"'{self.move_group_action}' is unavailable.",
                None,
            )

        goal = self._build_move_group_goal(
            planning_pose
        )

        self.get_logger().info(
            f"Planning '{target_name}' from CURRENT real-arm state..."
        )

        send_future = (
            self.move_group_client.send_goal_async(
                goal
            )
        )

        goal_handle = self._wait_future(
            send_future,
            self.action_server_timeout,
        )

        if (
            goal_handle is None
            or
            not goal_handle.accepted
        ):
            return (
                False,
                f"MoveGroup rejected '{target_name}' goal.",
                None,
            )

        result_future = (
            goal_handle.get_result_async()
        )

        wrapped_result = self._wait_future(
            result_future,
            self.plan_result_timeout,
        )

        if wrapped_result is None:
            return (
                False,
                f"Planning '{target_name}' timed out.",
                None,
            )

        result = wrapped_result.result

        error_code = int(
            result.error_code.val
        )

        if error_code != MoveItErrorCodes.SUCCESS:
            return (
                False,
                f"Planning '{target_name}' failed; "
                f"MoveIt error_code={error_code}.",
                None,
            )

        trajectory = (
            result.planned_trajectory
        )

        if (
            len(
                trajectory
                .joint_trajectory
                .points
            )
            == 0
        ):
            return (
                False,
                "MoveIt returned SUCCESS but "
                "trajectory contains no points.",
                None,
            )

        self.last_trajectory = trajectory
        self.last_trajectory_start = (
            result.trajectory_start
        )
        self.last_plan_target = (
            target_name
        )

        display = DisplayTrajectory()
        display.trajectory_start = (
            result.trajectory_start
        )
        display.trajectory = [
            trajectory
        ]

        self.display_pub.publish(
            display
        )

        self._publish_plan_state(
            True,
            target_name,
        )

        point_count = len(
            trajectory
            .joint_trajectory
            .points
        )

        return (
            True,
            (
                f"PLAN OK: {target_name}; "
                f"{point_count} trajectory points."
            ),
            trajectory,
        )

    def _plan_service(
        self,
        target_name,
        request,
        response,
    ):
        del request

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            (
                ok,
                message,
                trajectory,
            ) = self._plan_target_internal(
                target_name
            )

            response.success = bool(ok)
            response.message = message

            if ok:
                self.get_logger().info(
                    message
                )

                if not self.allow_execute:
                    self.get_logger().warning(
                        "Generic execution remains DISABLED "
                        "(allow_execute=false)."
                    )
            else:
                self.get_logger().error(
                    message
                )

            return response

        finally:
            self.operation_lock.release()


    # ==============================================================
    # Execute / clear
    # ==============================================================

    def _execute_pregrasp_service(
        self,
        request,
        response,
    ):
        del request

        if not self.allow_pregrasp_execute:
            response.success = False
            response.message = (
                "PreGrasp execution blocked: "
                "allow_pregrasp_execute=false."
            )
            return response

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            problems = (
                self._check_pregrasp_interlocks()
            )

            if problems:
                response.success = False
                response.message = (
                    "PreGrasp interlock failed: "
                    + ", ".join(problems)
                )
                return response

            target = self.targets[
                "pregrasp"
            ]

            # Stop YOLO / MediaPipe before moving the eye-in-hand camera.
            (
                perception_ok,
                perception_message,
            ) = self._stop_perception_for_motion()

            if not perception_ok:
                response.success = False
                response.message = (
                    "PreGrasp blocked: "
                    + perception_message
                )
                return response

            self.get_logger().info(
                "Perception paused before robot motion."
            )

            # Always generate a FRESH plan after perception has stopped,
            # using the robot's current real joint state.
            (
                plan_ok,
                plan_message,
                trajectory,
            ) = self._plan_target_internal(
                "pregrasp"
            )

            if not plan_ok:
                response.success = False
                response.message = (
                    "Fresh PreGrasp planning failed: "
                    + plan_message
                )
                return response

            self.get_logger().warning(
                "STEP 7B: executing ONLY PreGrasp. "
                "No gripper close, no Grasp descent, no Lift."
            )

            (
                execute_ok,
                execute_message,
            ) = self._execute_trajectory_internal(
                trajectory,
                "pregrasp",
            )

            if not execute_ok:
                response.success = False
                response.message = (
                    "PreGrasp execution failed: "
                    + execute_message
                )
                return response

            (
                verify_ok,
                verify_message,
            ) = self._wait_for_tool_target(
                target
            )

            if not verify_ok:
                response.success = False
                response.message = (
                    "PreGrasp trajectory completed, but "
                    "post-execution pose verification failed after settling: "
                    + verify_message
                )
                return response

            response.success = True
            response.message = (
                "PREGRASP EXECUTE OK | "
                + verify_message
                + " | perception remains paused."
            )

            self.get_logger().info(
                response.message
            )

            return response

        finally:
            self.operation_lock.release()


    def _build_gripper_move_group_goal(
        self,
        target_width,
    ):
        """
        Build a MoveGroup goal for the AgileX MoveIt planning group
        "gripper".

        The generated trajectory contains the external joint "gripper".
        MoveIt/ros2_control remains the ONLY owner of /control/joint_states.
        """
        goal = MoveGroup.Goal()
        request = goal.request

        request.group_name = (
            self.gripper_planning_group
        )
        request.num_planning_attempts = max(
            1,
            self.num_planning_attempts,
        )
        request.allowed_planning_time = (
            self.allowed_planning_time
        )

        request.max_velocity_scaling_factor = (
            self.gripper_velocity_scaling
        )
        request.max_acceleration_scaling_factor = (
            self.gripper_acceleration_scaling
        )

        # Use the current monitored real robot + gripper state.
        request.start_state.is_diff = True

        constraints = Constraints()

        jc = JointConstraint()
        jc.joint_name = (
            self.gripper_joint_name
        )
        jc.position = float(
            target_width
        )
        jc.tolerance_above = float(
            self.gripper_goal_tolerance
        )
        jc.tolerance_below = float(
            self.gripper_goal_tolerance
        )
        jc.weight = 1.0

        constraints.joint_constraints = [
            jc
        ]

        request.goal_constraints = [
            constraints
        ]

        options = goal.planning_options
        options.plan_only = True
        options.look_around = False
        options.replan = False

        return goal

    def _plan_gripper_internal(
        self,
        target_width,
        label,
    ):
        if not self.move_group_client.wait_for_server(
            timeout_sec=self.action_server_timeout
        ):
            return (
                False,
                "MoveGroup action "
                f"'{self.move_group_action}' is unavailable.",
                None,
            )

        goal = self._build_gripper_move_group_goal(
            target_width
        )

        self.get_logger().info(
            f"MoveIt planning gripper '{label}' -> "
            f"{target_width:.3f} m ..."
        )

        send_future = (
            self.move_group_client.send_goal_async(
                goal
            )
        )

        goal_handle = self._wait_future(
            send_future,
            self.action_server_timeout,
        )

        if (
            goal_handle is None
            or
            not goal_handle.accepted
        ):
            return (
                False,
                "MoveGroup rejected gripper goal.",
                None,
            )

        result_future = (
            goal_handle.get_result_async()
        )

        wrapped_result = self._wait_future(
            result_future,
            self.plan_result_timeout,
        )

        if wrapped_result is None:
            return (
                False,
                "MoveIt gripper planning timed out.",
                None,
            )

        result = wrapped_result.result
        error_code = int(
            result.error_code.val
        )

        if error_code != MoveItErrorCodes.SUCCESS:
            return (
                False,
                "MoveIt gripper planning failed; "
                f"error_code={error_code}.",
                None,
            )

        trajectory = (
            result.planned_trajectory
        )

        joint_names = list(
            trajectory.joint_trajectory.joint_names
        )
        points = (
            trajectory
            .joint_trajectory
            .points
        )

        if len(points) == 0:
            return (
                False,
                "MoveIt gripper plan contains no trajectory points.",
                None,
            )

        if self.gripper_joint_name not in joint_names:
            return (
                False,
                "MoveIt gripper trajectory does not contain expected joint "
                f"'{self.gripper_joint_name}'. "
                f"trajectory joints={joint_names}",
                None,
            )

        display = DisplayTrajectory()
        display.trajectory_start = (
            result.trajectory_start
        )
        display.trajectory = [
            trajectory
        ]
        self.display_pub.publish(
            display
        )

        return (
            True,
            (
                f"GRIPPER PLAN OK: {label}; "
                f"target={target_width:.3f} m; "
                f"points={len(points)}; "
                f"joints={joint_names}"
            ),
            trajectory,
        )

    def _moveit_gripper_target_internal(
        self,
        target_width,
        label,
        execute,
    ):
        (
            plan_ok,
            plan_message,
            trajectory,
        ) = self._plan_gripper_internal(
            target_width,
            label,
        )

        if not plan_ok:
            return (
                False,
                plan_message,
            )

        self.get_logger().info(
            plan_message
        )

        if not execute:
            return (
                True,
                plan_message
                + " | PLAN ONLY; no physical gripper motion.",
            )

        if not self.allow_gripper_execute:
            return (
                False,
                "Gripper execution blocked: "
                "allow_gripper_execute=false.",
            )

        gate_opened_here = False

        try:
            if self.gripper_auto_gate:
                (
                    gate_ok,
                    gate_message,
                ) = self._set_control_gate(
                    True,
                    f"MoveIt gripper '{label}' trajectory starting",
                )

                if not gate_ok:
                    return (
                        False,
                        "Gripper execution blocked: cannot OPEN control Gate | "
                        + gate_message,
                    )

                gate_opened_here = True

            (
                execute_ok,
                execute_message,
            ) = self._execute_trajectory_internal(
                trajectory,
                f"moveit_gripper_{label}",
            )

            if not execute_ok:
                return (
                    False,
                    "MoveIt gripper execution failed: "
                    + execute_message,
                )

        finally:
            if gate_opened_here:
                (
                    close_ok,
                    close_message,
                ) = self._set_control_gate(
                    False,
                    f"MoveIt gripper '{label}' trajectory finished",
                )

                if not close_ok:
                    self.get_logger().error(
                        "FAILED TO CLOSE CONTROL GATE AFTER GRIPPER EXECUTION: "
                        + close_message
                    )

        # Empty-gripper open/close diagnostic:
        # verify real physical width reached the target.
        deadline = (
            self.get_clock().now()
            + Duration(seconds=3.0)
        )

        last_detail = "no feedback"

        while (
            rclpy.ok()
            and
            self.get_clock().now() < deadline
        ):
            (
                reached,
                detail,
            ) = self._verify_gripper_width(
                target_width
            )

            last_detail = detail

            if reached:
                return (
                    True,
                    f"MOVEIT GRIPPER {label.upper()} OK | "
                    + detail,
                )

            import time
            time.sleep(0.05)

        return (
            False,
            f"MoveIt gripper '{label}' trajectory completed, "
            "but real gripper did not reach target | "
            + last_detail,
        )

    def _moveit_open_gripper_internal(
        self,
        execute,
    ):
        return self._moveit_gripper_target_internal(
            self.gripper_open_width,
            "open",
            execute,
        )

    def _moveit_close_gripper_internal(
        self,
        execute,
    ):
        return self._moveit_gripper_target_internal(
            self.gripper_close_width,
            "close",
            execute,
        )

    def _plan_gripper_open_service(
        self,
        request,
        response,
    ):
        del request

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            (
                ok,
                message,
            ) = self._moveit_open_gripper_internal(
                execute=False
            )

            response.success = bool(ok)
            response.message = message

            if ok:
                self.get_logger().info(
                    message
                )
            else:
                self.get_logger().error(
                    message
                )

            return response

        finally:
            self.operation_lock.release()

    def _execute_gripper_open_service(
        self,
        request,
        response,
    ):
        del request

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            (
                ok,
                message,
            ) = self._moveit_open_gripper_internal(
                execute=True
            )

            response.success = bool(ok)
            response.message = message

            if ok:
                self.get_logger().info(
                    message
                )
            else:
                self.get_logger().error(
                    message
                )

            return response

        finally:
            self.operation_lock.release()

    def _plan_gripper_close_service(
        self,
        request,
        response,
    ):
        del request

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            (
                ok,
                message,
            ) = self._moveit_close_gripper_internal(
                execute=False
            )

            response.success = bool(ok)
            response.message = message

            if ok:
                self.get_logger().info(
                    message
                )
            else:
                self.get_logger().error(
                    message
                )

            return response

        finally:
            self.operation_lock.release()

    def _execute_gripper_close_service(
        self,
        request,
        response,
    ):
        del request

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            (
                ok,
                message,
            ) = self._moveit_close_gripper_internal(
                execute=True
            )

            response.success = bool(ok)
            response.message = message

            if ok:
                self.get_logger().info(
                    message
                )
            else:
                self.get_logger().error(
                    message
                )

            return response

        finally:
            self.operation_lock.release()

    def _compute_cartesian_target_trajectory(
        self,
        target,
        label,
    ):
        """
        Compute a collision-checked Cartesian path from the CURRENT
        real-arm state to a desired gripper_base target.
        """
        try:
            planning_pose = (
                self._desired_tool_to_planning_link(
                    target
                )
            )
        except Exception as exc:
            return (
                False,
                str(exc),
                None,
                0.0,
            )

        if not self.cartesian_path_client.wait_for_service(
            timeout_sec=self.action_server_timeout
        ):
            return (
                False,
                f"Cartesian path service "
                f"'{self.compute_cartesian_path_service}' unavailable.",
                None,
                0.0,
            )

        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.group_name = self.planning_group
        req.link_name = self.planning_link

        # Stop-and-Look: use MoveIt's latest monitored REAL state.
        req.start_state.is_diff = True
        req.waypoints = [
            planning_pose
        ]
        req.max_step = float(
            self.cartesian_max_step
        )
        req.jump_threshold = float(
            self.cartesian_jump_threshold
        )
        req.avoid_collisions = True

        result = self._wait_future(
            self.cartesian_path_client.call_async(
                req
            ),
            self.plan_result_timeout,
        )

        if result is None:
            return (
                False,
                f"Cartesian '{label}' path request timed out.",
                None,
                0.0,
            )

        fraction = float(
            result.fraction
        )
        error_code = int(
            result.error_code.val
        )

        if error_code != MoveItErrorCodes.SUCCESS:
            return (
                False,
                f"Cartesian '{label}' path failed: "
                f"MoveIt error_code={error_code}, "
                f"fraction={fraction:.3f}.",
                None,
                fraction,
            )

        if fraction < self.cartesian_min_fraction:
            return (
                False,
                f"Cartesian '{label}' path incomplete: "
                f"fraction={fraction:.3f} "
                f"< required {self.cartesian_min_fraction:.3f}.",
                None,
                fraction,
            )

        trajectory = result.solution

        if (
            len(
                trajectory
                .joint_trajectory
                .points
            )
            == 0
        ):
            return (
                False,
                f"Cartesian '{label}' path contains no trajectory points.",
                None,
                fraction,
            )

        display = DisplayTrajectory()
        display.trajectory = [
            trajectory
        ]
        self.display_pub.publish(
            display
        )

        return (
            True,
            (
                f"Cartesian '{label}' path OK | "
                f"fraction={fraction:.3f} | "
                f"points={len(trajectory.joint_trajectory.points)}"
            ),
            trajectory,
            fraction,
        )

    def _compute_cartesian_grasp_trajectory(
        self,
        grasp_target,
    ):
        return self._compute_cartesian_target_trajectory(
            grasp_target,
            "grasp",
        )

    def _execute_grasp_preview_service(
        self,
        request,
        response,
    ):
        """
        STEP 7C sequence:

            already at PreGrasp
                ↓
            verify current tool ≈ PreGrasp
                ↓
            stop perception (idempotent)
                ↓
            MoveIt gripper group -> open
                ↓
            compute Cartesian path PreGrasp -> Grasp
                ↓
            collision check must pass
                ↓
            execute ONLY the descent
                ↓
            verify Grasp pose
                ↓
            STOP

        No gripper close. No Lift. No direct /control/* publishing.
        """
        del request

        if not self.allow_grasp_preview_execute:
            response.success = False
            response.message = (
                "Grasp-preview execution blocked: "
                "allow_grasp_preview_execute=false."
            )
            return response

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            # Reuse the same scene/target safety interlocks.
            problems = (
                self._check_pregrasp_interlocks()
            )

            if self.targets.get("grasp") is None:
                problems.append(
                    "no grasp target"
                )

            if problems:
                response.success = False
                response.message = (
                    "Grasp-preview interlock failed: "
                    + ", ".join(problems)
                )
                return response

            pregrasp_target = self.targets[
                "pregrasp"
            ]
            grasp_target = self.targets[
                "grasp"
            ]

            # Critical: do NOT descend unless the robot is already at the
            # verified PreGrasp pose.
            (
                at_pregrasp,
                pregrasp_error,
            ) = self._verify_tool_target(
                pregrasp_target,
                position_tolerance=(
                    self.pregrasp_required_position_tolerance
                ),
                orientation_tolerance=(
                    self.pregrasp_required_orientation_tolerance
                ),
            )

            if not at_pregrasp:
                response.success = False
                response.message = (
                    "Grasp-preview blocked: robot is not at PreGrasp | "
                    + pregrasp_error
                )
                return response

            # Stop perception again; this is safe/idempotent and guarantees
            # no Stop-and-Look locking while the eye-in-hand camera moves.
            (
                perception_ok,
                perception_message,
            ) = self._stop_perception_for_motion()

            if not perception_ok:
                response.success = False
                response.message = (
                    "Grasp-preview blocked: "
                    + perception_message
                )
                return response

            # ------------------------------------------------------
            # Open the gripper THROUGH MoveIt.
            #
            # Do not publish a second /control/joint_states source while
            # ros2_control is already streaming that topic at ~200 Hz.
            # ------------------------------------------------------

            (
                gripper_ok,
                gripper_message,
            ) = self._moveit_open_gripper_internal(
                execute=True
            )

            if not gripper_ok:
                response.success = False
                response.message = (
                    "Grasp-preview blocked: "
                    + gripper_message
                )
                return response

            self.get_logger().info(
                gripper_message
            )
            self.get_logger().info(
                "MoveIt gripper OPEN verified. "
                "Computing Cartesian PreGrasp -> Grasp path."
            )

            (
                cart_ok,
                cart_message,
                trajectory,
                fraction,
            ) = self._compute_cartesian_grasp_trajectory(
                grasp_target
            )

            if not cart_ok:
                response.success = False
                response.message = cart_message
                return response

            self.get_logger().warning(
                "STEP 7C: executing ONLY Cartesian descent to Grasp. "
                "Gripper will remain OPEN. No Lift."
            )

            (
                execute_ok,
                execute_message,
            ) = self._execute_trajectory_internal(
                trajectory,
                "grasp_preview_cartesian",
            )

            if not execute_ok:
                response.success = False
                response.message = (
                    "Grasp-preview execution failed: "
                    + execute_message
                )
                return response

            (
                verify_ok,
                verify_message,
            ) = self._wait_for_tool_target(
                grasp_target
            )

            if not verify_ok:
                response.success = False
                response.message = (
                    "Reached end of trajectory but Grasp pose verification "
                    "failed after settling: "
                    + verify_message
                )
                return response

            response.success = True
            response.message = (
                "GRASP PREVIEW EXECUTE OK | "
                f"cartesian_fraction={fraction:.3f} | "
                + verify_message
                + " | gripper remains OPEN | no Lift."
            )

            self.get_logger().info(
                response.message
            )

            return response

        finally:
            self.operation_lock.release()

    def _execute_gripper_grasp_service(
        self,
        request,
        response,
    ):
        """
        REAL object grasp.

        MoveIt still commands a close target, but success is determined
        from REAL jaw width after execution.  If an object is present,
        the jaws should stop at a non-zero width.
        """
        del request

        if not self.allow_object_grasp_execute:
            response.success = False
            response.message = (
                "Object grasp blocked: "
                "allow_object_grasp_execute=false."
            )
            return response

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            grasp_target = self.targets.get(
                "grasp"
            )

            if grasp_target is None:
                response.success = False
                response.message = (
                    "Object grasp blocked: no grasp target."
                )
                return response

            (
                at_grasp,
                pose_detail,
            ) = self._verify_tool_target(
                grasp_target
            )

            if not at_grasp:
                response.success = False
                response.message = (
                    "Object grasp blocked: robot is not at Grasp | "
                    + pose_detail
                )
                return response

            initial_width = (
                self._get_gripper_width()
            )

            if initial_width is None:
                response.success = False
                response.message = (
                    "Object grasp blocked: no real gripper feedback."
                )
                return response

            self._publish_object_grasped(
                False
            )

            (
                plan_ok,
                plan_message,
                trajectory,
            ) = self._plan_gripper_internal(
                self.gripper_close_width,
                "object_grasp",
            )

            if not plan_ok:
                response.success = False
                response.message = plan_message
                return response

            gate_opened = False

            try:
                if self.gripper_auto_gate:
                    (
                        gate_ok,
                        gate_message,
                    ) = self._set_control_gate(
                        True,
                        "real object grasp starting",
                    )

                    if not gate_ok:
                        response.success = False
                        response.message = (
                            "Object grasp blocked: cannot open Gate | "
                            + gate_message
                        )
                        return response

                    gate_opened = True

                (
                    execute_ok,
                    execute_message,
                ) = self._execute_trajectory_internal(
                    trajectory,
                    "gripper_object_grasp",
                )

                if not execute_ok:
                    response.success = False
                    response.message = (
                        "Object grasp execution failed: "
                        + execute_message
                    )
                    return response

            finally:
                if gate_opened:
                    (
                        close_ok,
                        close_message,
                    ) = self._set_control_gate(
                        False,
                        "real object grasp finished",
                    )

                    if not close_ok:
                        self.get_logger().error(
                            "FAILED TO CLOSE CONTROL GATE AFTER OBJECT GRASP: "
                            + close_message
                        )

            time.sleep(
                max(
                    0.0,
                    self.object_grasp_settle_sec,
                )
            )

            final_width = (
                self._get_gripper_width()
            )

            if final_width is None:
                response.success = False
                response.message = (
                    "Object grasp executed but final gripper feedback "
                    "is unavailable."
                )
                return response

            closure = float(
                initial_width
                - final_width
            )

            width_ok = (
                self.object_grasp_min_width
                <= final_width
                <= self.object_grasp_max_width
            )
            closure_ok = (
                closure
                >= self.object_grasp_min_closure
            )

            if not width_ok:
                response.success = False
                response.message = (
                    "OBJECT GRASP NOT VERIFIED | "
                    f"initial={initial_width:.4f} m, "
                    f"final={final_width:.4f} m. "
                    "Final width is outside configured object-width range "
                    f"[{self.object_grasp_min_width:.4f}, "
                    f"{self.object_grasp_max_width:.4f}] m. "
                    "If final width is near zero, the fingers probably "
                    "closed without catching the case."
                )
                return response

            if not closure_ok:
                response.success = False
                response.message = (
                    "OBJECT GRASP NOT VERIFIED | "
                    f"closure={closure * 1000.0:.1f} mm "
                    f"< required "
                    f"{self.object_grasp_min_closure * 1000.0:.1f} mm."
                )
                return response

            self._publish_object_grasped(
                True
            )

            response.success = True
            response.message = (
                "OBJECT GRASP VERIFIED | "
                f"initial_width={initial_width:.4f} m | "
                f"held_width={final_width:.4f} m | "
                f"closure={closure * 1000.0:.1f} mm."
            )

            self.get_logger().info(
                response.message
            )
            return response

        finally:
            self.operation_lock.release()

    def _execute_lift_service(
        self,
        request,
        response,
    ):
        """
        Cartesian lift from current Grasp pose to /pick/lift_pose.
        """
        del request

        if not self.allow_lift_execute:
            response.success = False
            response.message = (
                "Lift blocked: allow_lift_execute=false."
            )
            return response

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            grasp_target = self.targets.get(
                "grasp"
            )
            lift_target = self.targets.get(
                "lift"
            )

            if (
                grasp_target is None
                or lift_target is None
            ):
                response.success = False
                response.message = (
                    "Lift blocked: grasp/lift target missing."
                )
                return response

            if (
                self.require_object_grasped_for_lift
                and
                not self.object_grasped
            ):
                response.success = False
                response.message = (
                    "Lift blocked: /pick/object_grasped is false. "
                    "Use /moveit/execute_gripper_grasp first."
                )
                return response

            (
                at_grasp,
                grasp_detail,
            ) = self._verify_tool_target(
                grasp_target,
                position_tolerance=(
                    self.lift_required_position_tolerance
                ),
                orientation_tolerance=(
                    self.lift_required_orientation_tolerance
                ),
            )

            if not at_grasp:
                response.success = False
                response.message = (
                    "Lift blocked: robot is not at Grasp | "
                    + grasp_detail
                )
                return response

            (
                perception_ok,
                perception_message,
            ) = self._stop_perception_for_motion()

            if not perception_ok:
                response.success = False
                response.message = (
                    "Lift blocked: "
                    + perception_message
                )
                return response

            (
                cart_ok,
                cart_message,
                trajectory,
                fraction,
            ) = self._compute_cartesian_target_trajectory(
                lift_target,
                "lift",
            )

            if not cart_ok:
                response.success = False
                response.message = cart_message
                return response

            self.get_logger().warning(
                "STEP 7D: executing Cartesian LIFT with object."
            )

            (
                execute_ok,
                execute_message,
            ) = self._execute_trajectory_internal(
                trajectory,
                "lift_cartesian",
            )

            if not execute_ok:
                response.success = False
                response.message = (
                    "Lift execution failed: "
                    + execute_message
                )
                return response

            (
                verify_ok,
                verify_message,
            ) = self._wait_for_tool_target(
                lift_target
            )

            if not verify_ok:
                response.success = False
                response.message = (
                    "Lift trajectory ended but target verification failed "
                    "after settling: "
                    + verify_message
                )
                return response

            # IMPORTANT: target-tolerance verification is not the same as
            # dynamic settling.  The first TF sample may already be inside the
            # (relatively wide) target tolerance while the real arm/controller
            # state is still converging.  Hold here before manager is allowed
            # to start Observe-Hand planning.
            settle_sec = max(
                0.0,
                float(self.post_lift_settle_sec),
            )

            if settle_sec > 0.0:
                self.get_logger().info(
                    "Lift target verified. Waiting "
                    f"{settle_sec:.2f}s for real-arm/controller state "
                    "to settle before Observe-Hand."
                )
                time.sleep(settle_sec)

            response.success = True
            response.message = (
                "LIFT EXECUTE OK | "
                f"cartesian_fraction={fraction:.3f} | "
                + verify_message
                + f" | post_lift_settle={settle_sec:.2f}s"
                + " | perception remains PAUSED."
            )

            self.get_logger().info(
                response.message
            )
            return response

        finally:
            self.operation_lock.release()

    def _execute_observe_hand_service(
        self,
        request,
        response,
    ):
        """
        Move from Lift to a generated Observe-Hand pose, stop, arm a NEW
        Palm Final lock, then restart perception.

        This preserves Stop-and-Look:
            move -> stop -> arm final palm -> perception -> lock NEW palm.
        """
        del request

        if not self.allow_observe_hand_execute:
            response.success = False
            response.message = (
                "Observe-Hand execution blocked: "
                "allow_observe_hand_execute=false."
            )
            return response

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            observe_target = self.targets.get(
                "observe_hand"
            )
            lift_target = self.targets.get(
                "lift"
            )

            if observe_target is None:
                response.success = False
                response.message = (
                    "Observe-Hand blocked: "
                    "no /handover/observe_hand_pose."
                )
                return response

            if (
                self.require_lift_before_observe
                and
                lift_target is not None
            ):
                (
                    at_lift,
                    lift_detail,
                ) = self._verify_tool_target(
                    lift_target,
                    position_tolerance=(
                        self.observe_required_position_tolerance
                    ),
                    orientation_tolerance=(
                        self.observe_required_orientation_tolerance
                    ),
                )

                if not at_lift:
                    response.success = False
                    response.message = (
                        "Observe-Hand blocked: robot is not at Lift | "
                        + lift_detail
                    )
                    return response

            (
                perception_ok,
                perception_message,
            ) = self._stop_perception_for_motion()

            if not perception_ok:
                response.success = False
                response.message = (
                    "Observe-Hand blocked: "
                    + perception_message
                )
                return response

            (
                plan_ok,
                plan_message,
                trajectory,
            ) = self._plan_target_internal(
                "observe_hand"
            )

            if not plan_ok:
                response.success = False
                response.message = (
                    "Observe-Hand planning failed: "
                    + plan_message
                )
                return response

            self.get_logger().warning(
                "STEP 7D: moving toward Palm Initial observation pose."
            )

            (
                execute_ok,
                execute_message,
            ) = self._execute_trajectory_internal(
                trajectory,
                "observe_hand",
            )

            if not execute_ok:
                response.success = False
                response.message = (
                    "Observe-Hand execution failed: "
                    + execute_message
                )
                return response

            (
                verify_ok,
                verify_message,
            ) = self._wait_for_tool_target(
                observe_target
            )

            if not verify_ok:
                response.success = False
                response.message = (
                    "Observe-Hand target verification failed after settling: "
                    + verify_message
                )
                return response

            time.sleep(
                max(
                    0.0,
                    self.observe_settle_sec,
                )
            )

            # Clear old palm samples and arm a fresh Palm Final lock FIRST.
            (
                palm_arm_ok,
                palm_arm_message,
            ) = self._call_trigger_service(
                self.arm_palm_final_client,
                self.arm_palm_final_service,
            )

            if not palm_arm_ok:
                response.success = False
                response.message = (
                    "Robot reached Observe-Hand, but Palm Final could not "
                    "be armed: "
                    + palm_arm_message
                )
                return response

            # Only after the final-lock window has been cleared do we
            # restart perception.
            (
                perception_start_ok,
                perception_start_message,
            ) = self._call_trigger_service(
                self.perception_start_client,
                self.perception_start_service,
            )

            if not perception_start_ok:
                response.success = False
                response.message = (
                    "Palm Final armed, but perception could not restart: "
                    + perception_start_message
                )
                return response

            response.success = True
            response.message = (
                "OBSERVE HAND EXECUTE OK | "
                + verify_message
                + " | Palm Final ARMED | perception STARTED | "
                "now wait for /targets/palm_final_locked = true."
            )

            self.get_logger().info(
                response.message
            )
            return response

        finally:
            self.operation_lock.release()

    def _execute_handover_approach_service(
        self,
        request,
        response,
    ):
        """
        Move toward the NEW Palm Final, but stop at a conservative
        approach distance. Perception is paused during motion.
        """
        del request

        if not self.allow_handover_approach_execute:
            response.success = False
            response.message = (
                "Handover Approach blocked: "
                "allow_handover_approach_execute=false."
            )
            return response

        if not self.palm_final_locked:
            response.success = False
            response.message = (
                "Handover Approach blocked: "
                "/targets/palm_final_locked is false."
            )
            return response

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            target = self.targets.get(
                "handover_approach"
            )

            if target is None:
                response.success = False
                response.message = (
                    "Handover Approach blocked: "
                    "no /handover/approach_pose."
                )
                return response

            (
                perception_ok,
                perception_message,
            ) = self._stop_perception_for_motion()

            if not perception_ok:
                response.success = False
                response.message = (
                    "Handover Approach blocked: "
                    + perception_message
                )
                return response

            (
                plan_ok,
                plan_message,
                trajectory,
            ) = self._plan_target_internal(
                "handover_approach"
            )

            if not plan_ok:
                response.success = False
                response.message = (
                    "Handover Approach planning failed: "
                    + plan_message
                )
                return response

            self.get_logger().warning(
                "STEP 7E: executing Handover Approach toward Palm Final."
            )

            (
                execute_ok,
                execute_message,
            ) = self._execute_trajectory_internal(
                trajectory,
                "handover_approach",
            )

            if not execute_ok:
                response.success = False
                response.message = (
                    "Handover Approach execution failed: "
                    + execute_message
                )
                return response

            (
                verify_ok,
                verify_message,
            ) = self._wait_for_tool_target(
                target,
                position_tolerance=(
                    self.handover_approach_required_position_tolerance
                ),
                orientation_tolerance=(
                    self.handover_approach_required_orientation_tolerance
                ),
            )

            if not verify_ok:
                response.success = False
                response.message = (
                    "Handover Approach verification failed after settling: "
                    + verify_message
                )
                return response

            response.success = True
            response.message = (
                "HANDOVER APPROACH OK | "
                + verify_message
                + " | perception remains PAUSED."
            )
            return response

        finally:
            self.operation_lock.release()

    def _plan_handover_final_service(
        self,
        request,
        response,
    ):
        """
        Plan-only Cartesian straight approach from Handover Approach
        to Handover Final. No robot motion.
        """
        del request

        if not self.palm_final_locked:
            response.success = False
            response.message = (
                "Handover Final planning blocked: "
                "/targets/palm_final_locked is false."
            )
            return response

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            approach = self.targets.get(
                "handover_approach"
            )
            final = self.targets.get(
                "handover_final"
            )

            if (
                approach is None
                or final is None
            ):
                response.success = False
                response.message = (
                    "Handover Final planning blocked: "
                    "approach/final target missing."
                )
                return response

            (
                at_approach,
                detail,
            ) = self._verify_tool_target(
                approach,
                position_tolerance=(
                    self.handover_approach_required_position_tolerance
                ),
                orientation_tolerance=(
                    self.handover_approach_required_orientation_tolerance
                ),
            )

            if not at_approach:
                response.success = False
                response.message = (
                    "Handover Final planning blocked: "
                    "robot is not at Handover Approach | "
                    + detail
                )
                return response

            (
                cart_ok,
                cart_message,
                trajectory,
                fraction,
            ) = self._compute_cartesian_target_trajectory(
                final,
                "handover_final",
            )

            response.success = bool(
                cart_ok
            )
            response.message = (
                cart_message
                + (
                    " | PLAN ONLY; no physical motion."
                    if cart_ok
                    else ""
                )
            )
            return response

        finally:
            self.operation_lock.release()

    def _execute_handover_final_service(
        self,
        request,
        response,
    ):
        """
        Final straight Cartesian handover approach toward Palm Final.

        This stage DOES NOT release the gripper. It only moves the held
        object to the configured final handover standoff.
        """
        del request

        if not self.allow_handover_final_execute:
            response.success = False
            response.message = (
                "Handover Final blocked: "
                "allow_handover_final_execute=false."
            )
            return response

        if not self.palm_final_locked:
            response.success = False
            response.message = (
                "Handover Final blocked: "
                "/targets/palm_final_locked is false."
            )
            return response

        if not self.object_grasped:
            response.success = False
            response.message = (
                "Handover Final blocked: "
                "/pick/object_grasped is false."
            )
            return response

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            approach = self.targets.get(
                "handover_approach"
            )
            final = self.targets.get(
                "handover_final"
            )

            if (
                approach is None
                or final is None
            ):
                response.success = False
                response.message = (
                    "Handover Final blocked: "
                    "approach/final target missing."
                )
                return response

            (
                at_approach,
                approach_detail,
            ) = self._verify_tool_target(
                approach,
                position_tolerance=(
                    self.handover_approach_required_position_tolerance
                ),
                orientation_tolerance=(
                    self.handover_approach_required_orientation_tolerance
                ),
            )

            if not at_approach:
                response.success = False
                response.message = (
                    "Handover Final blocked: "
                    "robot is not at Handover Approach | "
                    + approach_detail
                )
                return response

            (
                cart_ok,
                cart_message,
                trajectory,
                fraction,
            ) = self._compute_cartesian_target_trajectory(
                final,
                "handover_final",
            )

            if not cart_ok:
                response.success = False
                response.message = cart_message
                return response

            self.get_logger().warning(
                "STEP 7E: FINAL HANDOVER MOTION toward human hand. "
                "Gripper will remain CLOSED."
            )

            (
                execute_ok,
                execute_message,
            ) = self._execute_trajectory_internal(
                trajectory,
                "handover_final",
            )

            if not execute_ok:
                response.success = False
                response.message = (
                    "Handover Final execution failed: "
                    + execute_message
                )
                return response

            (
                verify_ok,
                verify_message,
            ) = self._wait_for_tool_target(
                final,
                position_tolerance=(
                    self.handover_final_required_position_tolerance
                ),
                orientation_tolerance=(
                    self.handover_final_required_orientation_tolerance
                ),
            )

            if not verify_ok:
                response.success = False
                response.message = (
                    "Handover Final verification failed after settling: "
                    + verify_message
                )
                return response

            response.success = True
            response.message = (
                "HANDOVER FINAL OK | "
                f"cartesian_fraction={fraction:.3f} | "
                + verify_message
                + " | gripper remains CLOSED; no release yet."
            )
            return response

        finally:
            self.operation_lock.release()

    def _execute_service(
        self,
        request,
        response,
    ):
        del request

        if not self.allow_execute:
            response.success = False
            response.message = (
                "Execution blocked: "
                "allow_execute=false. "
                "STEP 6 is plan-only."
            )
            return response

        if not self.operation_lock.acquire(
            blocking=False
        ):
            response.success = False
            response.message = (
                "MoveIt operation already in progress."
            )
            return response

        try:
            if self.last_trajectory is None:
                response.success = False
                response.message = (
                    "No valid stored trajectory."
                )
                return response

            if not self.execute_client.wait_for_server(
                timeout_sec=(
                    self.action_server_timeout
                )
            ):
                response.success = False
                response.message = (
                    f"ExecuteTrajectory action "
                    f"'{self.execute_action}' "
                    "is unavailable."
                )
                return response

            goal = ExecuteTrajectory.Goal()
            goal.trajectory = (
                self.last_trajectory
            )

            self.get_logger().warning(
                "EXECUTING stored trajectory: "
                f"{self.last_plan_target}"
            )

            send_future = (
                self.execute_client
                .send_goal_async(goal)
            )

            goal_handle = self._wait_future(
                send_future,
                self.action_server_timeout,
            )

            if (
                goal_handle is None
                or
                not goal_handle.accepted
            ):
                response.success = False
                response.message = (
                    "ExecuteTrajectory goal rejected."
                )
                return response

            result_future = (
                goal_handle.get_result_async()
            )

            wrapped_result = self._wait_future(
                result_future,
                self.execute_result_timeout,
            )

            if wrapped_result is None:
                response.success = False
                response.message = (
                    "Trajectory execution timed out."
                )
                return response

            result = wrapped_result.result

            error_code = int(
                result.error_code.val
            )

            response.success = (
                error_code
                == MoveItErrorCodes.SUCCESS
            )

            response.message = (
                "EXECUTE OK"
                if response.success
                else (
                    "EXECUTE FAILED: "
                    f"MoveIt error_code={error_code}"
                )
            )

            return response

        finally:
            self.operation_lock.release()

    def _clear_service(
        self,
        request,
        response,
    ):
        del request

        self._clear_last_plan()

        response.success = True
        response.message = (
            "Stored MoveIt plan cleared."
        )

        return response


def main(args=None):
    rclpy.init(args=args)

    node = MoveItExecutorNode()

    executor = MultiThreadedExecutor(
        num_threads=4
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
