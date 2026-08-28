import copy
import math
import time

import numpy as np
import rclpy

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, Empty
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener


def quat_angle_deg(q1, q2):
    """
    Smallest angular distance between two quaternions, in degrees.
    q = [x, y, z, w]
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    n1 = np.linalg.norm(q1)
    n2 = np.linalg.norm(q2)

    if n1 < 1e-12 or n2 < 1e-12:
        return 180.0

    q1 = q1 / n1
    q2 = q2 / n2

    dot = abs(
        float(
            np.clip(
                np.dot(q1, q2),
                -1.0,
                1.0,
            )
        )
    )

    return float(
        np.degrees(
            2.0 * np.arccos(dot)
        )
    )


class MoveItAutoHandshakeController(Node):
    """
    Automatic sequence:

        WAIT_TARGET
          -> PLAN_APPROACH
          -> EXECUTE_APPROACH
          -> VERIFY_APPROACH
          -> PLAN_HANDSHAKE
          -> EXECUTE_HANDSHAKE
          -> VERIFY_HANDSHAKE
          -> DWELL
          -> PLAN_HOME
          -> EXECUTE_HOME
          -> VERIFY_HOME
          -> DONE

    "HOME" here means the arm joint configuration captured at the
    beginning of the cycle. It does NOT mean mechanical zero.

    The final handshake target should still keep a non-contact clearance
    (e.g. 0.05 m) unless a separate force-controlled contact strategy is
    implemented.
    """

    WAIT_TARGET = "WAIT_TARGET"
    PLANNING_APPROACH = "PLANNING_APPROACH"
    EXECUTING_APPROACH = "EXECUTING_APPROACH"
    VERIFYING_APPROACH = "VERIFYING_APPROACH"

    PLANNING_HANDSHAKE = "PLANNING_HANDSHAKE"
    EXECUTING_HANDSHAKE = "EXECUTING_HANDSHAKE"
    VERIFYING_HANDSHAKE = "VERIFYING_HANDSHAKE"

    DWELL = "DWELL"

    PLANNING_HOME = "PLANNING_HOME"
    EXECUTING_HOME = "EXECUTING_HOME"
    VERIFYING_HOME = "VERIFYING_HOME"

    DONE = "DONE"
    FAILED = "FAILED"

    def __init__(self):
        super().__init__("moveit_auto_handshake_controller")

        # ==========================================================
        # Parameters
        # ==========================================================

        self.declare_parameter("planning_group", "arm")
        self.declare_parameter("pose_link", "gripper_base")
        self.declare_parameter("base_frame", "base_link")

        self.declare_parameter("move_action", "/move_action")
        self.declare_parameter(
            "execute_action",
            "/execute_trajectory"
        )

        self.declare_parameter(
            "joint_feedback_topic",
            "/feedback/joint_states"
        )

        self.declare_parameter(
            "arm_joint_names",
            [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ]
        )

        # Master safety switch.
        self.declare_parameter(
            "enable_auto_motion",
            False
        )

        # If True, the sequence automatically starts once both
        # approach_pose + handshake_pose + plan_ready are available.
        self.declare_parameter(
            "auto_start_on_plan_ready",
            True
        )

        self.declare_parameter(
            "planning_time_s",
            5.0
        )

        self.declare_parameter(
            "planning_attempts",
            5
        )

        # Goal tolerances used by MoveIt planning
        self.declare_parameter(
            "position_tolerance_m",
            0.010
        )

        self.declare_parameter(
            "orientation_tolerance_deg",
            5.0
        )

        # Different speed limits for each phase
        self.declare_parameter(
            "approach_velocity_scaling",
            0.10
        )

        self.declare_parameter(
            "approach_acceleration_scaling",
            0.10
        )

        self.declare_parameter(
            "handshake_velocity_scaling",
            0.05
        )

        self.declare_parameter(
            "handshake_acceleration_scaling",
            0.05
        )

        self.declare_parameter(
            "return_velocity_scaling",
            0.10
        )

        self.declare_parameter(
            "return_acceleration_scaling",
            0.10
        )

        # Precision verification after execution
        self.declare_parameter(
            "approach_verify_position_m",
            0.020
        )

        self.declare_parameter(
            "approach_verify_orientation_deg",
            10.0
        )

        self.declare_parameter(
            "handshake_verify_position_m",
            0.015
        )

        self.declare_parameter(
            "handshake_verify_orientation_deg",
            8.0
        )

        self.declare_parameter(
            "home_verify_joint_deg",
            2.0
        )

        # Require several consecutive good samples before declaring
        # the pose "precisely reached".
        self.declare_parameter(
            "verify_stable_samples",
            5
        )

        self.declare_parameter(
            "verify_timeout_s",
            5.0
        )

        # Stay at the final non-contact handshake pose before returning
        self.declare_parameter(
            "handshake_dwell_s",
            1.5
        )

        # ==========================================================
        # Read parameters
        # ==========================================================

        self.planning_group = str(
            self.get_parameter(
                "planning_group"
            ).value
        )

        self.pose_link = str(
            self.get_parameter(
                "pose_link"
            ).value
        )

        self.base_frame = str(
            self.get_parameter(
                "base_frame"
            ).value
        )

        self.move_action_name = str(
            self.get_parameter(
                "move_action"
            ).value
        )

        self.execute_action_name = str(
            self.get_parameter(
                "execute_action"
            ).value
        )

        self.joint_feedback_topic = str(
            self.get_parameter(
                "joint_feedback_topic"
            ).value
        )

        self.arm_joint_names = list(
            self.get_parameter(
                "arm_joint_names"
            ).value
        )

        self.enable_auto_motion = bool(
            self.get_parameter(
                "enable_auto_motion"
            ).value
        )

        self.auto_start_on_plan_ready = bool(
            self.get_parameter(
                "auto_start_on_plan_ready"
            ).value
        )

        self.planning_time = float(
            self.get_parameter(
                "planning_time_s"
            ).value
        )

        self.planning_attempts = int(
            self.get_parameter(
                "planning_attempts"
            ).value
        )

        self.position_tolerance = float(
            self.get_parameter(
                "position_tolerance_m"
            ).value
        )

        self.orientation_tolerance = math.radians(
            float(
                self.get_parameter(
                    "orientation_tolerance_deg"
                ).value
            )
        )

        self.approach_velocity = float(
            self.get_parameter(
                "approach_velocity_scaling"
            ).value
        )

        self.approach_acceleration = float(
            self.get_parameter(
                "approach_acceleration_scaling"
            ).value
        )

        self.handshake_velocity = float(
            self.get_parameter(
                "handshake_velocity_scaling"
            ).value
        )

        self.handshake_acceleration = float(
            self.get_parameter(
                "handshake_acceleration_scaling"
            ).value
        )

        self.return_velocity = float(
            self.get_parameter(
                "return_velocity_scaling"
            ).value
        )

        self.return_acceleration = float(
            self.get_parameter(
                "return_acceleration_scaling"
            ).value
        )

        self.approach_verify_position = float(
            self.get_parameter(
                "approach_verify_position_m"
            ).value
        )

        self.approach_verify_orientation = float(
            self.get_parameter(
                "approach_verify_orientation_deg"
            ).value
        )

        self.handshake_verify_position = float(
            self.get_parameter(
                "handshake_verify_position_m"
            ).value
        )

        self.handshake_verify_orientation = float(
            self.get_parameter(
                "handshake_verify_orientation_deg"
            ).value
        )

        self.home_verify_joint = math.radians(
            float(
                self.get_parameter(
                    "home_verify_joint_deg"
                ).value
            )
        )

        self.verify_stable_samples = int(
            self.get_parameter(
                "verify_stable_samples"
            ).value
        )

        self.verify_timeout = float(
            self.get_parameter(
                "verify_timeout_s"
            ).value
        )

        self.handshake_dwell = float(
            self.get_parameter(
                "handshake_dwell_s"
            ).value
        )

        # ==========================================================
        # State
        # ==========================================================

        self.state = self.WAIT_TARGET

        self.approach_pose = None
        self.handshake_pose = None
        self.plan_ready = False

        self.current_joint_state = None

        # Snapshot of robot's starting joint configuration
        self.home_joint_positions = None

        # Frozen target snapshots for this cycle
        self.cycle_approach_pose = None
        self.cycle_handshake_pose = None

        self.pending_plan_kind = None
        self.pending_execute_kind = None

        self.verify_good_count = 0
        self.verify_start_time = 0.0
        self.dwell_start_time = 0.0

        # ==========================================================
        # TF
        # ==========================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ==========================================================
        # ROS interfaces
        # ==========================================================

        self.create_subscription(
            PoseStamped,
            "/handshake/approach_pose",
            self.approach_callback,
            10
        )

        self.create_subscription(
            PoseStamped,
            "/handshake/handshake_pose",
            self.handshake_callback,
            10
        )

        self.create_subscription(
            Bool,
            "/handshake/plan_ready",
            self.plan_ready_callback,
            10
        )

        self.create_subscription(
            JointState,
            self.joint_feedback_topic,
            self.joint_state_callback,
            10
        )

        self.create_subscription(
            Empty,
            "/handshake/reset_event",
            self.reset_callback,
            10
        )

        self.start_service = self.create_service(
            Trigger,
            "/handshake/start_auto_cycle",
            self.start_service_callback
        )

        self.cycle_success_pub = self.create_publisher(
            Bool,
            "/handshake/auto_cycle_success",
            10
        )

        self.display_pub = self.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            10
        )

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            self.move_action_name
        )

        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            self.execute_action_name
        )

        # 10 Hz state/verification loop
        self.timer = self.create_timer(
            0.1,
            self.timer_callback
        )

        self.get_logger().info(
            "Automatic MoveIt handshake controller started"
        )

        self.get_logger().info(
            "HOME = robot joint configuration captured "
            "at start of each cycle"
        )

        if self.enable_auto_motion:

            self.get_logger().warning(
                "AUTO REAL ROBOT MOTION ENABLED"
            )

        else:

            self.get_logger().warning(
                "AUTO MOTION DISABLED - set "
                "enable_auto_motion:=true to allow execution"
            )

    # ==============================================================
    # Input callbacks
    # ==============================================================

    def approach_callback(
        self,
        msg: PoseStamped
    ):

        if self.state != self.WAIT_TARGET:
            return

        self.approach_pose = msg

        self.maybe_auto_start()

    def handshake_callback(
        self,
        msg: PoseStamped
    ):

        if self.state != self.WAIT_TARGET:
            return

        self.handshake_pose = msg

        self.maybe_auto_start()

    def plan_ready_callback(
        self,
        msg: Bool
    ):

        if self.state != self.WAIT_TARGET:
            return

        self.plan_ready = bool(
            msg.data
        )

        self.maybe_auto_start()

    def joint_state_callback(
        self,
        msg: JointState
    ):

        self.current_joint_state = msg

    # ==============================================================
    # Start/reset
    # ==============================================================

    def ready_to_start(self):

        return (
            self.plan_ready
            and
            self.approach_pose is not None
            and
            self.handshake_pose is not None
            and
            self.current_joint_state is not None
        )

    def maybe_auto_start(self):

        if not self.auto_start_on_plan_ready:
            return

        if not self.enable_auto_motion:
            return

        if self.state != self.WAIT_TARGET:
            return

        if not self.ready_to_start():
            return

        self.start_cycle()

    def start_service_callback(
        self,
        request,
        response
    ):

        if not self.enable_auto_motion:

            response.success = False
            response.message = (
                "enable_auto_motion is false"
            )

            return response

        if self.state != self.WAIT_TARGET:

            response.success = False
            response.message = (
                f"Controller is in state {self.state}"
            )

            return response

        if not self.ready_to_start():

            response.success = False
            response.message = (
                "Approach/handshake targets or joint feedback "
                "are not ready"
            )

            return response

        ok, reason = self.start_cycle()

        response.success = ok
        response.message = reason

        return response

    def start_cycle(self):

        # Capture arm start configuration as HOME.
        current_map = {
            name: float(position)
            for name, position
            in zip(
                self.current_joint_state.name,
                self.current_joint_state.position
            )
        }

        missing = [
            name
            for name in self.arm_joint_names
            if name not in current_map
        ]

        if missing:

            reason = (
                f"Cannot capture HOME. Missing joints: "
                f"{missing}"
            )

            self.fail(reason)

            return False, reason

        self.home_joint_positions = {
            name: current_map[name]
            for name in self.arm_joint_names
        }

        self.cycle_approach_pose = copy.deepcopy(
            self.approach_pose
        )

        self.cycle_handshake_pose = copy.deepcopy(
            self.handshake_pose
        )

        self.get_logger().warning(
            "======================================"
        )

        self.get_logger().warning(
            "AUTO HANDSHAKE CYCLE START"
        )

        self.get_logger().warning(
            "Start joint pose captured as HOME."
        )

        self.get_logger().warning(
            "Final handshake pose must remain "
            "NON-CONTACT / SAFE CLEARANCE."
        )

        self.get_logger().warning(
            "======================================"
        )

        self.plan_pose_target(
            kind="approach",
            target=self.cycle_approach_pose,
            velocity=self.approach_velocity,
            acceleration=self.approach_acceleration,
        )

        return (
            True,
            "Automatic handshake cycle started"
        )

    def reset_callback(
        self,
        msg: Empty
    ):

        # Do not attempt to cancel an already active controller action here.
        # Reset is intended for idle/DONE/FAILED state.
        if self.state in (
            self.EXECUTING_APPROACH,
            self.EXECUTING_HANDSHAKE,
            self.EXECUTING_HOME,
        ):

            self.get_logger().warning(
                "Reset received while robot is executing. "
                "Ignoring local reset until execution completes."
            )

            return

        self.state = self.WAIT_TARGET

        self.approach_pose = None
        self.handshake_pose = None
        self.plan_ready = False

        self.home_joint_positions = None

        self.cycle_approach_pose = None
        self.cycle_handshake_pose = None

        self.pending_plan_kind = None
        self.pending_execute_kind = None

        self.verify_good_count = 0

        result = Bool()
        result.data = False

        self.cycle_success_pub.publish(
            result
        )

        self.get_logger().info(
            "Automatic handshake cycle reset"
        )

    # ==============================================================
    # MoveIt constraints
    # ==============================================================

    def build_pose_constraints(
        self,
        target: PoseStamped
    ):

        constraints = Constraints()

        constraints.name = (
            "handshake_pose_target"
        )

        # Position
        pc = PositionConstraint()

        pc.header.frame_id = (
            self.base_frame
        )

        pc.link_name = (
            self.pose_link
        )

        pc.weight = 1.0

        sphere = SolidPrimitive()

        sphere.type = (
            SolidPrimitive.SPHERE
        )

        sphere.dimensions = [
            self.position_tolerance
        ]

        sphere_pose = Pose()

        sphere_pose.position = copy.deepcopy(
            target.pose.position
        )

        sphere_pose.orientation.w = 1.0

        pc.constraint_region.primitives.append(
            sphere
        )

        pc.constraint_region.primitive_poses.append(
            sphere_pose
        )

        # Orientation
        oc = OrientationConstraint()

        oc.header.frame_id = (
            self.base_frame
        )

        oc.link_name = (
            self.pose_link
        )

        oc.orientation = copy.deepcopy(
            target.pose.orientation
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

        constraints.position_constraints.append(
            pc
        )

        constraints.orientation_constraints.append(
            oc
        )

        return constraints

    def build_home_constraints(self):

        constraints = Constraints()

        constraints.name = (
            "captured_home_joint_state"
        )

        for name in self.arm_joint_names:

            jc = JointConstraint()

            jc.joint_name = name

            jc.position = float(
                self.home_joint_positions[name]
            )

            # roughly +/- 1 degree target tolerance
            jc.tolerance_above = math.radians(1.0)
            jc.tolerance_below = math.radians(1.0)
            jc.weight = 1.0

            constraints.joint_constraints.append(
                jc
            )

        return constraints

    # ==============================================================
    # Planning
    # ==============================================================

    def make_move_group_goal(
        self,
        constraints,
        velocity,
        acceleration
    ):

        goal = MoveGroup.Goal()

        # Use current real robot state.
        goal.request.start_state.is_diff = True

        goal.request.group_name = (
            self.planning_group
        )

        goal.request.num_planning_attempts = (
            self.planning_attempts
        )

        goal.request.allowed_planning_time = (
            self.planning_time
        )

        goal.request.max_velocity_scaling_factor = (
            velocity
        )

        goal.request.max_acceleration_scaling_factor = (
            acceleration
        )

        goal.request.goal_constraints = [
            constraints
        ]

        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False

        goal.planning_options.planning_scene_diff.is_diff = True

        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        return goal

    def plan_pose_target(
        self,
        kind,
        target,
        velocity,
        acceleration
    ):

        if not self.move_group_client.wait_for_server(
            timeout_sec=2.0
        ):

            self.fail(
                f"MoveIt action "
                f"{self.move_action_name} unavailable"
            )

            return

        if kind == "approach":
            self.state = self.PLANNING_APPROACH

        elif kind == "handshake":
            self.state = self.PLANNING_HANDSHAKE

        self.pending_plan_kind = kind

        constraints = (
            self.build_pose_constraints(
                target
            )
        )

        goal = self.make_move_group_goal(
            constraints,
            velocity,
            acceleration,
        )

        self.get_logger().info(
            f"Planning {kind} pose..."
        )

        future = (
            self.move_group_client.send_goal_async(
                goal
            )
        )

        future.add_done_callback(
            self.plan_goal_response_callback
        )

    def plan_home(self):

        if not self.move_group_client.wait_for_server(
            timeout_sec=2.0
        ):

            self.fail(
                f"MoveIt action "
                f"{self.move_action_name} unavailable"
            )

            return

        self.state = self.PLANNING_HOME
        self.pending_plan_kind = "home"

        goal = self.make_move_group_goal(
            self.build_home_constraints(),
            self.return_velocity,
            self.return_acceleration,
        )

        self.get_logger().info(
            "Planning return to captured HOME..."
        )

        future = (
            self.move_group_client.send_goal_async(
                goal
            )
        )

        future.add_done_callback(
            self.plan_goal_response_callback
        )

    def plan_goal_response_callback(
        self,
        future
    ):

        goal_handle = future.result()

        if not goal_handle.accepted:

            self.fail(
                f"MoveIt rejected "
                f"{self.pending_plan_kind} plan"
            )

            return

        result_future = (
            goal_handle.get_result_async()
        )

        result_future.add_done_callback(
            self.plan_result_callback
        )

    def plan_result_callback(
        self,
        future
    ):

        result = (
            future.result().result
        )

        error_code = int(
            result.error_code.val
        )

        kind = self.pending_plan_kind

        if error_code != 1:

            self.fail(
                f"MoveIt {kind} planning failed, "
                f"error_code={error_code}"
            )

            return

        trajectory = (
            result.planned_trajectory
        )

        points = (
            trajectory
            .joint_trajectory
            .points
        )

        if len(points) == 0:

            self.fail(
                f"MoveIt {kind} plan has no trajectory points"
            )

            return

        self.get_logger().info(
            f"{kind.upper()} PLAN SUCCESS: "
            f"{len(points)} trajectory points, "
            f"planning_time={result.planning_time:.3f}s"
        )

        # RViz display
        display = DisplayTrajectory()

        display.trajectory_start = (
            result.trajectory_start
        )

        display.trajectory.append(
            trajectory
        )

        self.display_pub.publish(
            display
        )

        # Automatically execute immediately after successful plan.
        self.execute_trajectory(
            kind,
            trajectory
        )

    # ==============================================================
    # Execution
    # ==============================================================

    def execute_trajectory(
        self,
        kind,
        trajectory
    ):

        if not self.enable_auto_motion:

            self.fail(
                "Auto execution blocked because "
                "enable_auto_motion=false"
            )

            return

        if not self.execute_client.wait_for_server(
            timeout_sec=2.0
        ):

            self.fail(
                f"ExecuteTrajectory action "
                f"{self.execute_action_name} unavailable"
            )

            return

        if kind == "approach":
            self.state = self.EXECUTING_APPROACH

        elif kind == "handshake":
            self.state = self.EXECUTING_HANDSHAKE

        elif kind == "home":
            self.state = self.EXECUTING_HOME

        self.pending_execute_kind = kind

        goal = ExecuteTrajectory.Goal()

        goal.trajectory = trajectory

        # Empty list => MoveIt selects configured controller(s)
        goal.controller_names = []

        self.get_logger().warning(
            f"Executing {kind.upper()} trajectory..."
        )

        future = (
            self.execute_client.send_goal_async(
                goal
            )
        )

        future.add_done_callback(
            self.execute_goal_response_callback
        )

    def execute_goal_response_callback(
        self,
        future
    ):

        goal_handle = future.result()

        if not goal_handle.accepted:

            self.fail(
                f"ExecuteTrajectory rejected "
                f"{self.pending_execute_kind}"
            )

            return

        result_future = (
            goal_handle.get_result_async()
        )

        result_future.add_done_callback(
            self.execute_result_callback
        )

    def execute_result_callback(
        self,
        future
    ):

        result = (
            future.result().result
        )

        error_code = int(
            result.error_code.val
        )

        kind = (
            self.pending_execute_kind
        )

        if error_code != 1:

            self.fail(
                f"{kind} execution failed, "
                f"MoveIt error_code={error_code}"
            )

            return

        self.get_logger().info(
            f"{kind.upper()} trajectory execution returned SUCCESS"
        )

        self.verify_good_count = 0
        self.verify_start_time = (
            time.monotonic()
        )

        if kind == "approach":

            self.state = (
                self.VERIFYING_APPROACH
            )

        elif kind == "handshake":

            self.state = (
                self.VERIFYING_HANDSHAKE
            )

        elif kind == "home":

            self.state = (
                self.VERIFYING_HOME
            )

    # ==============================================================
    # Verification
    # ==============================================================

    def get_current_gripper_pose(self):

        try:

            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.pose_link,
                Time(),
                timeout=Duration(
                    seconds=0.1
                )
            )

        except TransformException:
            return None

        p = np.array(
            [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ],
            dtype=np.float64
        )

        q = np.array(
            [
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w,
            ],
            dtype=np.float64
        )

        return p, q

    def verify_pose_target(
        self,
        target,
        position_tolerance,
        orientation_tolerance_deg
    ):

        current = (
            self.get_current_gripper_pose()
        )

        if current is None:
            return False, None, None

        current_p, current_q = current

        target_p = np.array(
            [
                target.pose.position.x,
                target.pose.position.y,
                target.pose.position.z,
            ],
            dtype=np.float64
        )

        target_q = np.array(
            [
                target.pose.orientation.x,
                target.pose.orientation.y,
                target.pose.orientation.z,
                target.pose.orientation.w,
            ],
            dtype=np.float64
        )

        pos_error = float(
            np.linalg.norm(
                target_p - current_p
            )
        )

        angle_error = (
            quat_angle_deg(
                target_q,
                current_q
            )
        )

        good = (
            pos_error
            <= position_tolerance
            and
            angle_error
            <= orientation_tolerance_deg
        )

        return (
            good,
            pos_error,
            angle_error
        )

    def verify_home(self):

        if (
            self.current_joint_state is None
            or
            self.home_joint_positions is None
        ):

            return False, None

        current_map = {
            name: float(position)
            for name, position
            in zip(
                self.current_joint_state.name,
                self.current_joint_state.position
            )
        }

        errors = []

        for name in self.arm_joint_names:

            if name not in current_map:
                return False, None

            errors.append(
                abs(
                    current_map[name]
                    -
                    self.home_joint_positions[name]
                )
            )

        max_error = float(
            max(errors)
        )

        return (
            max_error <= self.home_verify_joint,
            max_error
        )

    # ==============================================================
    # State timer
    # ==============================================================

    def timer_callback(self):

        # ----------------------------------------------------------
        # Approach precise verification
        # ----------------------------------------------------------

        if self.state == self.VERIFYING_APPROACH:

            good, pos_err, ang_err = (
                self.verify_pose_target(
                    self.cycle_approach_pose,
                    self.approach_verify_position,
                    self.approach_verify_orientation,
                )
            )

            self.handle_pose_verification(
                good=good,
                pos_err=pos_err,
                ang_err=ang_err,
                next_action="handshake",
            )

        # ----------------------------------------------------------
        # Final handshake precise verification
        # ----------------------------------------------------------

        elif self.state == self.VERIFYING_HANDSHAKE:

            good, pos_err, ang_err = (
                self.verify_pose_target(
                    self.cycle_handshake_pose,
                    self.handshake_verify_position,
                    self.handshake_verify_orientation,
                )
            )

            self.handle_pose_verification(
                good=good,
                pos_err=pos_err,
                ang_err=ang_err,
                next_action="dwell",
            )

        # ----------------------------------------------------------
        # Dwell at final non-contact pose
        # ----------------------------------------------------------

        elif self.state == self.DWELL:

            if (
                time.monotonic()
                - self.dwell_start_time
                >= self.handshake_dwell
            ):

                self.plan_home()

        # ----------------------------------------------------------
        # Verify HOME using arm joint angles
        # ----------------------------------------------------------

        elif self.state == self.VERIFYING_HOME:

            good, max_err = (
                self.verify_home()
            )

            if good:

                self.verify_good_count += 1

            else:

                self.verify_good_count = 0

            if (
                self.verify_good_count
                >= self.verify_stable_samples
            ):

                self.state = self.DONE

                result = Bool()
                result.data = True

                self.cycle_success_pub.publish(
                    result
                )

                self.get_logger().info(
                    "======================================"
                )

                self.get_logger().info(
                    "AUTO HANDSHAKE CYCLE COMPLETE"
                )

                self.get_logger().info(
                    f"Returned to captured HOME, "
                    f"max joint error="
                    f"{math.degrees(max_err):.2f} deg"
                )

                self.get_logger().info(
                    "Call /handshake/reset before next cycle."
                )

                self.get_logger().info(
                    "======================================"
                )

                return

            if (
                time.monotonic()
                - self.verify_start_time
                > self.verify_timeout
            ):

                self.fail(
                    "HOME verification timeout"
                )

    def handle_pose_verification(
        self,
        good,
        pos_err,
        ang_err,
        next_action
    ):

        if good:

            self.verify_good_count += 1

        else:

            self.verify_good_count = 0

        if (
            self.verify_good_count
            >= self.verify_stable_samples
        ):

            self.get_logger().info(
                f"Pose verified: "
                f"position error="
                f"{pos_err * 1000.0:.1f} mm, "
                f"orientation error="
                f"{ang_err:.2f} deg"
            )

            self.verify_good_count = 0

            if next_action == "handshake":

                self.plan_pose_target(
                    kind="handshake",
                    target=self.cycle_handshake_pose,
                    velocity=self.handshake_velocity,
                    acceleration=self.handshake_acceleration,
                )

            elif next_action == "dwell":

                self.state = self.DWELL

                self.dwell_start_time = (
                    time.monotonic()
                )

                self.get_logger().warning(
                    "Final non-contact handshake pose reached precisely."
                )

                self.get_logger().warning(
                    f"Dwelling for "
                    f"{self.handshake_dwell:.1f}s, "
                    f"then automatically returning HOME."
                )

            return

        if (
            time.monotonic()
            - self.verify_start_time
            > self.verify_timeout
        ):

            pos_text = (
                "N/A"
                if pos_err is None
                else f"{pos_err * 1000.0:.1f} mm"
            )

            ang_text = (
                "N/A"
                if ang_err is None
                else f"{ang_err:.2f} deg"
            )

            self.fail(
                f"Pose verification timeout; "
                f"last position error={pos_text}, "
                f"orientation error={ang_text}"
            )

    # ==============================================================
    # Failure
    # ==============================================================

    def fail(
        self,
        reason
    ):

        self.state = self.FAILED

        result = Bool()
        result.data = False

        self.cycle_success_pub.publish(
            result
        )

        self.get_logger().error(
            "======================================"
        )

        self.get_logger().error(
            "AUTO HANDSHAKE CYCLE FAILED"
        )

        self.get_logger().error(
            reason
        )

        self.get_logger().error(
            "No further automatic motion will be started."
        )

        self.get_logger().error(
            "======================================"
        )


def main(args=None):

    rclpy.init(args=args)

    node = (
        MoveItAutoHandshakeController()
    )

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
