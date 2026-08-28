import math
import time

import numpy as np
import rclpy

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, Empty
from std_srvs.srv import Trigger


class MoveItApproachController(Node):
    """
    STEP 4 + STEP 5

    STEP 4:
        /handshake/plan_approach
        -> MoveGroup(plan_only=True)
        -> store exact planned trajectory
        -> publish /display_planned_path

    STEP 5:
        /handshake/execute_approach
        -> safety checks
        -> ExecuteTrajectory with the EXACT stored trajectory
        -> robot moves only after explicit manual confirmation

    The approach target is never regenerated inside STEP 5.
    """

    def __init__(self):
        super().__init__("moveit_approach_controller")

        # ==========================================================
        # Parameters
        # ==========================================================

        self.declare_parameter(
            "planning_group",
            "arm"
        )

        self.declare_parameter(
            "pose_link",
            "gripper_base"
        )

        self.declare_parameter(
            "base_frame",
            "base_link"
        )

        self.declare_parameter(
            "move_action",
            "/move_action"
        )

        self.declare_parameter(
            "execute_action",
            "/execute_trajectory"
        )

        self.declare_parameter(
            "joint_feedback_topic",
            "/feedback/joint_states"
        )

        self.declare_parameter(
            "tcp_feedback_topic",
            "/feedback/tcp_pose"
        )

        self.declare_parameter(
            "planning_time_s",
            5.0
        )

        self.declare_parameter(
            "planning_attempts",
            5
        )

        self.declare_parameter(
            "position_tolerance_m",
            0.010
        )

        self.declare_parameter(
            "orientation_tolerance_deg",
            5.0
        )

        # 生成的轨迹速度限制
        self.declare_parameter(
            "velocity_scaling",
            0.10
        )

        self.declare_parameter(
            "acceleration_scaling",
            0.10
        )

        # ----------------------------------------------------------
        # STEP 5 safety
        # ----------------------------------------------------------

        # Plan 生成后超过这么久就不允许执行，需要重新规划
        self.declare_parameter(
            "max_plan_age_s",
            30.0
        )

        # 当前机器人和规划起点的最大关节差
        self.declare_parameter(
            "max_start_joint_error_deg",
            5.0
        )

        # ExecuteTrajectory 成功后，TCP距离 Approach 的允许误差
        self.declare_parameter(
            "final_position_tolerance_m",
            0.025
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

        self.tcp_feedback_topic = str(
            self.get_parameter(
                "tcp_feedback_topic"
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

        self.max_plan_age = float(
            self.get_parameter(
                "max_plan_age_s"
            ).value
        )

        self.max_start_joint_error = math.radians(
            float(
                self.get_parameter(
                    "max_start_joint_error_deg"
                ).value
            )
        )

        self.final_position_tolerance = float(
            self.get_parameter(
                "final_position_tolerance_m"
            ).value
        )

        # ==========================================================
        # State
        # ==========================================================

        self.approach_pose = None
        self.plan_ready = False

        self.current_joint_state = None
        self.current_tcp_pose = None

        self.request_running = False
        self.execution_running = False

        self.plan_success = False
        self.last_plan_time = 0.0

        self.planned_trajectory = None
        self.trajectory_start = None

        # ==========================================================
        # Subscribers
        # ==========================================================

        self.create_subscription(
            PoseStamped,
            "/handshake/approach_pose",
            self.approach_callback,
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
            PoseStamped,
            self.tcp_feedback_topic,
            self.tcp_pose_callback,
            10
        )

        self.create_subscription(
            Empty,
            "/handshake/reset_event",
            self.reset_callback,
            10
        )

        # ==========================================================
        # Services
        # ==========================================================

        self.plan_service = self.create_service(
            Trigger,
            "/handshake/plan_approach",
            self.plan_service_callback
        )

        self.execute_service = self.create_service(
            Trigger,
            "/handshake/execute_approach",
            self.execute_service_callback
        )

        # ==========================================================
        # Publishers
        # ==========================================================

        self.plan_success_pub = self.create_publisher(
            Bool,
            "/handshake/moveit_plan_success",
            10
        )

        self.execution_success_pub = self.create_publisher(
            Bool,
            "/handshake/approach_execution_success",
            10
        )

        self.display_pub = self.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            10
        )

        # ==========================================================
        # MoveIt action clients
        # ==========================================================

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

        self.get_logger().info(
            "MoveIt STEP 4+5 approach controller started"
        )

        self.get_logger().info(
            f"group={self.planning_group}, "
            f"pose_link={self.pose_link}, "
            f"base_frame={self.base_frame}"
        )

        self.get_logger().warning(
            "Execution requires explicit call to "
            "/handshake/execute_approach"
        )

    # ==============================================================
    # Input callbacks
    # ==============================================================

    def approach_callback(
        self,
        msg: PoseStamped
    ):

        # 一旦已有规划，不允许后台新消息悄悄替换执行目标。
        # 必须 reset -> 重新锁定 -> 重新规划。
        if self.plan_success:
            return

        self.approach_pose = msg

    def plan_ready_callback(
        self,
        msg: Bool
    ):

        self.plan_ready = bool(
            msg.data
        )

    def joint_state_callback(
        self,
        msg: JointState
    ):

        self.current_joint_state = msg

    def tcp_pose_callback(
        self,
        msg: PoseStamped
    ):

        self.current_tcp_pose = msg

    # ==============================================================
    # Reset
    # ==============================================================

    def reset_callback(
        self,
        msg: Empty
    ):

        if self.execution_running:

            self.get_logger().warning(
                "Reset received while trajectory execution "
                "is running; local plan state will be cleared "
                "after execution returns."
            )

        self.approach_pose = None
        self.plan_ready = False

        self.request_running = False

        self.plan_success = False
        self.last_plan_time = 0.0

        self.planned_trajectory = None
        self.trajectory_start = None

        plan_msg = Bool()
        plan_msg.data = False
        self.plan_success_pub.publish(
            plan_msg
        )

        execute_msg = Bool()
        execute_msg.data = False
        self.execution_success_pub.publish(
            execute_msg
        )

        self.get_logger().info(
            "MoveIt approach plan cleared"
        )

    # ==============================================================
    # Goal constraints
    # ==============================================================

    def build_goal_constraints(
        self,
        target: PoseStamped
    ):

        constraints = Constraints()
        constraints.name = (
            "handshake_approach_pose"
        )

        # ----------------------------------------------------------
        # Position
        # ----------------------------------------------------------

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

        sphere_pose.position.x = float(
            target.pose.position.x
        )

        sphere_pose.position.y = float(
            target.pose.position.y
        )

        sphere_pose.position.z = float(
            target.pose.position.z
        )

        sphere_pose.orientation.w = 1.0

        pc.constraint_region.primitives.append(
            sphere
        )

        pc.constraint_region.primitive_poses.append(
            sphere_pose
        )

        # ----------------------------------------------------------
        # Orientation
        # ----------------------------------------------------------

        oc = OrientationConstraint()

        oc.header.frame_id = (
            self.base_frame
        )

        oc.link_name = (
            self.pose_link
        )

        oc.orientation = (
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

    # ==============================================================
    # STEP 4 - Plan only
    # ==============================================================

    def plan_service_callback(
        self,
        request,
        response
    ):

        if self.execution_running:

            response.success = False
            response.message = (
                "Trajectory execution is running"
            )

            return response

        if self.request_running:

            response.success = False
            response.message = (
                "Planning request already running"
            )

            return response

        if not self.plan_ready:

            response.success = False
            response.message = (
                "Handshake target is not ready"
            )

            return response

        if self.approach_pose is None:

            response.success = False
            response.message = (
                "No approach pose received"
            )

            return response

        if (
            self.approach_pose.header.frame_id
            and
            self.approach_pose.header.frame_id
            != self.base_frame
        ):

            response.success = False
            response.message = (
                f"Approach pose frame is "
                f"'{self.approach_pose.header.frame_id}', "
                f"expected '{self.base_frame}'"
            )

            return response

        if not self.move_group_client.wait_for_server(
            timeout_sec=2.0
        ):

            response.success = False
            response.message = (
                f"MoveIt action server "
                f"'{self.move_action_name}' unavailable"
            )

            return response

        # 清除上一份规划
        self.plan_success = False
        self.planned_trajectory = None
        self.trajectory_start = None
        self.last_plan_time = 0.0

        goal_msg = MoveGroup.Goal()

        # 使用 MoveIt 当前机器人状态作为起点
        goal_msg.request.start_state.is_diff = True

        goal_msg.request.group_name = (
            self.planning_group
        )

        goal_msg.request.num_planning_attempts = (
            self.planning_attempts
        )

        goal_msg.request.allowed_planning_time = (
            self.planning_time
        )

        goal_msg.request.max_velocity_scaling_factor = (
            self.velocity_scaling
        )

        goal_msg.request.max_acceleration_scaling_factor = (
            self.acceleration_scaling
        )

        goal_msg.request.goal_constraints = [
            self.build_goal_constraints(
                self.approach_pose
            )
        ]

        # STEP 4核心：只规划
        goal_msg.planning_options.plan_only = True

        goal_msg.planning_options.look_around = False
        goal_msg.planning_options.replan = False

        goal_msg.planning_options.planning_scene_diff.is_diff = True

        goal_msg.planning_options.planning_scene_diff.robot_state.is_diff = True

        p = self.approach_pose.pose.position
        q = self.approach_pose.pose.orientation

        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            "MOVEIT APPROACH PLAN REQUEST"
        )

        self.get_logger().info(
            f"Position = "
            f"[{p.x:+.4f}, "
            f"{p.y:+.4f}, "
            f"{p.z:+.4f}] m"
        )

        self.get_logger().info(
            f"Quaternion xyzw = "
            f"[{q.x:+.4f}, "
            f"{q.y:+.4f}, "
            f"{q.z:+.4f}, "
            f"{q.w:+.4f}]"
        )

        self.get_logger().info(
            "plan_only = TRUE"
        )

        self.get_logger().info(
            "======================================"
        )

        self.request_running = True

        send_future = (
            self.move_group_client.send_goal_async(
                goal_msg
            )
        )

        send_future.add_done_callback(
            self.plan_goal_response_callback
        )

        response.success = True

        response.message = (
            "MoveIt plan-only request submitted"
        )

        return response

    def plan_goal_response_callback(
        self,
        future
    ):

        goal_handle = future.result()

        if not goal_handle.accepted:

            self.request_running = False

            msg = Bool()
            msg.data = False

            self.plan_success_pub.publish(
                msg
            )

            self.get_logger().error(
                "MoveIt rejected planning request"
            )

            return

        self.get_logger().info(
            "MoveIt accepted planning request"
        )

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

        self.request_running = False

        wrapped = future.result()
        result = wrapped.result

        error_code = int(
            result.error_code.val
        )

        success = (
            error_code == 1
        )

        success_msg = Bool()
        success_msg.data = success

        self.plan_success_pub.publish(
            success_msg
        )

        if not success:

            self.plan_success = False

            self.get_logger().error(
                f"MOVEIT PLAN FAILED, "
                f"MoveIt error code = {error_code}"
            )

            return

        # ==========================================================
        # Store EXACT verified plan for STEP 5
        # ==========================================================

        self.plan_success = True

        self.planned_trajectory = (
            result.planned_trajectory
        )

        self.trajectory_start = (
            result.trajectory_start
        )

        self.last_plan_time = (
            time.monotonic()
        )

        joint_names = (
            self.planned_trajectory
            .joint_trajectory
            .joint_names
        )

        points = (
            self.planned_trajectory
            .joint_trajectory
            .points
        )

        self.get_logger().info(
            "======================================"
        )

        self.get_logger().info(
            "MOVEIT PLAN SUCCESS"
        )

        self.get_logger().info(
            f"Planning time = "
            f"{result.planning_time:.3f} s"
        )

        self.get_logger().info(
            f"Trajectory joints = "
            f"{list(joint_names)}"
        )

        self.get_logger().info(
            f"Trajectory points = "
            f"{len(points)}"
        )

        self.get_logger().warning(
            "Trajectory STORED but NOT executed."
        )

        self.get_logger().warning(
            "Inspect RViz, then call "
            "/handshake/execute_approach manually."
        )

        self.get_logger().info(
            "======================================"
        )

        # ----------------------------------------------------------
        # RViz trajectory
        # ----------------------------------------------------------

        display = DisplayTrajectory()

        display.trajectory_start = (
            result.trajectory_start
        )

        display.trajectory.append(
            result.planned_trajectory
        )

        self.display_pub.publish(
            display
        )

    # ==============================================================
    # STEP 5 safety checks
    # ==============================================================

    def check_plan_before_execute(
        self
    ):

        if not self.plan_success:

            return (
                False,
                "No successful MoveIt plan is stored"
            )

        if self.planned_trajectory is None:

            return (
                False,
                "Stored trajectory is empty"
            )

        points = (
            self.planned_trajectory
            .joint_trajectory
            .points
        )

        if len(points) == 0:

            return (
                False,
                "Stored trajectory contains no points"
            )

        # ----------------------------------------------------------
        # Plan age
        # ----------------------------------------------------------

        age = (
            time.monotonic()
            - self.last_plan_time
        )

        if age > self.max_plan_age:

            return (
                False,
                (
                    f"Plan is too old: "
                    f"{age:.1f}s > "
                    f"{self.max_plan_age:.1f}s. "
                    "Please re-plan."
                )
            )

        # ----------------------------------------------------------
        # Current joint state vs planned start state
        # ----------------------------------------------------------

        if self.current_joint_state is None:

            return (
                False,
                "No current joint feedback available"
            )

        if self.trajectory_start is None:

            return (
                False,
                "No stored trajectory start state"
            )

        planned_start = (
            self.trajectory_start
            .joint_state
        )

        current_map = {
            name: position
            for name, position
            in zip(
                self.current_joint_state.name,
                self.current_joint_state.position
            )
        }

        errors = []

        for name, planned_pos in zip(
            planned_start.name,
            planned_start.position
        ):

            if name not in current_map:
                continue

            err = abs(
                float(current_map[name])
                -
                float(planned_pos)
            )

            errors.append(
                (
                    name,
                    err
                )
            )

        if not errors:

            return (
                False,
                (
                    "Could not match current joint names "
                    "with trajectory start joint names"
                )
            )

        max_joint_name, max_error = max(
            errors,
            key=lambda item: item[1]
        )

        if (
            max_error
            > self.max_start_joint_error
        ):

            return (
                False,
                (
                    f"Robot moved since planning. "
                    f"Max start-state error: "
                    f"{max_joint_name}="
                    f"{math.degrees(max_error):.2f}deg > "
                    f"{math.degrees(self.max_start_joint_error):.2f}deg. "
                    "Please re-plan."
                )
            )

        return (
            True,
            (
                f"Stored plan valid; age={age:.1f}s, "
                f"max start joint error="
                f"{math.degrees(max_error):.2f}deg"
            )
        )

    # ==============================================================
    # STEP 5 - Execute EXACT stored trajectory
    # ==============================================================

    def execute_service_callback(
        self,
        request,
        response
    ):

        if self.execution_running:

            response.success = False
            response.message = (
                "Trajectory execution already running"
            )

            return response

        if self.request_running:

            response.success = False
            response.message = (
                "Planning request is still running"
            )

            return response

        safe, reason = (
            self.check_plan_before_execute()
        )

        if not safe:

            self.get_logger().error(
                f"EXECUTION REJECTED: {reason}"
            )

            response.success = False
            response.message = reason

            return response

        if not self.execute_client.wait_for_server(
            timeout_sec=2.0
        ):

            response.success = False
            response.message = (
                f"MoveIt ExecuteTrajectory action "
                f"'{self.execute_action_name}' unavailable"
            )

            return response

        self.get_logger().warning(
            "======================================"
        )

        self.get_logger().warning(
            "REAL ROBOT APPROACH EXECUTION REQUESTED"
        )

        self.get_logger().warning(
            reason
        )

        self.get_logger().warning(
            "Executing the EXACT trajectory "
            "that was inspected in STEP 4."
        )

        self.get_logger().warning(
            "======================================"
        )

        goal = ExecuteTrajectory.Goal()

        goal.trajectory = (
            self.planned_trajectory
        )

        # 空列表表示让 MoveIt 使用配置好的默认 controller
        goal.controller_names = []

        self.execution_running = True

        send_future = (
            self.execute_client.send_goal_async(
                goal
            )
        )

        send_future.add_done_callback(
            self.execute_goal_response_callback
        )

        response.success = True

        response.message = (
            "Stored MoveIt trajectory submitted for execution"
        )

        return response

    def execute_goal_response_callback(
        self,
        future
    ):

        goal_handle = future.result()

        if not goal_handle.accepted:

            self.execution_running = False

            msg = Bool()
            msg.data = False

            self.execution_success_pub.publish(
                msg
            )

            self.get_logger().error(
                "ExecuteTrajectory goal rejected"
            )

            return

        self.get_logger().warning(
            "ExecuteTrajectory accepted - robot moving"
        )

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

        self.execution_running = False

        wrapped = future.result()
        result = wrapped.result

        error_code = int(
            result.error_code.val
        )

        moveit_success = (
            error_code == 1
        )

        # ----------------------------------------------------------
        # Extra TCP verification
        # ----------------------------------------------------------

        tcp_error = None

        if (
            moveit_success
            and
            self.current_tcp_pose is not None
            and
            self.approach_pose is not None
        ):

            current = np.array(
                [
                    self.current_tcp_pose.pose.position.x,
                    self.current_tcp_pose.pose.position.y,
                    self.current_tcp_pose.pose.position.z,
                ],
                dtype=np.float64
            )

            target = np.array(
                [
                    self.approach_pose.pose.position.x,
                    self.approach_pose.pose.position.y,
                    self.approach_pose.pose.position.z,
                ],
                dtype=np.float64
            )

            tcp_error = float(
                np.linalg.norm(
                    target - current
                )
            )

        success = moveit_success

        if (
            tcp_error is not None
            and
            tcp_error > self.final_position_tolerance
        ):

            success = False

        msg = Bool()
        msg.data = success

        self.execution_success_pub.publish(
            msg
        )

        self.get_logger().info(
            "======================================"
        )

        if success:

            self.get_logger().info(
                "STEP 5 APPROACH EXECUTION SUCCESS"
            )

            if tcp_error is not None:

                self.get_logger().info(
                    f"Final TCP position error = "
                    f"{tcp_error * 1000.0:.1f} mm"
                )

            self.get_logger().info(
                "Robot is now at the Approach pose."
            )

            self.get_logger().info(
                "DO NOT continue toward the hand yet."
            )

        else:

            self.get_logger().error(
                "STEP 5 APPROACH EXECUTION FAILED"
            )

            self.get_logger().error(
                f"MoveIt error code = {error_code}"
            )

            if tcp_error is not None:

                self.get_logger().error(
                    f"Final TCP position error = "
                    f"{tcp_error * 1000.0:.1f} mm"
                )

        self.get_logger().info(
            "======================================"
        )


def main(args=None):

    rclpy.init(args=args)

    node = MoveItApproachController()

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
