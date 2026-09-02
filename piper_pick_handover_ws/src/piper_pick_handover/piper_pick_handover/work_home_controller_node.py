#!/usr/bin/env python3
"""
Pure MoveIt WORK_HOME controller for PIPER-X.

This node intentionally does NOT:
- publish /control/joint_states
- call /control_enable directly
- provide /work_home/startup

All WORK_HOME motion is handled through MoveIt:
    current real robot state
        -> MoveGroup joint-space planning
        -> ExecuteTrajectory
        -> real robot
        -> feedback verification

Services:
    /work_home/plan
    /work_home/execute

Requirement:
    MoveIt must already be running, and the PIPER MoveIt bringup should use
    automatic control-gate handling (auto_control_gate:=true).
"""

import math
import threading
import time

import rclpy
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    MoveItErrorCodes,
)
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


class WorkHomeControllerNode(Node):
    """Pure-MoveIt WORK_HOME controller."""

    def __init__(self):
        super().__init__("work_home_controller_node")

        self.cb_group = ReentrantCallbackGroup()
        self.operation_lock = threading.Lock()
        self.feedback_lock = threading.Lock()

        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------
        self.declare_parameter(
            "planning_group",
            "arm",
        )

        self.declare_parameter(
            "joint_names",
            [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ],
        )

        # User's current WORK_HOME joint angles [rad].
        self.declare_parameter(
            "work_home_positions",
            [
                0.5119050696099369,
                0.48549823802726266,
                -0.7454475768192981,
                1.072312839132796,
                0.12866567245702198,
                -0.11885692206081386,
            ],
        )

        self.declare_parameter(
            "feedback_topic",
            "/feedback/joint_states",
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
            "display_topic",
            "/display_planned_path",
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
            0.08,
        )

        self.declare_parameter(
            "acceleration_scaling",
            0.05,
        )

        self.declare_parameter(
            "joint_goal_tolerance_rad",
            0.015,
        )

        self.declare_parameter(
            "post_execute_tolerance_rad",
            0.030,
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

        # True by default because this node is now specifically intended
        # to be the real WORK_HOME execution interface.
        self.declare_parameter(
            "allow_work_home_execute",
            True,
        )

        # --------------------------------------------------------------
        # Read parameters
        # --------------------------------------------------------------
        self.planning_group = str(
            self.get_parameter("planning_group").value
        )

        self.joint_names = list(
            self.get_parameter("joint_names").value
        )

        self.work_home = [
            float(v)
            for v in self.get_parameter("work_home_positions").value
        ]

        self.feedback_topic = str(
            self.get_parameter("feedback_topic").value
        )

        self.move_group_action = str(
            self.get_parameter("move_group_action").value
        )

        self.execute_action = str(
            self.get_parameter("execute_action").value
        )

        self.display_topic = str(
            self.get_parameter("display_topic").value
        )

        self.allowed_planning_time = float(
            self.get_parameter("allowed_planning_time").value
        )

        self.num_planning_attempts = int(
            self.get_parameter("num_planning_attempts").value
        )

        self.velocity_scaling = float(
            self.get_parameter("velocity_scaling").value
        )

        self.acceleration_scaling = float(
            self.get_parameter("acceleration_scaling").value
        )

        self.goal_tolerance = float(
            self.get_parameter("joint_goal_tolerance_rad").value
        )

        self.post_tolerance = float(
            self.get_parameter("post_execute_tolerance_rad").value
        )

        self.action_server_timeout = float(
            self.get_parameter("action_server_timeout_sec").value
        )

        self.plan_result_timeout = float(
            self.get_parameter("plan_result_timeout_sec").value
        )

        self.execute_result_timeout = float(
            self.get_parameter("execute_result_timeout_sec").value
        )

        self.allow_execute = bool(
            self.get_parameter("allow_work_home_execute").value
        )

        if len(self.joint_names) != 6 or len(self.work_home) != 6:
            raise ValueError(
                "WORK_HOME must contain exactly 6 arm joints."
            )

        # --------------------------------------------------------------
        # Real robot feedback
        # --------------------------------------------------------------
        self.current = {}

        self.create_subscription(
            JointState,
            self.feedback_topic,
            self._feedback_cb,
            20,
            callback_group=self.cb_group,
        )

        # --------------------------------------------------------------
        # RViz planned trajectory display
        # --------------------------------------------------------------
        self.display_pub = self.create_publisher(
            DisplayTrajectory,
            self.display_topic,
            10,
        )

        # --------------------------------------------------------------
        # MoveIt action clients
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Services
        # --------------------------------------------------------------
        self.create_service(
            Trigger,
            "/work_home/plan",
            self._plan_cb,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/work_home/execute",
            self._execute_cb,
            callback_group=self.cb_group,
        )

        self.get_logger().info(
            "Pure MoveIt WORK_HOME controller started."
        )
        self.get_logger().info(
            "WORK_HOME = ["
            + ", ".join(f"{v:+.6f}" for v in self.work_home)
            + "] rad"
        )
        self.get_logger().info(
            "Services: /work_home/plan, /work_home/execute"
        )
        self.get_logger().info(
            "Direct /control/joint_states control is disabled in this node."
        )
        self.get_logger().info(
            "Real execution relies on MoveIt + automatic control Gate."
        )

    # ==================================================================
    # Feedback
    # ==================================================================

    def _feedback_cb(self, msg):
        with self.feedback_lock:
            for idx, name in enumerate(msg.name):
                if idx < len(msg.position):
                    self.current[str(name)] = float(msg.position[idx])

    def _current_arm(self):
        with self.feedback_lock:
            result = []

            for name in self.joint_names:
                if name not in self.current:
                    return None

                result.append(self.current[name])

            return result

    # ==================================================================
    # Async wait helper
    # ==================================================================

    @staticmethod
    def _wait_future(future, timeout_sec):
        """
        Wait without recursively spinning this node.

        The node itself runs in a MultiThreadedExecutor, so other executor
        threads can continue processing action responses while this callback
        waits on the Event.
        """
        event = threading.Event()

        future.add_done_callback(
            lambda _future: event.set()
        )

        if not event.wait(max(0.05, float(timeout_sec))):
            return None

        try:
            return future.result()
        except Exception:
            return None

    # ==================================================================
    # MoveIt planning
    # ==================================================================

    def _build_goal(self):
        goal = MoveGroup.Goal()
        req = goal.request

        req.group_name = self.planning_group
        req.num_planning_attempts = max(
            1,
            self.num_planning_attempts,
        )
        req.allowed_planning_time = self.allowed_planning_time
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.acceleration_scaling

        # Use MoveIt's currently monitored real-arm state as the start state.
        req.start_state.is_diff = True

        constraints = Constraints()

        for name, target in zip(
            self.joint_names,
            self.work_home,
        ):
            jc = JointConstraint()
            jc.joint_name = str(name)
            jc.position = float(target)
            jc.tolerance_above = self.goal_tolerance
            jc.tolerance_below = self.goal_tolerance
            jc.weight = 1.0

            constraints.joint_constraints.append(jc)

        req.goal_constraints = [constraints]

        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False

        return goal

    def _plan_internal(self):
        if not self.move_group_client.wait_for_server(
            timeout_sec=self.action_server_timeout
        ):
            return (
                False,
                f"MoveGroup action unavailable: {self.move_group_action}",
                None,
            )

        send_future = self.move_group_client.send_goal_async(
            self._build_goal()
        )

        goal_handle = self._wait_future(
            send_future,
            self.action_server_timeout,
        )

        if goal_handle is None or not goal_handle.accepted:
            return (
                False,
                "MoveGroup rejected WORK_HOME goal.",
                None,
            )

        wrapped = self._wait_future(
            goal_handle.get_result_async(),
            self.plan_result_timeout,
        )

        if wrapped is None:
            return (
                False,
                "WORK_HOME planning timed out.",
                None,
            )

        result = wrapped.result
        code = int(result.error_code.val)

        if code != MoveItErrorCodes.SUCCESS:
            return (
                False,
                "WORK_HOME planning failed; "
                f"MoveIt error_code={code}.",
                None,
            )

        trajectory = result.planned_trajectory

        if len(trajectory.joint_trajectory.points) == 0:
            return (
                False,
                "WORK_HOME trajectory has no points.",
                None,
            )

        display = DisplayTrajectory()
        display.trajectory_start = result.trajectory_start
        display.trajectory = [trajectory]
        self.display_pub.publish(display)

        return (
            True,
            "WORK_HOME PLAN OK | "
            f"points={len(trajectory.joint_trajectory.points)}",
            trajectory,
        )

    # ==================================================================
    # MoveIt execution
    # ==================================================================

    def _execute_trajectory(self, trajectory):
        if not self.execute_client.wait_for_server(
            timeout_sec=self.action_server_timeout
        ):
            return (
                False,
                f"ExecuteTrajectory action unavailable: {self.execute_action}",
            )

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory

        send_future = self.execute_client.send_goal_async(goal)

        handle = self._wait_future(
            send_future,
            self.action_server_timeout,
        )

        if handle is None or not handle.accepted:
            return (
                False,
                "WORK_HOME trajectory rejected.",
            )

        wrapped = self._wait_future(
            handle.get_result_async(),
            self.execute_result_timeout,
        )

        if wrapped is None:
            return (
                False,
                "WORK_HOME execution timed out.",
            )

        code = int(wrapped.result.error_code.val)

        if code != MoveItErrorCodes.SUCCESS:
            return (
                False,
                "WORK_HOME execution failed; "
                f"MoveIt error_code={code}.",
            )

        return (
            True,
            "trajectory execution succeeded",
        )

    # ==================================================================
    # Real robot verification
    # ==================================================================

    def _verify(self):
        current = self._current_arm()

        if current is None:
            return (
                False,
                "no complete real joint feedback",
            )

        errors = [
            abs(float(now) - float(target))
            for now, target in zip(
                current,
                self.work_home,
            )
        ]

        maximum = max(errors)

        return (
            maximum <= self.post_tolerance,
            (
                f"max_joint_error={maximum:.6f} rad "
                f"({math.degrees(maximum):.2f} deg)"
            ),
        )

    # ==================================================================
    # ROS services
    # ==================================================================

    def _plan_cb(self, request, response):
        del request

        if not self.operation_lock.acquire(blocking=False):
            response.success = False
            response.message = (
                "WORK_HOME operation already in progress."
            )
            return response

        try:
            ok, msg, _ = self._plan_internal()

            response.success = bool(ok)
            response.message = msg
            return response

        finally:
            self.operation_lock.release()

    def _execute_cb(self, request, response):
        del request

        if not self.allow_execute:
            response.success = False
            response.message = (
                "WORK_HOME execution blocked: "
                "allow_work_home_execute=false."
            )
            return response

        if not self.operation_lock.acquire(blocking=False):
            response.success = False
            response.message = (
                "WORK_HOME operation already in progress."
            )
            return response

        try:
            # Plan from the current monitored real robot state.
            ok, msg, trajectory = self._plan_internal()

            if not ok:
                response.success = False
                response.message = msg
                return response

            self.get_logger().warning(
                "EXECUTING REAL ROBOT -> WORK_HOME via MoveIt."
            )

            ok, msg = self._execute_trajectory(trajectory)

            if not ok:
                response.success = False
                response.message = msg
                return response

            # Allow real feedback to settle after trajectory completion.
            time.sleep(0.25)

            verified, detail = self._verify()

            response.success = bool(verified)

            if verified:
                response.message = (
                    "WORK_HOME VERIFIED | " + detail
                )
            else:
                response.message = (
                    "WORK_HOME execution completed, "
                    "but verification failed | "
                    + detail
                )

            return response

        finally:
            self.operation_lock.release()


def main(args=None):
    rclpy.init(args=args)

    node = WorkHomeControllerNode()

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
