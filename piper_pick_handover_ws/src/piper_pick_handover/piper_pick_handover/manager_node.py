#!/usr/bin/env python3
"""
PIPER Pick & Handover task manager.

Final task architecture:

    START
      -> reset target locks
      -> WORK_HOME (custom project pose, via MoveIt)
      -> detect/lock CASE ONLY
      -> stop perception
      -> create table collision
      -> PreGrasp
      -> open gripper
      -> descend to Grasp
      -> grasp object + verify
      -> Lift
      -> DRIVER MECHANICAL HOME (/move_home)
         joint1..joint6 -> mechanical zero
         gripper remains holding the object
      -> verify real joint1..joint6 are near zero
      -> reset old handover plan
      -> arm Palm Final
      -> start perception
      -> detect/lock fresh Palm Final
      -> stop perception
      -> recompute handover from current mechanical-HOME gripper TF
      -> Handover Approach
      -> Handover Final
      -> release
      -> WORK_HOME (custom project pose, via MoveIt)
      -> remove table
      -> DONE

Important:
- No Palm Initial is used.
- No Observe-Hand planner/motion is used.
- /move_home is the official agx_arm_ctrl std_srvs/Empty service.
- After /move_home returns, manager still verifies real joint feedback before
  starting hand perception.
- No nested rclpy.spin_until_future_complete().
"""

import enum
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Empty, Trigger


class State(enum.Enum):
    IDLE = 0

    RESET_TARGETS = 1
    WORK_HOME_EMPTY = 2

    START_CASE_OBSERVATION = 3
    WAIT_CASE = 4
    STOP_CASE_OBSERVATION = 5

    CREATE_TABLE = 6
    WAIT_GRASP_PLAN = 7

    PREGRASP = 8
    OPEN_GRIPPER = 9
    DESCEND_GRASP = 10
    GRASP_OBJECT = 11
    WAIT_OBJECT_GRASPED = 12
    LIFT = 13

    MOVE_MECHANICAL_HOME = 14
    WAIT_MECHANICAL_HOME = 15

    RESET_HANDOVER = 16
    ARM_PALM_FINAL = 17
    START_HAND_OBSERVATION = 18
    WAIT_PALM_FINAL = 19
    STOP_HAND_OBSERVATION = 20
    RECOMPUTE_HANDOVER = 21

    HANDOVER_APPROACH = 22
    HANDOVER_FINAL = 23
    RELEASE = 24

    RETURN_WORK_HOME_EMPTY = 25
    REMOVE_TABLE_END = 26

    DONE = 27
    ERROR = 28


class PiperManager(Node):

    def __init__(self):
        super().__init__("piper_manager_node")

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        self.declare_parameter("loop_period_sec", 0.10)
        self.declare_parameter("service_wait_timeout_sec", 5.0)

        self.declare_parameter("wait_case_timeout_sec", 30.0)
        self.declare_parameter("wait_grasp_plan_timeout_sec", 10.0)
        self.declare_parameter("wait_object_grasped_timeout_sec", 8.0)
        self.declare_parameter("wait_palm_final_timeout_sec", 30.0)

        # Mechanical HOME verification:
        # joint1..joint6 must all be close to 0 rad.
        self.declare_parameter(
            "mechanical_home_joint_tolerance_rad",
            0.03,
        )
        self.declare_parameter(
            "mechanical_home_timeout_sec",
            20.0,
        )
        self.declare_parameter(
            "mechanical_home_settle_sec",
            0.50,
        )

        self.loop_period = float(
            self.get_parameter("loop_period_sec").value
        )
        self.service_wait_timeout = float(
            self.get_parameter("service_wait_timeout_sec").value
        )
        self.case_timeout = float(
            self.get_parameter("wait_case_timeout_sec").value
        )
        self.grasp_plan_timeout = float(
            self.get_parameter("wait_grasp_plan_timeout_sec").value
        )
        self.object_grasped_timeout = float(
            self.get_parameter("wait_object_grasped_timeout_sec").value
        )
        self.palm_final_timeout = float(
            self.get_parameter("wait_palm_final_timeout_sec").value
        )

        self.mechanical_home_tolerance = float(
            self.get_parameter(
                "mechanical_home_joint_tolerance_rad"
            ).value
        )
        self.mechanical_home_timeout = float(
            self.get_parameter(
                "mechanical_home_timeout_sec"
            ).value
        )
        self.mechanical_home_settle_sec = float(
            self.get_parameter(
                "mechanical_home_settle_sec"
            ).value
        )

        # ------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------
        self.state = State.IDLE
        self.running = False
        self.state_entered_at = time.monotonic()

        self.run_id = 0
        self.pending_future = None
        self.pending_key = None

        self.case_locked = False
        self.grasp_plan_ready = False
        self.object_grasped = False
        self.palm_final_locked = False
        self.handover_plan_ready = False

        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        self.current_joint_map = {}
        self.mechanical_home_in_tolerance_since = None

        # ------------------------------------------------------------
        # Trigger service interfaces
        # ------------------------------------------------------------
        trigger_service_map = {
            "targets_reset":
                "/targets/reset",

            "work_home":
                "/work_home/execute",

            "start_observation":
                "/perception/start_observation",

            "stop_observation":
                "/perception/stop_observation",

            "create_table":
                "/scene/apply_table_from_case",

            "remove_table":
                "/scene/remove_table",

            "pregrasp":
                "/moveit/execute_pregrasp",

            "gripper_open":
                "/moveit/execute_gripper_open",

            "grasp_preview":
                "/moveit/execute_grasp_preview",

            "gripper_grasp":
                "/moveit/execute_gripper_grasp",

            "lift":
                "/moveit/execute_lift",

            "handover_reset":
                "/handover/reset_plan",

            "arm_palm_final":
                "/targets/arm_palm_final",

            "handover_recompute":
                "/handover/recompute_plan",

            "handover_approach":
                "/moveit/execute_handover_approach",

            "handover_final":
                "/moveit/execute_handover_final",
        }

        self.service_clients = {
            key: self.create_client(
                Trigger,
                service_name,
            )
            for key, service_name
            in trigger_service_map.items()
        }
        self.service_names = dict(
            trigger_service_map
        )

        # Official agx_arm_ctrl mechanical-home service.
        self.move_home_service = "/move_home"
        self.move_home_client = self.create_client(
            Empty,
            self.move_home_service,
        )

        # ------------------------------------------------------------
        # Readiness/status topics
        # ------------------------------------------------------------
        self.create_subscription(
            Bool,
            "/targets/case_locked",
            self._case_locked_cb,
            10,
        )

        self.create_subscription(
            Bool,
            "/pick/grasp_plan_ready",
            self._grasp_plan_ready_cb,
            10,
        )

        self.create_subscription(
            Bool,
            "/pick/object_grasped",
            self._object_grasped_cb,
            10,
        )

        self.create_subscription(
            Bool,
            "/targets/palm_final_locked",
            self._palm_final_locked_cb,
            10,
        )

        self.create_subscription(
            Bool,
            "/handover/plan_ready",
            self._handover_plan_ready_cb,
            10,
        )

        self.create_subscription(
            JointState,
            "/feedback/joint_states",
            self._joint_state_cb,
            20,
        )

        # ------------------------------------------------------------
        # Manager API
        # ------------------------------------------------------------
        self.create_service(
            Trigger,
            "/manager/start",
            self._start_cb,
        )

        self.create_service(
            Trigger,
            "/manager/stop",
            self._stop_cb,
        )

        self.create_service(
            Trigger,
            "/manager/status",
            self._status_cb,
        )

        self.timer = self.create_timer(
            self.loop_period,
            self._loop,
        )

        self.get_logger().info(
            "PIPER manager ready."
        )
        self.get_logger().info(
            "Flow: CASE-only pick -> Lift -> /move_home mechanical zero -> "
            "fresh Palm Final -> Handover."
        )
        self.get_logger().info(
            "Initial/final project pose remains /work_home/execute."
        )

    # ================================================================
    # Topic callbacks
    # ================================================================

    def _case_locked_cb(self, msg):
        self.case_locked = bool(
            msg.data
        )

    def _grasp_plan_ready_cb(self, msg):
        self.grasp_plan_ready = bool(
            msg.data
        )

    def _object_grasped_cb(self, msg):
        self.object_grasped = bool(
            msg.data
        )

    def _palm_final_locked_cb(self, msg):
        self.palm_final_locked = bool(
            msg.data
        )

    def _handover_plan_ready_cb(self, msg):
        self.handover_plan_ready = bool(
            msg.data
        )

    def _joint_state_cb(self, msg):
        for index, name in enumerate(
            msg.name
        ):
            if index < len(msg.position):
                self.current_joint_map[
                    str(name)
                ] = float(
                    msg.position[index]
                )

    # ================================================================
    # Manager services
    # ================================================================

    def _start_cb(
        self,
        request,
        response,
    ):
        del request

        if self.running:
            response.success = False
            response.message = (
                f"Task already running; state={self.state.name}"
            )
            return response

        self.run_id += 1

        self.pending_future = None
        self.pending_key = None

        self.case_locked = False
        self.grasp_plan_ready = False
        self.object_grasped = False
        self.palm_final_locked = False
        self.handover_plan_ready = False

        self.mechanical_home_in_tolerance_since = None

        self.running = True
        self._set_state(
            State.RESET_TARGETS
        )

        response.success = True
        response.message = (
            "Task accepted. "
            "CASE pick -> Lift -> mechanical HOME -> Palm Final -> Handover."
        )

        self.get_logger().warning(
            "=================================================="
        )
        self.get_logger().warning(
            "AUTOMATIC TASK STARTED - REAL ROBOT MAY MOVE"
        )
        self.get_logger().warning(
            "=================================================="
        )

        return response

    def _stop_cb(
        self,
        request,
        response,
    ):
        del request

        self.run_id += 1
        self.running = False

        self.pending_future = None
        self.pending_key = None

        self.mechanical_home_in_tolerance_since = None

        self._set_state(
            State.IDLE
        )

        response.success = True
        response.message = (
            "Manager task stopped. "
            "This does not cancel a motion already executing."
        )

        return response

    def _status_cb(
        self,
        request,
        response,
    ):
        del request

        current = self._current_arm_positions()

        current_text = (
            "incomplete"
            if current is None
            else "["
            + ", ".join(
                f"{v:+.4f}"
                for v in current
            )
            + "]"
        )

        response.success = True
        response.message = (
            f"state={self.state.name}, "
            f"running={self.running}, "
            f"service_pending={self.pending_key}, "
            f"case_locked={self.case_locked}, "
            f"grasp_plan_ready={self.grasp_plan_ready}, "
            f"object_grasped={self.object_grasped}, "
            f"palm_final_locked={self.palm_final_locked}, "
            f"handover_plan_ready={self.handover_plan_ready}, "
            f"arm_joints={current_text}"
        )

        return response

    # ================================================================
    # State helpers
    # ================================================================

    def _set_state(
        self,
        new_state,
    ):
        if self.state != new_state:
            self.get_logger().info(
                f"STATE: {self.state.name} -> {new_state.name}"
            )

        self.state = new_state
        self.state_entered_at = time.monotonic()

        if (
            new_state
            !=
            State.WAIT_MECHANICAL_HOME
        ):
            self.mechanical_home_in_tolerance_since = None

    def _state_age(self):
        return (
            time.monotonic()
            -
            self.state_entered_at
        )

    def _abort(
        self,
        message,
    ):
        self.get_logger().error(
            f"TASK ABORTED at {self.state.name}: {message}"
        )

        self.running = False
        self.pending_future = None
        self.pending_key = None

        self._set_state(
            State.ERROR
        )

    def _current_arm_positions(self):
        values = []

        for name in self.arm_joint_names:
            if name not in self.current_joint_map:
                return None

            values.append(
                float(
                    self.current_joint_map[name]
                )
            )

        return values

    def _mechanical_home_error(self):
        current = (
            self._current_arm_positions()
        )

        if current is None:
            return None

        return max(
            abs(value)
            for value in current
        )

    # ================================================================
    # Non-blocking Trigger call
    # ================================================================

    def _call_trigger(
        self,
        key,
        next_state,
        *,
        optional=False,
    ):
        if self.pending_future is not None:
            return

        client = self.service_clients[key]
        service_name = self.service_names[key]

        if not client.service_is_ready():
            if (
                self._state_age()
                <=
                self.service_wait_timeout
            ):
                return

            if optional:
                self.get_logger().warning(
                    f"OPTIONAL service unavailable, skipping: {service_name}"
                )
                self._set_state(
                    next_state
                )
            else:
                self._abort(
                    f"required service unavailable: {service_name}"
                )

            return

        this_run_id = self.run_id

        self.get_logger().info(
            f"CALL {service_name}"
        )

        future = client.call_async(
            Trigger.Request()
        )

        self.pending_future = future
        self.pending_key = key

        def done_callback(
            done_future,
        ):
            if this_run_id != self.run_id:
                return

            self.pending_future = None
            self.pending_key = None

            try:
                result = (
                    done_future.result()
                )

            except Exception as exc:
                if optional:
                    self.get_logger().warning(
                        f"OPTIONAL {service_name} exception; skipping: {exc}"
                    )

                    if self.running:
                        self._set_state(
                            next_state
                        )

                else:
                    self._abort(
                        f"{service_name} raised exception: {exc}"
                    )

                return

            if result is None:
                if optional:
                    self.get_logger().warning(
                        f"OPTIONAL {service_name} returned no response; skipping."
                    )

                    if self.running:
                        self._set_state(
                            next_state
                        )

                else:
                    self._abort(
                        f"{service_name} returned no response"
                    )

                return

            if not bool(
                result.success
            ):
                if optional:
                    self.get_logger().warning(
                        f"OPTIONAL {service_name} failed; skipping: "
                        f"{result.message}"
                    )

                    if self.running:
                        self._set_state(
                            next_state
                        )

                else:
                    self._abort(
                        f"{service_name} failed: {result.message}"
                    )

                return

            self.get_logger().info(
                f"OK {service_name} | {result.message}"
            )

            if self.running:
                self._set_state(
                    next_state
                )

        future.add_done_callback(
            done_callback
        )

    # ================================================================
    # Non-blocking std_srvs/Empty /move_home
    # ================================================================

    def _call_move_home(self):
        if self.pending_future is not None:
            return

        if not self.move_home_client.service_is_ready():
            if (
                self._state_age()
                >
                self.service_wait_timeout
            ):
                self._abort(
                    "required service unavailable: /move_home"
                )
            return

        this_run_id = self.run_id

        self.get_logger().warning(
            "CALL /move_home | DRIVER MECHANICAL HOME | "
            "target joint1..joint6 = 0 rad | gripper remains commanded closed."
        )

        future = self.move_home_client.call_async(
            Empty.Request()
        )

        self.pending_future = future
        self.pending_key = "move_home"

        def done_callback(
            done_future,
        ):
            if this_run_id != self.run_id:
                return

            self.pending_future = None
            self.pending_key = None

            try:
                result = (
                    done_future.result()
                )

            except Exception as exc:
                self._abort(
                    f"/move_home raised exception: {exc}"
                )
                return

            if result is None:
                self._abort(
                    "/move_home returned no response"
                )
                return

            # Empty has no success field. A valid response only means the
            # service request completed. Real joint feedback is verified next.
            self.get_logger().info(
                "OK /move_home service returned. "
                "Now verifying real joint1..joint6 -> 0 rad."
            )

            if self.running:
                self._set_state(
                    State.WAIT_MECHANICAL_HOME
                )

        future.add_done_callback(
            done_callback
        )

    # ================================================================
    # FSM
    # ================================================================

    def _loop(self):
        if not self.running:
            return

        if self.pending_future is not None:
            return

        # ------------------------------------------------------------
        # Start / project WORK_HOME
        # ------------------------------------------------------------

        if self.state == State.RESET_TARGETS:
            self._call_trigger(
                "targets_reset",
                State.WORK_HOME_EMPTY,
            )

        elif self.state == State.WORK_HOME_EMPTY:
            self._call_trigger(
                "work_home",
                State.START_CASE_OBSERVATION,
            )

        # ------------------------------------------------------------
        # First perception: CASE ONLY
        # ------------------------------------------------------------

        elif self.state == State.START_CASE_OBSERVATION:
            self.case_locked = False
            self.grasp_plan_ready = False

            self._call_trigger(
                "start_observation",
                State.WAIT_CASE,
            )

        elif self.state == State.WAIT_CASE:
            if self.case_locked:
                self.get_logger().info(
                    "CASE locked. Palm detections are intentionally ignored."
                )
                self._set_state(
                    State.STOP_CASE_OBSERVATION
                )

            elif (
                self._state_age()
                >
                self.case_timeout
            ):
                self._abort(
                    "timeout waiting for /targets/case_locked"
                )

        elif self.state == State.STOP_CASE_OBSERVATION:
            self._call_trigger(
                "stop_observation",
                State.CREATE_TABLE,
            )

        # ------------------------------------------------------------
        # Table + grasp
        # ------------------------------------------------------------

        elif self.state == State.CREATE_TABLE:
            self._call_trigger(
                "create_table",
                State.WAIT_GRASP_PLAN,
            )

        elif self.state == State.WAIT_GRASP_PLAN:
            if self.grasp_plan_ready:
                self.get_logger().info(
                    "Grasp geometry is ready."
                )
                self._set_state(
                    State.PREGRASP
                )

            elif (
                self._state_age()
                >
                self.grasp_plan_timeout
            ):
                self._abort(
                    "timeout waiting for /pick/grasp_plan_ready"
                )

        elif self.state == State.PREGRASP:
            self._call_trigger(
                "pregrasp",
                State.OPEN_GRIPPER,
            )

        elif self.state == State.OPEN_GRIPPER:
            self._call_trigger(
                "gripper_open",
                State.DESCEND_GRASP,
            )

        elif self.state == State.DESCEND_GRASP:
            self._call_trigger(
                "grasp_preview",
                State.GRASP_OBJECT,
            )

        elif self.state == State.GRASP_OBJECT:
            self.object_grasped = False

            self._call_trigger(
                "gripper_grasp",
                State.WAIT_OBJECT_GRASPED,
            )

        elif self.state == State.WAIT_OBJECT_GRASPED:
            if self.object_grasped:
                self.get_logger().info(
                    "Object grasp verification received."
                )
                self._set_state(
                    State.LIFT
                )

            elif (
                self._state_age()
                >
                self.object_grasped_timeout
            ):
                self._abort(
                    "timeout waiting for /pick/object_grasped"
                )

        elif self.state == State.LIFT:
            self._call_trigger(
                "lift",
                State.MOVE_MECHANICAL_HOME,
            )

        # ------------------------------------------------------------
        # NEW: driver-level mechanical HOME, joint1..joint6 = 0.
        # ------------------------------------------------------------

        elif self.state == State.MOVE_MECHANICAL_HOME:
            self._call_move_home()

        elif self.state == State.WAIT_MECHANICAL_HOME:
            error = (
                self._mechanical_home_error()
            )

            if error is not None:
                if (
                    error
                    <=
                    self.mechanical_home_tolerance
                ):
                    if (
                        self.mechanical_home_in_tolerance_since
                        is None
                    ):
                        self.mechanical_home_in_tolerance_since = (
                            time.monotonic()
                        )

                    settled_for = (
                        time.monotonic()
                        -
                        self.mechanical_home_in_tolerance_since
                    )

                    if (
                        settled_for
                        >=
                        self.mechanical_home_settle_sec
                    ):
                        self.get_logger().info(
                            "MECHANICAL HOME VERIFIED | "
                            f"max_abs_joint={error:.5f} rad | "
                            f"settled={settled_for:.2f}s"
                        )

                        self._set_state(
                            State.RESET_HANDOVER
                        )

                else:
                    self.mechanical_home_in_tolerance_since = None

            if (
                self.state
                ==
                State.WAIT_MECHANICAL_HOME
                and
                self._state_age()
                >
                self.mechanical_home_timeout
            ):
                if error is None:
                    detail = (
                        "no complete /feedback/joint_states"
                    )
                else:
                    detail = (
                        f"max_abs_joint={error:.5f} rad"
                    )

                self._abort(
                    "mechanical HOME verification timeout | "
                    + detail
                )

        # ------------------------------------------------------------
        # Second perception: Palm Final ONLY, at mechanical HOME.
        # ------------------------------------------------------------

        elif self.state == State.RESET_HANDOVER:
            self.handover_plan_ready = False
            self.palm_final_locked = False

            self._call_trigger(
                "handover_reset",
                State.ARM_PALM_FINAL,
            )

        elif self.state == State.ARM_PALM_FINAL:
            self._call_trigger(
                "arm_palm_final",
                State.START_HAND_OBSERVATION,
            )

        elif self.state == State.START_HAND_OBSERVATION:
            self._call_trigger(
                "start_observation",
                State.WAIT_PALM_FINAL,
            )

        elif self.state == State.WAIT_PALM_FINAL:
            if self.palm_final_locked:
                self.get_logger().info(
                    "Fresh Palm Final locked at MECHANICAL HOME."
                )
                self._set_state(
                    State.STOP_HAND_OBSERVATION
                )

            elif (
                self._state_age()
                >
                self.palm_final_timeout
            ):
                self._abort(
                    "timeout waiting for /targets/palm_final_locked"
                )

        elif self.state == State.STOP_HAND_OBSERVATION:
            self._call_trigger(
                "stop_observation",
                State.RECOMPUTE_HANDOVER,
            )

        elif self.state == State.RECOMPUTE_HANDOVER:
            self._call_trigger(
                "handover_recompute",
                State.HANDOVER_APPROACH,
            )

        # ------------------------------------------------------------
        # Handover
        # ------------------------------------------------------------

        elif self.state == State.HANDOVER_APPROACH:
            self._call_trigger(
                "handover_approach",
                State.HANDOVER_FINAL,
            )

        elif self.state == State.HANDOVER_FINAL:
            self._call_trigger(
                "handover_final",
                State.RELEASE,
            )

        elif self.state == State.RELEASE:
            self._call_trigger(
                "gripper_open",
                State.RETURN_WORK_HOME_EMPTY,
            )

        # ------------------------------------------------------------
        # Finish at project WORK_HOME.
        # ------------------------------------------------------------

        elif self.state == State.RETURN_WORK_HOME_EMPTY:
            self._call_trigger(
                "work_home",
                State.REMOVE_TABLE_END,
            )

        elif self.state == State.REMOVE_TABLE_END:
            self._call_trigger(
                "remove_table",
                State.DONE,
                optional=True,
            )

        elif self.state == State.DONE:
            self.running = False

            self.get_logger().info(
                "=========================================="
            )
            self.get_logger().info(
                "TASK FINISHED"
            )
            self.get_logger().info(
                "=========================================="
            )

        elif self.state == State.ERROR:
            self.running = False


def main(args=None):
    rclpy.init(args=args)

    node = PiperManager()

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
