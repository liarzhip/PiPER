#!/usr/bin/env python3
"""
PIPER Pick & Handover task manager.

Key rules:
- No nested rclpy.spin_until_future_complete().
- Service calls are asynchronous.
- Startup cleanup is minimal:
    /targets/reset
    -> WORK_HOME
  We do NOT require /handover/reset_plan or /scene/remove_table at startup.
- /handover/reset_plan is used only when a fresh Palm Final observation is needed.
"""

import enum
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


class State(enum.Enum):
    IDLE = 0

    RESET_TARGETS = 1
    WORK_HOME = 2

    START_OBSERVATION = 3
    WAIT_INITIAL_TARGETS = 4
    STOP_INITIAL_OBSERVATION = 5

    CREATE_TABLE = 6
    WAIT_GRASP_PLAN = 7

    PREGRASP = 8
    OPEN_GRIPPER = 9
    DESCEND_GRASP = 10
    GRASP_OBJECT = 11
    WAIT_OBJECT_GRASPED = 12
    LIFT = 13

    RESET_HANDOVER_BEFORE_OBSERVE = 14
    OBSERVE_HAND = 15
    WAIT_HANDOVER_PLAN = 16

    HANDOVER_APPROACH = 17
    HANDOVER_FINAL = 18
    RELEASE = 19

    RETURN_HOME = 20
    REMOVE_TABLE_END = 21

    DONE = 22
    ERROR = 23


class PiperManager(Node):

    def __init__(self):
        super().__init__("piper_manager_node")

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        self.declare_parameter("loop_period_sec", 0.10)
        self.declare_parameter("service_wait_timeout_sec", 5.0)
        self.declare_parameter("wait_initial_targets_timeout_sec", 30.0)
        self.declare_parameter("wait_grasp_plan_timeout_sec", 10.0)
        self.declare_parameter("wait_object_grasped_timeout_sec", 8.0)
        self.declare_parameter("wait_handover_plan_timeout_sec", 30.0)

        self.loop_period = float(
            self.get_parameter("loop_period_sec").value
        )
        self.service_wait_timeout = float(
            self.get_parameter("service_wait_timeout_sec").value
        )
        self.initial_targets_timeout = float(
            self.get_parameter("wait_initial_targets_timeout_sec").value
        )
        self.grasp_plan_timeout = float(
            self.get_parameter("wait_grasp_plan_timeout_sec").value
        )
        self.object_grasped_timeout = float(
            self.get_parameter("wait_object_grasped_timeout_sec").value
        )
        self.handover_plan_timeout = float(
            self.get_parameter("wait_handover_plan_timeout_sec").value
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

        self.start_targets_ready = False
        self.grasp_plan_ready = False
        self.object_grasped = False
        self.handover_plan_ready = False

        # ------------------------------------------------------------
        # Real service interfaces
        # ------------------------------------------------------------
        service_map = {
            "targets_reset":
                "/targets/reset",

            "handover_reset":
                "/handover/reset_plan",

            "remove_table":
                "/scene/remove_table",

            "work_home":
                "/work_home/execute",

            "start_observation":
                "/perception/start_observation",

            "stop_observation":
                "/perception/stop_observation",

            "create_table":
                "/scene/apply_table_from_case",

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

            "observe_hand":
                "/moveit/execute_observe_hand",

            "handover_approach":
                "/moveit/execute_handover_approach",

            "handover_final":
                "/moveit/execute_handover_final",
        }

        self.service_clients = {
            key: self.create_client(Trigger, service_name)
            for key, service_name in service_map.items()
        }
        self.service_names = dict(service_map)

        # ------------------------------------------------------------
        # Readiness topics
        # ------------------------------------------------------------
        self.create_subscription(
            Bool,
            "/targets/start_targets_ready",
            self._start_targets_ready_cb,
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
            "/handover/plan_ready",
            self._handover_plan_ready_cb,
            10,
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
            "Startup flow: RESET_TARGETS -> WORK_HOME -> perception."
        )

    # ================================================================
    # Topic callbacks
    # ================================================================

    def _start_targets_ready_cb(self, msg):
        self.start_targets_ready = bool(msg.data)

    def _grasp_plan_ready_cb(self, msg):
        self.grasp_plan_ready = bool(msg.data)

    def _object_grasped_cb(self, msg):
        self.object_grasped = bool(msg.data)

    def _handover_plan_ready_cb(self, msg):
        self.handover_plan_ready = bool(msg.data)

    # ================================================================
    # Manager services
    # ================================================================

    def _start_cb(self, request, response):
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

        self.start_targets_ready = False
        self.grasp_plan_ready = False
        self.object_grasped = False
        self.handover_plan_ready = False

        self.running = True
        self._set_state(State.RESET_TARGETS)

        response.success = True
        response.message = (
            "Task accepted. Starting automatic pick-and-handover sequence."
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

    def _stop_cb(self, request, response):
        del request

        self.run_id += 1
        self.running = False
        self.pending_future = None
        self.pending_key = None
        self._set_state(State.IDLE)

        response.success = True
        response.message = (
            "Manager task stopped. "
            "This does not cancel a trajectory already executing in MoveIt."
        )
        return response

    def _status_cb(self, request, response):
        del request

        response.success = True
        response.message = (
            f"state={self.state.name}, "
            f"running={self.running}, "
            f"service_pending={self.pending_key}, "
            f"start_targets_ready={self.start_targets_ready}, "
            f"grasp_plan_ready={self.grasp_plan_ready}, "
            f"object_grasped={self.object_grasped}, "
            f"handover_plan_ready={self.handover_plan_ready}"
        )
        return response

    # ================================================================
    # State helpers
    # ================================================================

    def _set_state(self, new_state):
        if self.state != new_state:
            self.get_logger().info(
                f"STATE: {self.state.name} -> {new_state.name}"
            )

        self.state = new_state
        self.state_entered_at = time.monotonic()

    def _state_age(self):
        return time.monotonic() - self.state_entered_at

    def _abort(self, message):
        self.get_logger().error(
            f"TASK ABORTED at {self.state.name}: {message}"
        )
        self.running = False
        self.pending_future = None
        self.pending_key = None
        self._set_state(State.ERROR)

    # ================================================================
    # Non-blocking Trigger service call
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
            if self._state_age() <= self.service_wait_timeout:
                return

            if optional:
                self.get_logger().warning(
                    f"OPTIONAL service unavailable, skipping: {service_name}"
                )
                self._set_state(next_state)
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

        def done_callback(done_future):
            if this_run_id != self.run_id:
                return

            self.pending_future = None
            self.pending_key = None

            try:
                result = done_future.result()
            except Exception as exc:
                if optional:
                    self.get_logger().warning(
                        f"OPTIONAL {service_name} exception; skipping: {exc}"
                    )
                    if self.running:
                        self._set_state(next_state)
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
                        self._set_state(next_state)
                else:
                    self._abort(
                        f"{service_name} returned no response"
                    )
                return

            if not bool(result.success):
                if optional:
                    self.get_logger().warning(
                        f"OPTIONAL {service_name} failed; skipping: "
                        f"{result.message}"
                    )
                    if self.running:
                        self._set_state(next_state)
                else:
                    self._abort(
                        f"{service_name} failed: {result.message}"
                    )
                return

            self.get_logger().info(
                f"OK {service_name} | {result.message}"
            )

            if self.running:
                self._set_state(next_state)

        future.add_done_callback(done_callback)

    # ================================================================
    # FSM
    # ================================================================

    def _loop(self):
        if not self.running:
            return

        if self.pending_future is not None:
            return

        # ------------------------------------------------------------
        # Start clean: target locks only.
        # No handover reset/table removal is required at startup.
        # ------------------------------------------------------------

        if self.state == State.RESET_TARGETS:
            self._call_trigger(
                "targets_reset",
                State.WORK_HOME,
            )

        # ------------------------------------------------------------
        # MoveIt -> WORK_HOME
        # ------------------------------------------------------------

        elif self.state == State.WORK_HOME:
            self._call_trigger(
                "work_home",
                State.START_OBSERVATION,
            )

        # ------------------------------------------------------------
        # Initial perception: Case + Palm Initial
        # ------------------------------------------------------------

        elif self.state == State.START_OBSERVATION:
            self.start_targets_ready = False
            self.grasp_plan_ready = False

            self._call_trigger(
                "start_observation",
                State.WAIT_INITIAL_TARGETS,
            )

        elif self.state == State.WAIT_INITIAL_TARGETS:
            if self.start_targets_ready:
                self.get_logger().info(
                    "Initial Case + Palm Initial are locked."
                )
                self._set_state(
                    State.STOP_INITIAL_OBSERVATION
                )

            elif self._state_age() > self.initial_targets_timeout:
                self._abort(
                    "timeout waiting for /targets/start_targets_ready"
                )

        elif self.state == State.STOP_INITIAL_OBSERVATION:
            self._call_trigger(
                "stop_observation",
                State.CREATE_TABLE,
            )

        # ------------------------------------------------------------
        # Table + grasp target
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
                self._set_state(State.PREGRASP)

            elif self._state_age() > self.grasp_plan_timeout:
                self._abort(
                    "timeout waiting for /pick/grasp_plan_ready"
                )

        # ------------------------------------------------------------
        # Pick
        # ------------------------------------------------------------

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
                self._set_state(State.LIFT)

            elif self._state_age() > self.object_grasped_timeout:
                self._abort(
                    "timeout waiting for /pick/object_grasped"
                )

        elif self.state == State.LIFT:
            self._call_trigger(
                "lift",
                State.RESET_HANDOVER_BEFORE_OBSERVE,
            )

        # ------------------------------------------------------------
        # Fresh Palm Final + handover
        # ------------------------------------------------------------

        elif self.state == State.RESET_HANDOVER_BEFORE_OBSERVE:
            self.handover_plan_ready = False

            self._call_trigger(
                "handover_reset",
                State.OBSERVE_HAND,
            )

        elif self.state == State.OBSERVE_HAND:
            self._call_trigger(
                "observe_hand",
                State.WAIT_HANDOVER_PLAN,
            )

        elif self.state == State.WAIT_HANDOVER_PLAN:
            if self.handover_plan_ready:
                self.get_logger().info(
                    "Fresh Palm Final handover plan is ready."
                )
                self._set_state(State.HANDOVER_APPROACH)

            elif self._state_age() > self.handover_plan_timeout:
                self._abort(
                    "timeout waiting for /handover/plan_ready"
                )

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
                State.RETURN_HOME,
            )

        # ------------------------------------------------------------
        # Finish
        # ------------------------------------------------------------

        elif self.state == State.RETURN_HOME:
            self._call_trigger(
                "work_home",
                State.REMOVE_TABLE_END,
            )

        elif self.state == State.REMOVE_TABLE_END:
            # Cleanup only. A missing/failed remove service must not
            # prevent the task from reaching DONE.
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
