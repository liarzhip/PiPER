import math

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    OrientationConstraint,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, Empty
from std_srvs.srv import Trigger


class MoveItPlanOnlyNode(Node):
    """
    STEP 4:
    Subscribe to /handshake/approach_pose and ask MoveIt2 to PLAN ONLY.

    This node NEVER requests trajectory execution.
    It publishes the returned trajectory to /display_planned_path for RViz.
    """

    def __init__(self):
        super().__init__("moveit_plan_only_node")

        # ----------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------
        self.declare_parameter("planning_group", "arm")
        self.declare_parameter("pose_link", "gripper_base")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("move_action", "/move_action")

        self.declare_parameter("planning_time_s", 5.0)
        self.declare_parameter("planning_attempts", 5)

        # Position goal tolerance, meters
        self.declare_parameter("position_tolerance_m", 0.010)

        # Orientation goal tolerance, degrees
        self.declare_parameter("orientation_tolerance_deg", 5.0)

        # These only affect the generated trajectory; STEP 4 does not execute it.
        self.declare_parameter("velocity_scaling", 0.10)
        self.declare_parameter("acceleration_scaling", 0.10)

        self.planning_group = str(
            self.get_parameter("planning_group").value
        )
        self.pose_link = str(
            self.get_parameter("pose_link").value
        )
        self.base_frame = str(
            self.get_parameter("base_frame").value
        )
        self.move_action_name = str(
            self.get_parameter("move_action").value
        )

        self.planning_time = float(
            self.get_parameter("planning_time_s").value
        )
        self.planning_attempts = int(
            self.get_parameter("planning_attempts").value
        )
        self.position_tolerance = float(
            self.get_parameter("position_tolerance_m").value
        )
        self.orientation_tolerance = math.radians(
            float(
                self.get_parameter(
                    "orientation_tolerance_deg"
                ).value
            )
        )
        self.velocity_scaling = float(
            self.get_parameter("velocity_scaling").value
        )
        self.acceleration_scaling = float(
            self.get_parameter("acceleration_scaling").value
        )

        # ----------------------------------------------------------
        # State
        # ----------------------------------------------------------
        self.approach_pose = None
        self.plan_ready = False
        self.request_running = False

        # ----------------------------------------------------------
        # ROS interfaces
        # ----------------------------------------------------------
        self.create_subscription(
            PoseStamped,
            "/handshake/approach_pose",
            self.approach_callback,
            10,
        )

        self.create_subscription(
            Bool,
            "/handshake/plan_ready",
            self.plan_ready_callback,
            10,
        )

        self.create_subscription(
            Empty,
            "/handshake/reset_event",
            self.reset_callback,
            10,
        )

        self.plan_service = self.create_service(
            Trigger,
            "/handshake/plan_approach",
            self.plan_service_callback,
        )

        self.result_pub = self.create_publisher(
            Bool,
            "/handshake/moveit_plan_success",
            10,
        )

        self.display_pub = self.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            10,
        )

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            self.move_action_name,
        )

        self.get_logger().info(
            "MoveIt STEP 4 plan-only node started"
        )
        self.get_logger().info(
            f"group={self.planning_group}, "
            f"pose_link={self.pose_link}, "
            f"base_frame={self.base_frame}"
        )
        self.get_logger().warning(
            "PLAN ONLY: this node will NOT execute robot motion"
        )

    # --------------------------------------------------------------
    # Input callbacks
    # --------------------------------------------------------------
    def approach_callback(self, msg: PoseStamped):
        self.approach_pose = msg

    def plan_ready_callback(self, msg: Bool):
        self.plan_ready = bool(msg.data)

    def reset_callback(self, msg: Empty):
        self.approach_pose = None
        self.plan_ready = False
        self.request_running = False

        result = Bool()
        result.data = False
        self.result_pub.publish(result)

        self.get_logger().info(
            "MoveIt STEP 4 planning state cleared"
        )

    # --------------------------------------------------------------
    # Constraint construction
    # --------------------------------------------------------------
    def build_goal_constraints(self, target: PoseStamped):
        constraints = Constraints()
        constraints.name = "handshake_approach_pose"

        # Position tolerance represented by a small sphere.
        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.pose_link
        pc.weight = 1.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
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

        # Orientation tolerance around the palm-derived target quaternion.
        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.pose_link
        oc.orientation = target.pose.orientation
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

        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)

        return constraints

    # --------------------------------------------------------------
    # Manual planning trigger
    # --------------------------------------------------------------
    def plan_service_callback(
        self,
        request,
        response,
    ):
        if self.request_running:
            response.success = False
            response.message = "Planning request already running"
            return response

        if not self.plan_ready:
            response.success = False
            response.message = (
                "Handshake planner has not produced a valid plan yet"
            )
            return response

        if self.approach_pose is None:
            response.success = False
            response.message = "No /handshake/approach_pose received"
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
                f"'{self.move_action_name}' is unavailable"
            )
            return response

        goal_msg = MoveGroup.Goal()

        # Ask MoveIt to use the real/current robot state.
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

        constraints = self.build_goal_constraints(
            self.approach_pose
        )
        goal_msg.request.goal_constraints = [
            constraints
        ]

        # Critical safety setting for STEP 4.
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
            f"[{p.x:+.4f}, {p.y:+.4f}, {p.z:+.4f}] m"
        )
        self.get_logger().info(
            f"Quaternion xyzw = "
            f"[{q.x:+.4f}, {q.y:+.4f}, "
            f"{q.z:+.4f}, {q.w:+.4f}]"
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
            self.goal_response_callback
        )

        response.success = True
        response.message = (
            "MoveIt plan-only request submitted"
        )

        return response

    # --------------------------------------------------------------
    # Action callbacks
    # --------------------------------------------------------------
    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.request_running = False

            msg = Bool()
            msg.data = False
            self.result_pub.publish(msg)

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

    def plan_result_callback(self, future):
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
        self.result_pub.publish(
            success_msg
        )

        if not success:
            self.get_logger().error(
                f"MOVEIT PLAN FAILED, "
                f"MoveIt error code = {error_code}"
            )
            return

        joint_names = (
            result
            .planned_trajectory
            .joint_trajectory
            .joint_names
        )

        points = (
            result
            .planned_trajectory
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
            f"Trajectory joints = {list(joint_names)}"
        )
        self.get_logger().info(
            f"Trajectory points = {len(points)}"
        )
        self.get_logger().info(
            "NO ROBOT MOTION - STEP 4 COMPLETE"
        )
        self.get_logger().info(
            "======================================"
        )

        # Visualize the returned trajectory in RViz.
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


def main(args=None):
    rclpy.init(args=args)

    node = MoveItPlanOnlyNode()

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
