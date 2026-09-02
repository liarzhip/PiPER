import math
import threading

import numpy as np
import rclpy

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


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


class TableSceneNode(Node):
    """
    STEP 7A - add a local tabletop collision patch to MoveIt.

    Input:
        /targets/earbud_case_pose_base

    Assumption:
        The locked case pose position is the center of the CASE TOP surface.
        The case +Z axis is normal to the top surface (sign may be flipped).

    The tabletop is known to be HORIZONTAL in base_link.

    From a measured case height, this node estimates only its Z height:

        table_z = case_top_z - case_height

    The case quaternion is intentionally ignored for table orientation.
    A horizontal local collision BOX is placed immediately below the tabletop.

    Services:
        /scene/apply_table_from_case
        /scene/remove_table

    This node does NOT move the robot.
    """

    def __init__(self):
        super().__init__("table_scene_node")

        self.cb_group = ReentrantCallbackGroup()
        self.operation_lock = threading.Lock()

        # ----------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------

        self.declare_parameter(
            "base_frame",
            "base_link",
        )
        self.declare_parameter(
            "table_object_id",
            "table_patch",
        )

        # Must be measured by the user.
        self.declare_parameter(
            "case_height_m",
            0.0,
        )

        # Local tabletop collision region around the earbud case.
        self.declare_parameter(
            "table_patch_size_x_m",
            0.50,
        )
        self.declare_parameter(
            "table_patch_size_y_m",
            0.50,
        )
        self.declare_parameter(
            "table_thickness_m",
            0.05,
        )

        # Raise the effective collision top slightly above the estimated
        # physical tabletop to provide a conservative margin.
        self.declare_parameter(
            "table_safety_margin_m",
            0.003,
        )

        self.declare_parameter(
            "apply_planning_scene_service",
            "/apply_planning_scene",
        )
        self.declare_parameter(
            "service_timeout_sec",
            5.0,
        )

        self.base_frame = str(
            self.get_parameter("base_frame").value
        )
        self.table_object_id = str(
            self.get_parameter("table_object_id").value
        )

        self.case_height = float(
            self.get_parameter("case_height_m").value
        )
        self.patch_x = float(
            self.get_parameter("table_patch_size_x_m").value
        )
        self.patch_y = float(
            self.get_parameter("table_patch_size_y_m").value
        )
        self.table_thickness = float(
            self.get_parameter("table_thickness_m").value
        )
        self.safety_margin = float(
            self.get_parameter("table_safety_margin_m").value
        )

        self.apply_service_name = str(
            self.get_parameter(
                "apply_planning_scene_service"
            ).value
        )
        self.service_timeout = float(
            self.get_parameter("service_timeout_sec").value
        )

        # ----------------------------------------------------------
        # State
        # ----------------------------------------------------------

        self.locked_case_pose = None
        self.table_applied = False

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Locked case pose is transient-local.
        self.create_subscription(
            PoseStamped,
            "/targets/earbud_case_pose_base",
            self.case_pose_callback,
            latched_qos,
            callback_group=self.cb_group,
        )

        self.table_ready_pub = self.create_publisher(
            Bool,
            "/scene/table_ready",
            10,
        )

        self.table_pose_pub = self.create_publisher(
            PoseStamped,
            "/scene/table_patch_pose",
            latched_qos,
        )

        self.apply_client = self.create_client(
            ApplyPlanningScene,
            self.apply_service_name,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/scene/apply_table_from_case",
            self.apply_table_callback,
            callback_group=self.cb_group,
        )

        self.create_service(
            Trigger,
            "/scene/remove_table",
            self.remove_table_callback,
            callback_group=self.cb_group,
        )

        # Periodically publish state so plain `ros2 topic echo --once`
        # works without needing transient-local QoS options.
        self.state_timer = self.create_timer(
            0.5,
            self.publish_current_state,
            callback_group=self.cb_group,
        )

        self.publish_ready(False)

        self.get_logger().info(
            "STEP 7A table planning-scene node started."
        )
        self.get_logger().info(
            "NO ROBOT MOTION."
        )

        if self.case_height <= 0.0:
            self.get_logger().warning(
                "case_height_m is not configured yet. "
                "Measure the real earbud-case thickness before applying table."
            )

    def publish_ready(self, value):
        msg = Bool()
        msg.data = bool(value)
        self.table_ready_pub.publish(msg)

    def publish_current_state(self):
        self.publish_ready(
            self.table_applied
        )

    def case_pose_callback(self, msg):
        if str(msg.header.frame_id) != self.base_frame:
            self.get_logger().warning(
                "Ignoring locked case pose: "
                f"frame='{msg.header.frame_id}', "
                f"expected='{self.base_frame}'."
            )
            return

        self.locked_case_pose = msg

    @staticmethod
    def wait_future(future, timeout_sec):
        event = threading.Event()

        future.add_done_callback(
            lambda _future: event.set()
        )

        if not event.wait(
            timeout=max(
                0.05,
                float(timeout_sec),
            )
        ):
            return None

        try:
            return future.result()
        except Exception:
            return None

    def compute_table_pose(self):
        """
        Horizontal-table model for the current project.

        IMPORTANT:
        The real tabletop is known to be horizontal in base_link.
        Therefore table orientation must NOT be inferred from the
        earbud-case quaternion / camera viewing angle.

        We use only the locked case TOP-CENTER position:
            table_z = case_top_z - measured_case_height

        The collision box is centered below that horizontal plane.
        """

        if self.locked_case_pose is None:
            raise RuntimeError(
                "No locked earbud-case pose received."
            )

        if not (0.001 <= self.case_height <= 0.20):
            raise RuntimeError(
                "case_height_m must be the measured real case thickness "
                "in meters (reasonable range 0.001~0.20)."
            )

        if (
            self.patch_x <= 0.05
            or self.patch_y <= 0.05
            or self.table_thickness <= 0.005
        ):
            raise RuntimeError(
                "Invalid table collision dimensions."
            )

        msg = self.locked_case_pose

        case_top_x = float(
            msg.pose.position.x
        )
        case_top_y = float(
            msg.pose.position.y
        )
        case_top_z = float(
            msg.pose.position.z
        )

        # ----------------------------------------------------------
        # Known geometry:
        #
        # base_link +Z is vertical/up for this installation and the
        # physical tabletop is horizontal.
        #
        # DO NOT use the case quaternion here.
        # ----------------------------------------------------------

        physical_table_z = (
            case_top_z
            - self.case_height
        )

        # Conservative collision surface slightly ABOVE the physical table.
        collision_top_z = (
            physical_table_z
            + self.safety_margin
        )

        # Box center is half its thickness below its upper surface.
        table_center_z = (
            collision_top_z
            - 0.5 * self.table_thickness
        )

        pose = Pose()

        # Local table patch centered below the detected earbud case.
        pose.position.x = case_top_x
        pose.position.y = case_top_y
        pose.position.z = table_center_z

        # Horizontal in base_link:
        # no roll, no pitch, no yaw required for a rectangular patch.
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        physical_table_top = np.array(
            [
                case_top_x,
                case_top_y,
                physical_table_z,
            ],
            dtype=np.float64,
        )

        surface_up = np.array(
            [
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )

        return (
            pose,
            physical_table_top,
            surface_up,
        )

    def call_apply_scene(self, scene):
        if not self.apply_client.wait_for_service(
            timeout_sec=self.service_timeout
        ):
            raise RuntimeError(
                f"MoveIt service '{self.apply_service_name}' is unavailable."
            )

        req = ApplyPlanningScene.Request()
        req.scene = scene

        future = self.apply_client.call_async(
            req
        )

        result = self.wait_future(
            future,
            self.service_timeout,
        )

        if result is None:
            raise RuntimeError(
                "ApplyPlanningScene service timed out."
            )

        if not result.success:
            raise RuntimeError(
                "MoveIt rejected PlanningScene update."
            )

    def build_add_scene(self):
        pose, physical_top, surface_up = (
            self.compute_table_pose()
        )

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [
            float(self.patch_x),
            float(self.patch_y),
            float(self.table_thickness),
        ]

        obj = CollisionObject()
        obj.header.frame_id = self.base_frame
        obj.id = self.table_object_id
        obj.primitives = [
            primitive
        ]
        obj.primitive_poses = [
            pose
        ]
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [
            obj
        ]

        pose_msg = PoseStamped()
        pose_msg.header.stamp = (
            self.get_clock().now().to_msg()
        )
        pose_msg.header.frame_id = self.base_frame
        pose_msg.pose = pose

        return (
            scene,
            pose_msg,
            physical_top,
            surface_up,
        )

    def apply_table_callback(
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
                "Planning-scene operation already in progress."
            )
            return response

        try:
            try:
                (
                    scene,
                    pose_msg,
                    physical_top,
                    surface_up,
                ) = self.build_add_scene()

                self.call_apply_scene(
                    scene
                )

            except Exception as exc:
                self.table_applied = False
                self.publish_ready(False)

                response.success = False
                response.message = str(exc)

                self.get_logger().error(
                    response.message
                )
                return response

            self.table_applied = True
            self.table_pose_pub.publish(
                pose_msg
            )
            self.publish_ready(True)

            response.success = True
            response.message = (
                "Table collision patch applied to MoveIt."
            )

            self.get_logger().info(
                "TABLE COLLISION APPLIED | "
                f"id={self.table_object_id} | "
                f"size=[{self.patch_x:.3f}, "
                f"{self.patch_y:.3f}, "
                f"{self.table_thickness:.3f}] m"
            )

            self.get_logger().info(
                "Estimated physical tabletop point = "
                f"[{physical_top[0]:+.4f}, "
                f"{physical_top[1]:+.4f}, "
                f"{physical_top[2]:+.4f}] m"
            )

            self.get_logger().info(
                "Surface up = "
                f"[{surface_up[0]:+.3f}, "
                f"{surface_up[1]:+.3f}, "
                f"{surface_up[2]:+.3f}]"
            )

            return response

        finally:
            self.operation_lock.release()

    def remove_table_callback(
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
                "Planning-scene operation already in progress."
            )
            return response

        try:
            obj = CollisionObject()
            obj.header.frame_id = (
                self.base_frame
            )
            obj.id = self.table_object_id
            obj.operation = (
                CollisionObject.REMOVE
            )

            scene = PlanningScene()
            scene.is_diff = True
            scene.world.collision_objects = [
                obj
            ]

            try:
                self.call_apply_scene(
                    scene
                )

            except Exception as exc:
                response.success = False
                response.message = str(exc)
                return response

            self.table_applied = False
            self.publish_ready(False)

            response.success = True
            response.message = (
                "Table collision patch removed."
            )

            self.get_logger().info(
                "TABLE COLLISION REMOVED"
            )

            return response

        finally:
            self.operation_lock.release()


def main(args=None):
    rclpy.init(args=args)

    node = TableSceneNode()

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
