import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
)
from rclpy.time import Time
from std_msgs.msg import Bool, Empty
from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)

class HandshakePlanner(Node):
    def __init__(self):
        super().__init__("handshake_planner")
        # ==========================================================
        # Parameters
        # ==========================================================

        self.declare_parameter(
            "base_frame",
            "base_link"
        )
        self.declare_parameter(
            "ee_frame",
            "gripper_base"
        )
        # 机械臂首先停在距离手掌 15 cm 的地方
        self.declare_parameter(
            "approach_distance_m",
            0.15
        )

        # 最终握手位置：
        # 第一版仍然离手 5 cm，不真正接触
        self.declare_parameter(
            "handshake_clearance_m",
            0.05
        )
        self.base_frame = str(
            self.get_parameter(
                "base_frame"
            ).value
        )
        self.ee_frame = str(
            self.get_parameter(
                "ee_frame"
            ).value
        )
        self.approach_distance = float(
            self.get_parameter(
                "approach_distance_m"
            ).value
        )
        self.handshake_clearance = float(
            self.get_parameter(
                "handshake_clearance_m"
            ).value
        )

        # ==========================================================
        # State
        # ==========================================================
        self.plan_generated = False
        self.approach_pose = None
        self.handshake_pose = None

        # ==========================================================
        # TF
        # ==========================================================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ==========================================================
        # QoS
        # ==========================================================

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ==========================================================
        # Input
        # ==========================================================

        self.create_subscription(
            PoseStamped,
            "/handshake/locked_pose_base",
            self.target_callback,
            10
        )

        self.create_subscription(
            Empty,
            "/handshake/reset_event",
            self.reset_callback,
            10
        )

        # ==========================================================
        # Output
        # ==========================================================
        self.approach_pub = self.create_publisher(
            PoseStamped,
            "/handshake/approach_pose",
            qos
        )
        self.handshake_pub = self.create_publisher(
            PoseStamped,
            "/handshake/handshake_pose",
            qos
        )
        self.ready_pub = self.create_publisher(
            Bool,
            "/handshake/plan_ready",
            qos
        )

        # 周期重发，方便 ros2 topic echo / RViz
        self.timer = self.create_timer(
            0.2,
            self.timer_callback
        )
        self.get_logger().info(
            "Handshake planner started"
        )
        self.get_logger().info(
            f"Approach distance: "
            f"{self.approach_distance * 100:.1f} cm"
        )
        self.get_logger().info(
            f"Handshake clearance: "
            f"{self.handshake_clearance * 100:.1f} cm"
        )

    # ==============================================================
    # Generate handshake plan
    # ==============================================================

    def target_callback(
        self,
        msg: PoseStamped
    ):
        if self.plan_generated:
            return
        if msg.header.frame_id != self.base_frame:
            self.get_logger().error(
                f"Expected frame '{self.base_frame}', "
                f"but got '{msg.header.frame_id}'"
            )
            return

        # ----------------------------------------------------------
        # Get current gripper pose from TF
        # ----------------------------------------------------------
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                Time(),
                timeout=Duration(
                    seconds=0.5
                )
            )

        except TransformException as e:
            self.get_logger().error(
                f"Cannot get "
                f"{self.base_frame} -> "
                f"{self.ee_frame}: {e}"
            )
            return
        current = np.array(
            [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ],
            dtype=np.float64
        )
        hand = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64
        )

        # ----------------------------------------------------------
        # Current gripper -> Hand
        # ----------------------------------------------------------
        direction = hand - current
        distance = float(
            np.linalg.norm(direction)
        )
        if distance < 1e-6:
            self.get_logger().error(
                "Hand target and gripper position "
                "are nearly identical."
            )
            return
        unit_direction = (
            direction / distance
        )

        # ----------------------------------------------------------
        # Safety check
        # ----------------------------------------------------------
        if distance <= self.approach_distance:
            self.get_logger().error(
                "Hand is already closer than the "
                "approach distance."
            )
            return

        # ----------------------------------------------------------
        # Generate approach position
        #
        # 机械臂沿当前夹爪 -> 手掌方向靠近，
        # 但提前15 cm停止。
        # ----------------------------------------------------------

        approach = (
            hand
            -
            unit_direction
            * self.approach_distance
        )

        # ----------------------------------------------------------
        # Generate handshake position
        #
        # 第一版仍然在手前5cm处。
        # ----------------------------------------------------------

        handshake = (
            hand
            -
            unit_direction
            * self.handshake_clearance
        )

        # ==========================================================
        # Preserve current end-effector orientation
        # ==========================================================
        q = tf.transform.rotation

        # ----------------------------------------------------------
        # Approach Pose
        # ----------------------------------------------------------
        approach_msg = PoseStamped()
        approach_msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )
        approach_msg.header.frame_id = (
            self.base_frame
        )
        approach_msg.pose.position.x = float(
            approach[0]
        )
        approach_msg.pose.position.y = float(
            approach[1]
        )
        approach_msg.pose.position.z = float(
            approach[2]
        )
        approach_msg.pose.orientation.x = q.x
        approach_msg.pose.orientation.y = q.y
        approach_msg.pose.orientation.z = q.z
        approach_msg.pose.orientation.w = q.w

        # ----------------------------------------------------------
        # Handshake Pose
        # ----------------------------------------------------------
        handshake_msg = PoseStamped()
        handshake_msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )
        handshake_msg.header.frame_id = (
            self.base_frame
        )
        handshake_msg.pose.position.x = float(
            handshake[0]
        )
        handshake_msg.pose.position.y = float(
            handshake[1]
        )
        handshake_msg.pose.position.z = float(
            handshake[2]
        )
        handshake_msg.pose.orientation.x = q.x
        handshake_msg.pose.orientation.y = q.y
        handshake_msg.pose.orientation.z = q.z
        handshake_msg.pose.orientation.w = q.w

        # ----------------------------------------------------------
        # Freeze plan
        # ----------------------------------------------------------
        self.approach_pose = approach_msg
        self.handshake_pose = handshake_msg
        self.plan_generated = True
        self.approach_pub.publish(
            approach_msg
        )
        self.handshake_pub.publish(
            handshake_msg
        )
        ready = Bool()
        ready.data = True
        self.ready_pub.publish(
            ready
        )

        # ----------------------------------------------------------
        # Information
        # ----------------------------------------------------------
        self.get_logger().info(
            "======================================"
        )
        self.get_logger().info(
            "HANDSHAKE PLAN GENERATED"
        )
        self.get_logger().info(
            f"Current gripper = "
            f"[{current[0]:+.4f}, "
            f"{current[1]:+.4f}, "
            f"{current[2]:+.4f}] m"
        )
        self.get_logger().info(
            f"Locked hand = "
            f"[{hand[0]:+.4f}, "
            f"{hand[1]:+.4f}, "
            f"{hand[2]:+.4f}] m"
        )
        self.get_logger().info(
            f"Distance to hand = "
            f"{distance * 100:.1f} cm"
        )
        self.get_logger().info(
            f"Approach = "
            f"[{approach[0]:+.4f}, "
            f"{approach[1]:+.4f}, "
            f"{approach[2]:+.4f}] m"
        )
        self.get_logger().info(
            f"Handshake = "
            f"[{handshake[0]:+.4f}, "
            f"{handshake[1]:+.4f}, "
            f"{handshake[2]:+.4f}] m"
        )
        self.get_logger().info(
            "NO ROBOT MOTION IN STEP 3"
        )
        self.get_logger().info(
            "======================================"
        )

    # ==============================================================
    # Republish frozen plan
    # ==============================================================
    def timer_callback(self):
        if not self.plan_generated:
            return
        now = (
            self.get_clock()
            .now()
            .to_msg()
        )
        self.approach_pose.header.stamp = now
        self.handshake_pose.header.stamp = now
        self.approach_pub.publish(
            self.approach_pose
        )
        self.handshake_pub.publish(
            self.handshake_pose
        )
        ready = Bool()
        ready.data = True
        self.ready_pub.publish(
            ready
        )

    # ==============================================================
    # Reset
    # ==============================================================
    def reset_callback(
        self,
        msg: Empty
    ):
        self.plan_generated = False
        self.approach_pose = None
        self.handshake_pose = None
        ready = Bool()
        ready.data = False
        self.ready_pub.publish(
            ready
        )
        self.get_logger().info(
            "Handshake plan cleared"
        )


def main(args=None):
    rclpy.init(args=args)
    node = HandshakePlanner()
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
