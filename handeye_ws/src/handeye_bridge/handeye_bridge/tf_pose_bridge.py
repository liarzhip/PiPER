import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Pose
from tf2_ros import Buffer, TransformListener


class TfPoseBridge(Node):

    def __init__(self):
        super().__init__('tf_pose_bridge')

        self.base_frame = 'base_link'
        self.ee_frame = 'gripper_base'

        self.publisher = self.create_publisher(
            Pose,
            '/handeye/ee_pose',
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.timer = self.create_timer(
            0.05,
            self.timer_callback
        )

        self.get_logger().info(
            'Publishing base_link -> gripper_base '
            'to /handeye/ee_pose'
        )

    def timer_callback(self):

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                Time()
            )

        except Exception as e:
            self.get_logger().warning(
                f'Cannot get TF: {e}'
            )
            return

        msg = Pose()

        msg.position.x = tf.transform.translation.x
        msg.position.y = tf.transform.translation.y
        msg.position.z = tf.transform.translation.z

        msg.orientation.x = tf.transform.rotation.x
        msg.orientation.y = tf.transform.rotation.y
        msg.orientation.z = tf.transform.rotation.z
        msg.orientation.w = tf.transform.rotation.w

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = TfPoseBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
