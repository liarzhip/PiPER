import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from .depth_utils import median_depth_mm
from .filters import EMAFilter
from .hand_detector import HandDetector
from .orbbec_camera import OrbbecRGBDCamera

def rotation_matrix_to_quaternion(R):

    trace = float(
        R[0, 0]
        + R[1, 1]
        + R[2, 2]
    )

    if trace > 0.0:
        s = (
            np.sqrt(trace + 1.0)
            * 2.0
        )
        qw = 0.25 * s
        qx = (
            R[2, 1] - R[1, 2]
        ) / s
        qy = (
            R[0, 2] - R[2, 0]
        ) / s
        qz = (
            R[1, 0] - R[0, 1]
        ) / s
    elif (
        R[0, 0] > R[1, 1]
        and
        R[0, 0] > R[2, 2]
    ):
        s = np.sqrt(
            1.0
            + R[0, 0]
            - R[1, 1]
            - R[2, 2]
        ) * 2.0
        qw = (
            R[2, 1] - R[1, 2]
        ) / s
        qx = 0.25 * s
        qy = (
            R[0, 1] + R[1, 0]
        ) / s
        qz = (
            R[0, 2] + R[2, 0]
        ) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(
            1.0
            + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        ) * 2.0
        qw = (
            R[0, 2] - R[2, 0]
        ) / s
        qx = (
            R[0, 1] + R[1, 0]
        ) / s
        qy = 0.25 * s
        qz = (
            R[1, 2] + R[2, 1]
        ) / s
    else:
        s = np.sqrt(
            1.0
            + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        ) * 2.0
        qw = (
            R[1, 0] - R[0, 1]
        ) / s
        qx = (
            R[0, 2] + R[2, 0]
        ) / s
        qy = (
            R[1, 2] + R[2, 1]
        ) / s
        qz = 0.25 * s

    q = np.array(
        [qx, qy, qz, qw],
        dtype=np.float64,
    )
    q /= np.linalg.norm(q)

    return q

class HandVisionNode(Node):
    """
    DaBai DC1 + MediaPipe hand tracking node.

    Important:
    - The camera is opened directly by pyorbbecsdk through OrbbecRGBDCamera.
      Therefore normal hand-follow operation does NOT need
      `ros2 launch orbbec_camera dabai.launch.py`.
    - RGB/Depth alignment is still handled by OrbbecRGBDCamera.
    - Palm pixel distortion is corrected internally with OpenCV before
      back-projecting the point to 3D.
    """

    def __init__(self):
        super().__init__("hand_vision_node")

        # ------------------------------------------------------------
        # Basic parameters
        # ------------------------------------------------------------
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("align_mode", "HW")
        self.declare_parameter("show_window", True)
        self.declare_parameter("depth_roi_radius", 7)
        self.declare_parameter("position_filter_alpha", 0.15)

        # ------------------------------------------------------------
        # Internal lens-distortion correction
        #
        # These are the DaBai DC1 color-camera D coefficients that were
        # read from /camera/color/camera_info during calibration.
        #
        # distortion_model: rational_polynomial
        #
        # OpenCV supports 8 coefficients:
        # [k1, k2, p1, p2, k3, k4, k5, k6]
        # ------------------------------------------------------------
        self.declare_parameter("enable_undistortion", True)
        self.declare_parameter(
            "distortion_coeffs",
            [
                -0.07695747166872025,
                -0.10455092042684555,
                0.0004042932123411447,
                -3.3492169677629136e-06,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )

        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.show = bool(self.get_parameter("show_window").value)
        self.radius = int(self.get_parameter("depth_roi_radius").value)
        self.enable_undistortion = bool(
            self.get_parameter("enable_undistortion").value
        )
        self.distortion_coeffs = np.asarray(
            self.get_parameter("distortion_coeffs").value,
            dtype=np.float64,
        ).reshape(-1)

        # ------------------------------------------------------------
        # Camera
        # ------------------------------------------------------------
        self.camera = OrbbecRGBDCamera(
            self.get_parameter("align_mode").value
        )
        self.camera.start()

        # Color intrinsics are obtained directly from pyorbbecsdk.
        intr = self.camera.color_intrinsics

        self.camera_matrix = np.array(
            [
                [float(intr.fx), 0.0, float(intr.cx)],
                [0.0, float(intr.fy), float(intr.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.get_logger().info(
            "Color intrinsics: "
            f"fx={intr.fx:.3f}, fy={intr.fy:.3f}, "
            f"cx={intr.cx:.3f}, cy={intr.cy:.3f}"
        )

        if self.enable_undistortion:
            self.get_logger().info(
                "Internal pixel undistortion enabled. "
                f"D={self.distortion_coeffs.tolist()}"
            )
        else:
            self.get_logger().warning(
                "Internal pixel undistortion disabled."
            )

        # ------------------------------------------------------------
        # Hand detector / filter / ROS
        # ------------------------------------------------------------
        self.detector = HandDetector()
        self.filter = EMAFilter(
            float(self.get_parameter("position_filter_alpha").value)
        )

        self.pub = self.create_publisher(
            PoseStamped,
            "/hand/pose_camera",
            10,
        )

        self.palm_pose_pub = (
            self.create_publisher(
                PoseStamped,
                "/handshake/palm_pose_camera",
                10
            )
        )

        self.timer = self.create_timer(
            1.0 / 30.0,
            self.tick,
        )
    def landmark_to_3d(
        self,
        depth_mm,
        pixel
    ):
        u, v = pixel
        z_mm = median_depth_mm(
            depth_mm,
            u,
            v,
            self.radius,
        )
        if z_mm is None:
            return None
        return self.deproject_undistorted(
            u,
            v,
            z_mm,
        )

    def undistort_to_normalized(self, u: float, v: float):
        """
        Convert a distorted color pixel (u, v) into normalized ideal
        pinhole coordinates (xn, yn).

        After this:
            X = xn * Z
            Y = yn * Z
            Z = Z

        This is preferable to using the raw distorted pixel directly in
        X=(u-cx)Z/fx and Y=(v-cy)Z/fy.
        """
        if not self.enable_undistortion:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

            xn = (float(u) - cx) / fx
            yn = (float(v) - cy) / fy
            return xn, yn

        pixel = np.array(
            [[[float(u), float(v)]]],
            dtype=np.float64,
        )

        # P=None makes OpenCV return normalized undistorted coordinates.
        normalized = cv2.undistortPoints(
            pixel,
            self.camera_matrix,
            self.distortion_coeffs,
            P=None,
        )

        xn = float(normalized[0, 0, 0])
        yn = float(normalized[0, 0, 1])

        return xn, yn

    def deproject_undistorted(self, u: float, v: float, z_mm: float):
        """
        Back-project the palm center to camera optical XYZ.

        NOTE:
        Depth is sampled at the ORIGINAL aligned pixel (u, v).
        Only the geometric ray is undistorted.
        """
        xn, yn = self.undistort_to_normalized(u, v)

        z_m = float(z_mm) / 1000.0

        return np.array(
            [
                xn * z_m,
                yn * z_m,
                z_m,
            ],
            dtype=np.float64,
        )

    def tick(self):
        packet = self.camera.read(50)

        if packet is None:
            return

        color, depth_mm = packet

        det, debug = self.detector.detect(color)

        if det:
            # Depth must be looked up at the original RGB-aligned pixel.
            z_mm = median_depth_mm(
                depth_mm,
                det.u,
                det.v,
                self.radius,
            )

            if z_mm is not None:
                xyz_raw = self.deproject_undistorted(
                    det.u,
                    det.v,
                    z_mm,
                )

                xyz = self.filter.update(xyz_raw)

                msg = PoseStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.camera_frame

                msg.pose.position.x = float(xyz[0])
                msg.pose.position.y = float(xyz[1])
                msg.pose.position.z = float(xyz[2])
                msg.pose.orientation.w = 1.0

                required_ids = [
                    0,      # Wrist
                    5,      # Index MCP
                    9,      # Middle MCP
                    13,     # Ring MCP
                    17,     # Pinky MCP
                ]
                points = {}
                valid = True
                for landmark_id in required_ids:
                    p = self.landmark_to_3d(
                        depth_mm,
                        det.landmarks[
                            landmark_id
                        ],
                    )
                    if p is None:
                        valid = False
                        break
                    points[
                        landmark_id
                    ] = p
                if valid:
                    P0 = points[0]
                    P5 = points[5]
                    P9 = points[9]
                    P13 = points[13]
                    P17 = points[17]

                    # ======================================================
                    # Palm Center
                    # ======================================================
                    palm_center = np.mean(
                        np.stack(
                            [
                                P0,
                                P5,
                                P9,
                                P13,
                                P17,
                            ]
                        ),
                        axis=0,
                    )

                    # ======================================================
                    # Palm X
                    #
                    # Pinky MCP -> Index MCP
                    # ======================================================
                    x_palm = (
                        P5 - P17
                    )
                    x_norm = np.linalg.norm(
                        x_palm
                    )
                    if x_norm < 1e-6:
                        return
                    x_palm /= x_norm

                    # ======================================================
                    # Palm Y
                    #
                    # Wrist -> Middle MCP
                    # ======================================================
                    y_hint = (
                        P9 - P0
                    )
                    # 去除在X方向的投影
                    # 保证X/Y正交
                    y_palm = (
                        y_hint
                        -
                        np.dot(
                            y_hint,
                            x_palm
                        )
                        * x_palm
                    )
                    y_norm = np.linalg.norm(
                        y_palm
                    )
                    if y_norm < 1e-6:
                        return
                    y_palm /= y_norm

                    # ======================================================
                    # Palm Normal
                    # ======================================================
                    z_palm = np.cross(
                        x_palm,
                        y_palm
                    )
                    z_norm = np.linalg.norm(
                        z_palm
                    )
                    if z_norm < 1e-6:
                        return
                    z_palm /= z_norm

                    # ======================================================
                    # Normal方向统一
                    #
                    # Camera optical frame:
                    # palm_center 是 Camera -> Hand
                    #
                    # 我们让 Z_palm 始终朝向 Camera。
                    # ======================================================
                    if (
                        np.dot(
                            z_palm,
                            palm_center
                        )
                        > 0
                    ):
                        # 同时翻转 X 和 Z，
                        # 保持右手坐标系
                        x_palm = -x_palm
                        z_palm = -z_palm

                    # 再正交一次
                    y_palm = np.cross(
                        z_palm,
                        x_palm
                    )

                    y_palm /= np.linalg.norm(
                        y_palm
                    )

                    # ======================================================
                    # Rotation Matrix
                    #
                    # 每一列分别是 Palm Frame 的 XYZ
                    # 在 Camera Frame 中的方向
                    # ======================================================
                    R_camera_palm = np.column_stack(
                        (
                            x_palm,
                            y_palm,
                            z_palm,
                        )
                    )
                    q = rotation_matrix_to_quaternion(
                        R_camera_palm
                    )

                    # ======================================================
                    # Publish Palm Pose
                    # ======================================================
                    palm_msg = PoseStamped()
                    palm_msg.header.stamp = (
                        self.get_clock()
                        .now()
                        .to_msg()
                    )
                    palm_msg.header.frame_id = (
                        self.camera_frame
                    )
                    palm_msg.pose.position.x = float(
                        palm_center[0]
                    )
                    palm_msg.pose.position.y = float(
                        palm_center[1]
                    )
                    palm_msg.pose.position.z = float(
                        palm_center[2]
                    )
                    palm_msg.pose.orientation.x = float(
                        q[0]
                    )
                    palm_msg.pose.orientation.y = float(
                        q[1]
                    )
                    palm_msg.pose.orientation.z = float(
                        q[2]
                    )
                    palm_msg.pose.orientation.w = float(
                        q[3]
                    )
                    self.palm_pose_pub.publish(
                        palm_msg
                    )

                self.pub.publish(msg)

                cv2.putText(
                    debug,
                    (
                        f"XYZ {xyz[0]:+.3f} "
                        f"{xyz[1]:+.3f} "
                        f"{xyz[2]:.3f} m"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    debug,
                    (
                        f"Palm px=({det.u},{det.v}) "
                        f"Z={z_mm / 1000.0:.3f}m "
                        f"undistort={'ON' if self.enable_undistortion else 'OFF'}"
                    ),
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 100),
                    2,
                )
        else:
            self.filter.reset()

        if self.show:
            cv2.imshow(
                "DaBai DC1 Hand Tracking",
                debug,
            )

            if (
                cv2.waitKey(1) & 0xFF
            ) in (
                ord("q"),
                ord("Q"),
                27,
            ):
                rclpy.shutdown()

    def destroy_node(self):
        self.camera.stop()
        self.detector.close()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = HandVisionNode()

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