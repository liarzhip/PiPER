import os
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from std_srvs.srv import SetBool, Trigger

from pyorbbecsdk import (
    Pipeline,
    Config,
    OBSensorType,
)

# Optional APIs vary slightly between pyorbbecsdk releases.
try:
    from pyorbbecsdk import (
        AlignFilter,
        OBStreamType,
    )
except ImportError:
    AlignFilter = None
    OBStreamType = None

try:
    from pyorbbecsdk import (
        OBFrameAggregateOutputMode,
    )
except ImportError:
    OBFrameAggregateOutputMode = None

try:
    from pyorbbecsdk import (
        OBAlignMode,
    )
except ImportError:
    OBAlignMode = None

# Copy pyorbbecsdk's working utils.py into this ROS package as:
#   piper_pick_handover/orbbec_utils.py
from .orbbec_utils import frame_to_bgr_image


def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return None if n < eps else v / n


def rotation_matrix_to_quaternion(R):
    """
    3x3 rotation matrix -> quaternion [x, y, z, w].
    Matrix columns are child X/Y/Z axes expressed in parent frame.
    """
    R = np.asarray(R, dtype=np.float64)
    t = float(np.trace(R))

    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
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

    q = np.array(
        [qx, qy, qz, qw],
        dtype=np.float64,
    )
    n = float(np.linalg.norm(q))
    return None if n < 1e-12 else q / n


class ScenePerceptionNode(Node):
    """
    DaBai DC1 + YOLO-Seg + MediaPipe.

    The DaBai is opened DIRECTLY through pyorbbecsdk.
    Depth is aligned D2C (Depth -> Color), because YOLO/MediaPipe
    pixels are defined in the color image.

    Outputs:
      /perception/earbud_case_pose_camera
      /perception/palm_pose_camera
      /perception/earbud_case_detected
      /perception/palm_detected

    This node never commands robot motion.
    """

    PALM_IDS = (0, 5, 9, 13, 17)

    def __init__(self):
        super().__init__("scene_perception_node")

        # ==========================================================
        # Parameters
        # ==========================================================

        self.declare_parameter(
            "camera_frame",
            "camera_color_optical_frame",
        )
        self.declare_parameter(
            "show_window",
            True,
        )
        self.declare_parameter(
            "perception_rate_hz",
            8.0,
        )

        # Stop-and-Look control:
        # camera stream stays alive continuously, while expensive
        # YOLO/MediaPipe inference can be enabled/disabled on demand.
        self.declare_parameter(
            "perception_enabled_on_startup",
            True,
        )
        self.declare_parameter(
            "camera_reader_timeout_ms",
            1000,
        )

        self.declare_parameter(
            "depth_roi_radius",
            5,
        )
        self.declare_parameter(
            "min_depth_mm",
            80.0,
        )
        self.declare_parameter(
            "max_depth_mm",
            2500.0,
        )

        # DaBai color-camera distortion from the previous calibration.
        self.declare_parameter(
            "enable_undistortion",
            True,
        )
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

        # YOLO-Seg
        self.declare_parameter(
            "enable_case_detector",
            True,
        )
        self.declare_parameter(
            "yolo_model",
            "",
        )
        self.declare_parameter(
            "yolo_device",
            "cpu",
        )
        self.declare_parameter(
            "yolo_imgsz",
            640,
        )
        self.declare_parameter(
            "yolo_conf",
            0.55,
        )
        self.declare_parameter(
            "earbud_case_class",
            "earbud_case",
        )

        self.declare_parameter(
            "case_mask_erode_px",
            5,
        )
        self.declare_parameter(
            "case_depth_band_mm",
            35.0,
        )
        self.declare_parameter(
            "case_max_3d_points",
            2500,
        )
        self.declare_parameter(
            "case_min_3d_points",
            40,
        )

        # MediaPipe
        self.declare_parameter(
            "hand_detection_confidence",
            0.60,
        )
        self.declare_parameter(
            "hand_tracking_confidence",
            0.60,
        )

        # ==========================================================
        # Read parameters
        # ==========================================================

        self.camera_frame = str(
            self.get_parameter(
                "camera_frame"
            ).value
        )
        self.show_window = bool(
            self.get_parameter(
                "show_window"
            ).value
        )
        self.rate_hz = float(
            self.get_parameter(
                "perception_rate_hz"
            ).value
        )
        self.perception_enabled = bool(
            self.get_parameter(
                "perception_enabled_on_startup"
            ).value
        )
        self.camera_reader_timeout_ms = int(
            self.get_parameter(
                "camera_reader_timeout_ms"
            ).value
        )

        self.depth_radius = int(
            self.get_parameter(
                "depth_roi_radius"
            ).value
        )
        self.min_depth_mm = float(
            self.get_parameter(
                "min_depth_mm"
            ).value
        )
        self.max_depth_mm = float(
            self.get_parameter(
                "max_depth_mm"
            ).value
        )

        self.enable_undistortion = bool(
            self.get_parameter(
                "enable_undistortion"
            ).value
        )
        self.distortion_coeffs = np.asarray(
            self.get_parameter(
                "distortion_coeffs"
            ).value,
            dtype=np.float64,
        ).reshape(-1)

        self.enable_case_detector = bool(
            self.get_parameter(
                "enable_case_detector"
            ).value
        )
        self.yolo_model_path = str(
            self.get_parameter(
                "yolo_model"
            ).value
        )
        self.yolo_device = str(
            self.get_parameter(
                "yolo_device"
            ).value
        )
        self.yolo_imgsz = int(
            self.get_parameter(
                "yolo_imgsz"
            ).value
        )
        self.yolo_conf = float(
            self.get_parameter(
                "yolo_conf"
            ).value
        )
        self.case_class_name = str(
            self.get_parameter(
                "earbud_case_class"
            ).value
        )

        self.case_mask_erode_px = int(
            self.get_parameter(
                "case_mask_erode_px"
            ).value
        )
        self.case_depth_band_mm = float(
            self.get_parameter(
                "case_depth_band_mm"
            ).value
        )
        self.case_max_3d_points = int(
            self.get_parameter(
                "case_max_3d_points"
            ).value
        )
        self.case_min_3d_points = int(
            self.get_parameter(
                "case_min_3d_points"
            ).value
        )

        self.hand_det_conf = float(
            self.get_parameter(
                "hand_detection_confidence"
            ).value
        )
        self.hand_track_conf = float(
            self.get_parameter(
                "hand_tracking_confidence"
            ).value
        )

        # ==========================================================
        # DaBai DC1 - direct pyorbbecsdk
        # ==========================================================

        # ==========================================================
        # DaBai DC1 - direct pyorbbecsdk
        #
        # Do not create an empty startup window here.
        # cv2.imshow() will create the window only when a real frame exists.
        # ==========================================================

        self.window_name = "PIPER Pick & Handover - Scene Perception"

        self.pipeline = Pipeline()
        self.config = Config()
        self.align_filter = None

        self._configure_and_start_camera()

        # ==========================================================
        # Dedicated camera reader thread
        #
        # This thread continuously drains the SDK frameset queue and
        # keeps ONLY the newest aligned RGB-D packet.  It prevents the
        # camera producer from outrunning the slow CPU perception loop.
        # ==========================================================

        self._camera_lock = threading.Lock()
        self._camera_stop_event = threading.Event()
        self._latest_packet = None
        self._camera_seq = 0
        self._last_processed_seq = -1
        self._last_reader_error_time = 0.0

        self._camera_thread = threading.Thread(
            target=self._camera_reader_loop,
            name="dabai_rgbd_reader",
            daemon=True,
        )
        self._camera_thread.start()

        # ==========================================================
        # MediaPipe
        # ==========================================================

        import mediapipe as mp

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self._create_hand_tracker()

        # ==========================================================
        # YOLO-Seg
        # ==========================================================

        self.yolo = None

        if self.enable_case_detector:
            self._init_yolo()

        self.last_case_x_axis = None

        # ==========================================================
        # ROS
        # ==========================================================

        self.case_pose_pub = self.create_publisher(
            PoseStamped,
            "/perception/earbud_case_pose_camera",
            10,
        )
        self.palm_pose_pub = self.create_publisher(
            PoseStamped,
            "/perception/palm_pose_camera",
            10,
        )

        self.case_detected_pub = self.create_publisher(
            Bool,
            "/perception/earbud_case_detected",
            10,
        )
        self.palm_detected_pub = self.create_publisher(
            Bool,
            "/perception/palm_detected",
            10,
        )

        self.case_conf_pub = self.create_publisher(
            Float32,
            "/perception/earbud_case_confidence",
            10,
        )
        self.case_axis_pub = self.create_publisher(
            Float32,
            "/perception/earbud_case_axis_angle_deg",
            10,
        )

        # ----------------------------------------------------------
        # Stop-and-Look services
        # ----------------------------------------------------------

        self.set_enabled_srv = self.create_service(
            SetBool,
            "/perception/set_enabled",
            self.set_enabled_callback,
        )

        self.start_observation_srv = self.create_service(
            Trigger,
            "/perception/start_observation",
            self.start_observation_callback,
        )

        self.stop_observation_srv = self.create_service(
            Trigger,
            "/perception/stop_observation",
            self.stop_observation_callback,
        )

        self.timer = self.create_timer(
            1.0 / max(
                self.rate_hz,
                1.0,
            ),
            self.tick,
        )

        self.get_logger().info(
            "STEP 2 scene perception started."
        )
        self.get_logger().info(
            "DaBai opened directly via pyorbbecsdk."
        )
        self.get_logger().info(
            "YOLO device = "
            f"{self.yolo_device}"
        )
        self.get_logger().info(
            "NO ROBOT MOTION."
        )
        self.get_logger().info(
            "Stop-and-Look camera reader thread is active."
        )
        self.get_logger().info(
            "Perception inference on startup = "
            f"{self.perception_enabled}"
        )

    # ==============================================================
    # Camera
    # ==============================================================

    def _configure_and_start_camera(self):
        """
        Preferred strategy:
          software D2C AlignFilter -> color image coordinate system.

        If this SDK does not expose AlignFilter, fall back to Config
        alignment API when available.

        We NEVER silently resize an unaligned depth image for geometry.
        """

        # ----------------------------------------------------------
        # Color stream
        # ----------------------------------------------------------

        color_profiles = (
            self.pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
        )

        color_profile = (
            color_profiles
            .get_default_video_stream_profile()
        )

        self.config.enable_stream(
            color_profile
        )

        self.get_logger().info(
            f"Color profile: {color_profile}"
        )

        # ----------------------------------------------------------
        # Depth stream
        # ----------------------------------------------------------

        depth_profiles = (
            self.pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            )
        )

        depth_profile = (
            depth_profiles
            .get_default_video_stream_profile()
        )

        self.config.enable_stream(
            depth_profile
        )

        self.get_logger().info(
            f"Depth profile: {depth_profile}"
        )

        # ----------------------------------------------------------
        # Frame aggregation
        # ----------------------------------------------------------

        if (
            OBFrameAggregateOutputMode
            is not None
            and
            hasattr(
                self.config,
                "set_frame_aggregate_output_mode",
            )
        ):
            try:
                self.config.set_frame_aggregate_output_mode(
                    OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
                )
            except Exception as exc:
                self.get_logger().warning(
                    "Frame aggregate mode not applied: "
                    f"{exc}"
                )

        # ----------------------------------------------------------
        # D2C alignment
        # ----------------------------------------------------------

        if (
            AlignFilter is not None
            and
            OBStreamType is not None
        ):
            self.align_filter = AlignFilter(
                align_to_stream=(
                    OBStreamType.COLOR_STREAM
                )
            )

            self.get_logger().info(
                "RGB-D alignment: "
                "software D2C AlignFilter"
            )

        elif (
            OBAlignMode is not None
            and
            hasattr(
                self.config,
                "set_align_mode",
            )
        ):
            # SDK fallback if AlignFilter is unavailable.
            try:
                self.config.set_align_mode(
                    OBAlignMode.HW_MODE
                )

                self.get_logger().warning(
                    "AlignFilter unavailable; "
                    "using Config HW D2C mode."
                )

            except Exception as exc:
                raise RuntimeError(
                    "This pyorbbecsdk build does not provide "
                    "a usable Depth->Color alignment method."
                ) from exc

        else:
            raise RuntimeError(
                "No RGB-D alignment API is available in "
                "the installed pyorbbecsdk."
            )

        # ----------------------------------------------------------
        # Camera intrinsics
        # ----------------------------------------------------------

        try:
            intr = (
                color_profile
                .get_intrinsic()
            )
        except Exception as exc:
            raise RuntimeError(
                "Cannot read color-camera intrinsics "
                "from the selected stream profile."
            ) from exc

        self.camera_matrix = np.array(
            [
                [
                    float(intr.fx),
                    0.0,
                    float(intr.cx),
                ],
                [
                    0.0,
                    float(intr.fy),
                    float(intr.cy),
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                ],
            ],
            dtype=np.float64,
        )

        self.get_logger().info(
            "Color intrinsics: "
            f"fx={intr.fx:.3f}, "
            f"fy={intr.fy:.3f}, "
            f"cx={intr.cx:.3f}, "
            f"cy={intr.cy:.3f}"
        )

        # ----------------------------------------------------------
        # Start
        # ----------------------------------------------------------

        try:
            self.pipeline.start(
                self.config
            )
        except Exception as exc:
            raise RuntimeError(
                "DaBai pipeline start failed."
            ) from exc

    def _read_rgbd_once(self):
        """
        Read ONE fresh RGB-D frameset from DaBai.

        Important:
        - raw color/depth existence is checked BEFORE AlignFilter;
        - incomplete framesets are dropped immediately;
        - depth is NEVER resized to fake alignment.
        """
        frames = self.pipeline.wait_for_frames(
            self.camera_reader_timeout_ms
        )

        if not frames:
            return None

        # ----------------------------------------------------------
        # Validate source frames BEFORE software alignment.
        #
        # This avoids feeding obviously incomplete FrameSets to
        # AlignFilter, which is one source of "Frame is nullptr".
        # ----------------------------------------------------------

        raw_color = frames.get_color_frame()
        raw_depth = frames.get_depth_frame()

        if raw_color is None or raw_depth is None:
            return None

        # ----------------------------------------------------------
        # Software D2C alignment
        # ----------------------------------------------------------

        if self.align_filter is not None:
            aligned_frames = self.align_filter.process(
                frames
            )

            if not aligned_frames:
                return None

            frames = aligned_frames

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if (
            color_frame is None
            or
            depth_frame is None
        ):
            return None

        color = frame_to_bgr_image(
            color_frame
        )

        if color is None:
            return None

        try:
            depth_raw = np.frombuffer(
                depth_frame.get_data(),
                dtype=np.uint16,
            ).reshape(
                (
                    depth_frame.get_height(),
                    depth_frame.get_width(),
                )
            ).copy()

        except ValueError:
            return None

        depth_mm = (
            depth_raw.astype(
                np.float32
            )
            * depth_frame.get_depth_scale()
        )

        # Hard geometry check.
        if (
            color.shape[:2]
            != depth_mm.shape[:2]
        ):
            self.get_logger().error(
                "Aligned RGB-D size mismatch: "
                f"color={color.shape[:2]}, "
                f"depth={depth_mm.shape[:2]}. "
                "Dropping this frame."
            )
            return None

        return (
            color,
            depth_mm,
        )

    def _camera_reader_loop(self):
        """
        Continuously drain the camera SDK queue and retain ONLY the newest
        complete aligned RGB-D packet.

        The expensive YOLO/MediaPipe path never calls wait_for_frames().
        """
        self.get_logger().info(
            "DaBai RGB-D reader thread started."
        )

        while not self._camera_stop_event.is_set():
            try:
                packet = self._read_rgbd_once()

                if packet is None:
                    continue

                color, depth_mm = packet

                with self._camera_lock:
                    self._camera_seq += 1

                    # Whole NumPy array references are replaced atomically
                    # under the lock. Old packets are intentionally discarded.
                    self._latest_packet = (
                        self._camera_seq,
                        color,
                        depth_mm,
                    )

            except Exception as exc:
                # Do not kill the ROS node because one camera read failed.
                now = time.monotonic()

                if (
                    now - self._last_reader_error_time
                    >= 2.0
                ):
                    self.get_logger().warning(
                        "DaBai reader recovered from frame error: "
                        f"{exc}"
                    )
                    self._last_reader_error_time = now

                time.sleep(0.01)

        self.get_logger().info(
            "DaBai RGB-D reader thread stopped."
        )

    def _get_latest_packet(self):
        with self._camera_lock:
            if self._latest_packet is None:
                return None

            seq, color, depth_mm = self._latest_packet

            return (
                seq,
                color,
                depth_mm,
            )

    # ==============================================================
    # Stop-and-Look runtime control
    # ==============================================================

    def _create_hand_tracker(self):
        return self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=(
                self.hand_det_conf
            ),
            min_tracking_confidence=(
                self.hand_track_conf
            ),
        )

    def _reset_hand_tracker(self):
        try:
            self.hands.close()
        except Exception:
            pass

        self.hands = self._create_hand_tracker()

    def _publish_detection_false(self):
        msg = Bool()
        msg.data = False
        self.case_detected_pub.publish(msg)

        msg = Bool()
        msg.data = False
        self.palm_detected_pub.publish(msg)

    def _set_perception_enabled(
        self,
        enabled,
    ):
        enabled = bool(enabled)

        if enabled == self.perception_enabled:
            return

        self.perception_enabled = enabled
        self._last_processed_seq = -1

        if enabled:
            # Start a fresh hand-tracking session for a new observation
            # phase (especially important for Palm Final).
            self._reset_hand_tracker()

            self.get_logger().info(
                "PERCEPTION ENABLED | "
                "Stop-and-Look observation started."
            )

        else:
            self._publish_detection_false()

            self.get_logger().info(
                "PERCEPTION DISABLED | "
                "camera continues draining frames, "
                "YOLO/MediaPipe paused."
            )

    def set_enabled_callback(
        self,
        request,
        response,
    ):
        self._set_perception_enabled(
            request.data
        )

        response.success = True
        response.message = (
            "perception enabled"
            if self.perception_enabled
            else "perception disabled"
        )
        return response

    def start_observation_callback(
        self,
        request,
        response,
    ):
        del request

        self._set_perception_enabled(
            True
        )

        response.success = True
        response.message = (
            "Stop-and-Look observation started."
        )
        return response

    def stop_observation_callback(
        self,
        request,
        response,
    ):
        del request

        self._set_perception_enabled(
            False
        )

        response.success = True
        response.message = (
            "Stop-and-Look observation stopped. "
            "Camera reader remains active."
        )
        return response

    # ==============================================================
    # YOLO
    # ==============================================================

    def _init_yolo(self):
        model_path = Path(
            self.yolo_model_path
        ).expanduser()

        if (
            not self.yolo_model_path
            or
            not model_path.is_file()
        ):
            self.get_logger().warning(
                "YOLO-Seg model not found: "
                f"'{self.yolo_model_path}'. "
                "Case detection disabled."
            )
            self.enable_case_detector = False
            return

        from ultralytics import YOLO

        self.yolo = YOLO(
            str(model_path)
        )

        self.get_logger().info(
            f"Loaded YOLO-Seg: {model_path}"
        )
        self.get_logger().info(
            f"YOLO classes: {self.yolo.names}"
        )

    # ==============================================================
    # Geometry
    # ==============================================================

    def undistort_pixels_to_normalized(
        self,
        pixels_uv,
    ):
        pixels = np.asarray(
            pixels_uv,
            dtype=np.float64,
        ).reshape(
            -1,
            1,
            2,
        )

        if self.enable_undistortion:
            return cv2.undistortPoints(
                pixels,
                self.camera_matrix,
                self.distortion_coeffs,
                P=None,
            ).reshape(
                -1,
                2,
            )

        uv = pixels.reshape(
            -1,
            2,
        )

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        return np.column_stack(
            (
                (uv[:, 0] - cx) / fx,
                (uv[:, 1] - cy) / fy,
            )
        )

    def deproject_pixels(
        self,
        pixels_uv,
        depths_mm,
    ):
        norm = (
            self.undistort_pixels_to_normalized(
                pixels_uv
            )
        )

        z = np.asarray(
            depths_mm,
            dtype=np.float64,
        ).reshape(-1) / 1000.0

        return np.column_stack(
            (
                norm[:, 0] * z,
                norm[:, 1] * z,
                z,
            )
        )

    def median_depth_mm(
        self,
        depth_mm,
        u,
        v,
        radius=None,
    ):
        radius = (
            self.depth_radius
            if radius is None
            else radius
        )

        h, w = depth_mm.shape[:2]

        u = int(round(u))
        v = int(round(v))

        x0 = max(
            0,
            u - radius,
        )
        x1 = min(
            w,
            u + radius + 1,
        )
        y0 = max(
            0,
            v - radius,
        )
        y1 = min(
            h,
            v + radius + 1,
        )

        roi = np.asarray(
            depth_mm[y0:y1, x0:x1],
            dtype=np.float32,
        ).reshape(-1)

        valid = roi[
            np.isfinite(roi)
            & (roi >= self.min_depth_mm)
            & (roi <= self.max_depth_mm)
        ]

        if valid.size == 0:
            return None

        return float(
            np.median(valid)
        )

    def deproject_single(
        self,
        depth_mm,
        pixel_uv,
    ):
        u, v = pixel_uv

        z_mm = self.median_depth_mm(
            depth_mm,
            u,
            v,
        )

        if z_mm is None:
            return None

        return self.deproject_pixels(
            [[u, v]],
            [z_mm],
        )[0]

    # ==============================================================
    # Earbud case
    # ==============================================================

    def select_case_mask(
        self,
        color,
    ):
        if (
            not self.enable_case_detector
            or
            self.yolo is None
        ):
            return None

        result = self.yolo.predict(
            source=color,
            imgsz=self.yolo_imgsz,
            conf=self.yolo_conf,
            device=self.yolo_device,
            verbose=False,
        )[0]

        if (
            result.boxes is None
            or
            result.masks is None
            or
            len(result.boxes) == 0
        ):
            return None

        best = None

        for i in range(
            len(result.boxes)
        ):
            conf = float(
                result.boxes.conf[i]
                .detach()
                .cpu()
                .item()
            )

            cls_id = int(
                result.boxes.cls[i]
                .detach()
                .cpu()
                .item()
            )

            class_name = str(
                result.names.get(
                    cls_id,
                    cls_id,
                )
            )

            if (
                self.case_class_name
                and
                class_name
                != self.case_class_name
            ):
                continue

            polygon = (
                result.masks.xy[i]
            )

            if (
                polygon is None
                or
                len(polygon) < 3
            ):
                continue

            if (
                best is None
                or
                conf > best["conf"]
            ):
                best = {
                    "conf": conf,
                    "class_name": class_name,
                    "polygon": np.asarray(
                        polygon,
                        dtype=np.float32,
                    ),
                }

        if best is None:
            return None

        h, w = color.shape[:2]

        mask = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        polygon_i = np.round(
            best["polygon"]
        ).astype(
            np.int32
        )

        cv2.fillPoly(
            mask,
            [polygon_i],
            255,
        )

        # Image-plane major-axis angle, debug only.
        ys, xs = np.where(
            mask > 0
        )

        axis_angle = 0.0

        if xs.size >= 10:
            uv = np.column_stack(
                (xs, ys)
            ).astype(
                np.float64
            )

            _, _, vt = np.linalg.svd(
                uv - uv.mean(axis=0),
                full_matrices=False,
            )

            axis = vt[0]

            axis_angle = math.degrees(
                math.atan2(
                    axis[1],
                    axis[0],
                )
            )

            while axis_angle >= 90.0:
                axis_angle -= 180.0

            while axis_angle < -90.0:
                axis_angle += 180.0

        return (
            mask,
            best["conf"],
            best["class_name"],
            axis_angle,
        )

    def estimate_case_pose(
        self,
        mask,
        depth_mm,
    ):
        work = mask.copy()

        if self.case_mask_erode_px > 0:
            k = (
                2 * self.case_mask_erode_px
                + 1
            )

            kernel = (
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (k, k),
                )
            )

            work = cv2.erode(
                work,
                kernel,
            )

        ys, xs = np.where(
            work > 0
        )

        if (
            xs.size
            < self.case_min_3d_points
        ):
            return None

        z = depth_mm[
            ys,
            xs,
        ].astype(
            np.float64
        )

        valid = (
            np.isfinite(z)
            & (z >= self.min_depth_mm)
            & (z <= self.max_depth_mm)
        )

        xs = xs[valid]
        ys = ys[valid]
        z = z[valid]

        if (
            z.size
            < self.case_min_3d_points
        ):
            return None

        # Keep only the depth band around the segmented object's median.
        median_z = float(
            np.median(z)
        )

        band = (
            np.abs(
                z - median_z
            )
            <= self.case_depth_band_mm
        )

        xs = xs[band]
        ys = ys[band]
        z = z[band]

        if (
            z.size
            < self.case_min_3d_points
        ):
            return None

        if (
            z.size
            > self.case_max_3d_points
        ):
            ids = np.linspace(
                0,
                z.size - 1,
                self.case_max_3d_points,
                dtype=np.int64,
            )

            xs = xs[ids]
            ys = ys[ids]
            z = z[ids]

        pixels = np.column_stack(
            (xs, ys)
        ).astype(
            np.float64
        )

        points = self.deproject_pixels(
            pixels,
            z,
        )

        center = np.median(
            points,
            axis=0,
        )

        try:
            _, _, vt = np.linalg.svd(
                points - center,
                full_matrices=False,
            )
        except np.linalg.LinAlgError:
            return None

        # 3D long axis
        x_axis = normalize(
            vt[0]
        )

        # Plane normal
        z_axis = normalize(
            vt[-1]
        )

        if (
            x_axis is None
            or
            z_axis is None
        ):
            return None

        # Top-surface normal points toward camera.
        if float(
            np.dot(
                z_axis,
                center,
            )
        ) > 0.0:
            z_axis = -z_axis

        # Orthogonalize long axis.
        x_axis = normalize(
            x_axis
            -
            np.dot(
                x_axis,
                z_axis,
            )
            * z_axis
        )

        if x_axis is None:
            return None

        # Resolve 180 deg axis sign ambiguity across frames.
        if (
            self.last_case_x_axis
            is not None
        ):
            if float(
                np.dot(
                    x_axis,
                    self.last_case_x_axis,
                )
            ) < 0.0:
                x_axis = -x_axis
        else:
            if (
                x_axis[0] < 0.0
                or
                (
                    abs(x_axis[0]) < 0.05
                    and
                    x_axis[1] < 0.0
                )
            ):
                x_axis = -x_axis

        y_axis = normalize(
            np.cross(
                z_axis,
                x_axis,
            )
        )

        if y_axis is None:
            return None

        x_axis = normalize(
            np.cross(
                y_axis,
                z_axis,
            )
        )

        if x_axis is None:
            return None

        self.last_case_x_axis = (
            x_axis.copy()
        )

        R = np.column_stack(
            (
                x_axis,
                y_axis,
                z_axis,
            )
        )

        q = rotation_matrix_to_quaternion(
            R
        )

        if q is None:
            return None

        return (
            center,
            q,
        )

    # ==============================================================
    # Palm
    # ==============================================================

    def estimate_palm_pose(
        self,
        color,
        depth_mm,
        debug,
    ):
        rgb = cv2.cvtColor(
            color,
            cv2.COLOR_BGR2RGB,
        )

        result = self.hands.process(
            rgb
        )

        if not result.multi_hand_landmarks:
            return None

        hand_landmarks = (
            result.multi_hand_landmarks[0]
        )

        h, w = color.shape[:2]

        pixels = []

        for lm in hand_landmarks.landmark:
            u = max(
                0,
                min(
                    w - 1,
                    int(
                        round(
                            lm.x * (w - 1)
                        )
                    ),
                ),
            )

            v = max(
                0,
                min(
                    h - 1,
                    int(
                        round(
                            lm.y * (h - 1)
                        )
                    ),
                ),
            )

            pixels.append(
                (u, v)
            )

        points = {}

        for idx in self.PALM_IDS:
            p = self.deproject_single(
                depth_mm,
                pixels[idx],
            )

            if p is None:
                return None

            points[idx] = p

        P0 = points[0]
        P5 = points[5]
        P9 = points[9]
        P13 = points[13]
        P17 = points[17]

        center = np.mean(
            np.stack(
                (
                    P0,
                    P5,
                    P9,
                    P13,
                    P17,
                )
            ),
            axis=0,
        )

        # Palm X: pinky MCP -> index MCP
        x_axis = normalize(
            P5 - P17
        )

        if x_axis is None:
            return None

        # Palm Y: wrist -> middle MCP
        y_hint = (
            P9 - P0
        )

        y_axis = normalize(
            y_hint
            -
            np.dot(
                y_hint,
                x_axis,
            )
            * x_axis
        )

        if y_axis is None:
            return None

        z_axis = normalize(
            np.cross(
                x_axis,
                y_axis,
            )
        )

        if z_axis is None:
            return None

        # Visible palm normal toward camera.
        if float(
            np.dot(
                z_axis,
                center,
            )
        ) > 0.0:
            x_axis = -x_axis
            z_axis = -z_axis

        y_axis = normalize(
            np.cross(
                z_axis,
                x_axis,
            )
        )

        if y_axis is None:
            return None

        q = rotation_matrix_to_quaternion(
            np.column_stack(
                (
                    x_axis,
                    y_axis,
                    z_axis,
                )
            )
        )

        if q is None:
            return None

        self.mp_draw.draw_landmarks(
            debug,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
        )

        palm_u = int(
            round(
                np.mean(
                    [
                        pixels[i][0]
                        for i in self.PALM_IDS
                    ]
                )
            )
        )

        palm_v = int(
            round(
                np.mean(
                    [
                        pixels[i][1]
                        for i in self.PALM_IDS
                    ]
                )
            )
        )

        cv2.circle(
            debug,
            (
                palm_u,
                palm_v,
            ),
            7,
            (
                0,
                255,
                255,
            ),
            -1,
        )

        return (
            center,
            q,
        )

    # ==============================================================
    # ROS helpers
    # ==============================================================

    def publish_pose(
        self,
        publisher,
        xyz,
        q,
    ):
        msg = PoseStamped()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )
        msg.header.frame_id = (
            self.camera_frame
        )

        msg.pose.position.x = float(
            xyz[0]
        )
        msg.pose.position.y = float(
            xyz[1]
        )
        msg.pose.position.z = float(
            xyz[2]
        )

        msg.pose.orientation.x = float(
            q[0]
        )
        msg.pose.orientation.y = float(
            q[1]
        )
        msg.pose.orientation.z = float(
            q[2]
        )
        msg.pose.orientation.w = float(
            q[3]
        )

        publisher.publish(
            msg
        )

    @staticmethod
    def draw_text(
        image,
        text,
        y,
    ):
        cv2.putText(
            image,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # ==============================================================
    # Main loop
    # ==============================================================

    def tick(self):
        packet = self._get_latest_packet()

        if packet is None:
            return

        seq, color, depth_mm = packet

        # ----------------------------------------------------------
        # Perception paused:
        #
        # Keep the camera reader alive and optionally show the newest
        # raw frame, but do NOT run YOLO or MediaPipe.
        # ----------------------------------------------------------

        if not self.perception_enabled:
            if self.show_window:
                debug = color.copy()

                self.draw_text(
                    debug,
                    "PERCEPTION PAUSED - camera stream alive",
                    30,
                )

                cv2.imshow(
                    self.window_name,
                    debug,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    rclpy.shutdown()

            return

        # Never run expensive inference twice on the same camera frame.
        if seq == self._last_processed_seq:
            return

        self._last_processed_seq = seq

        debug = color.copy()

        # ----------------------------------------------------------
        # Earbud case
        # ----------------------------------------------------------

        case_ok = False

        selected = self.select_case_mask(
            color
        )

        if selected is not None:
            (
                mask,
                conf,
                class_name,
                axis_angle,
            ) = selected

            pose = self.estimate_case_pose(
                mask,
                depth_mm,
            )

            if pose is not None:
                xyz, q = pose
                case_ok = True

                self.publish_pose(
                    self.case_pose_pub,
                    xyz,
                    q,
                )

                conf_msg = Float32()
                conf_msg.data = float(
                    conf
                )
                self.case_conf_pub.publish(
                    conf_msg
                )

                axis_msg = Float32()
                axis_msg.data = float(
                    axis_angle
                )
                self.case_axis_pub.publish(
                    axis_msg
                )

                overlay = np.zeros_like(
                    debug
                )
                overlay[
                    mask > 0
                ] = (
                    0,
                    180,
                    0,
                )

                debug = cv2.addWeighted(
                    debug,
                    1.0,
                    overlay,
                    0.25,
                    0.0,
                )

                self.draw_text(
                    debug,
                    (
                        f"CASE {class_name} "
                        f"{conf:.2f} "
                        f"XYZ=({xyz[0]:+.3f},"
                        f"{xyz[1]:+.3f},"
                        f"{xyz[2]:.3f})m "
                        f"axis2D={axis_angle:+.1f}deg"
                    ),
                    30,
                )

        detected = Bool()
        detected.data = bool(
            case_ok
        )
        self.case_detected_pub.publish(
            detected
        )

        # ----------------------------------------------------------
        # Palm
        # ----------------------------------------------------------

        palm_ok = False

        palm_pose = self.estimate_palm_pose(
            color,
            depth_mm,
            debug,
        )

        if palm_pose is not None:
            xyz, q = palm_pose
            palm_ok = True

            self.publish_pose(
                self.palm_pose_pub,
                xyz,
                q,
            )

            self.draw_text(
                debug,
                (
                    f"PALM XYZ=({xyz[0]:+.3f},"
                    f"{xyz[1]:+.3f},"
                    f"{xyz[2]:.3f})m"
                ),
                60,
            )

        detected = Bool()
        detected.data = bool(
            palm_ok
        )
        self.palm_detected_pub.publish(
            detected
        )

        # ----------------------------------------------------------
        # Display
        # ----------------------------------------------------------

        if self.show_window:
            cv2.imshow(
                self.window_name,
                debug,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                rclpy.shutdown()

    # ==============================================================
    # Cleanup
    # ==============================================================

    def destroy_node(self):
        # Stop the reader thread first so wait_for_frames() is no longer
        # using the pipeline when pipeline.stop() is called.
        try:
            self._camera_stop_event.set()

            if (
                hasattr(self, "_camera_thread")
                and
                self._camera_thread.is_alive()
            ):
                self._camera_thread.join(
                    timeout=1.5
                )
        except Exception:
            pass

        try:
            self.pipeline.stop()
        except Exception:
            pass

        try:
            self.hands.close()
        except Exception:
            pass

        cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = ScenePerceptionNode()

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
