import cv2
import numpy as np
from pyorbbecsdk import Config, OBAlignMode, OBFormat, OBSensorType, Pipeline

def frame_to_bgr(frame):
    w, h, fmt = frame.get_width(), frame.get_height(), frame.get_format()
    data = np.asanyarray(frame.get_data())
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(np.resize(data,(h,w,3)), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return np.resize(data,(h,w,3))
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(np.resize(data,(h,w,2)), cv2.COLOR_YUV2BGR_YUY2)
    if fmt == OBFormat.UYVY:
        return cv2.cvtColor(np.resize(data,(h,w,2)), cv2.COLOR_YUV2BGR_UYVY)
    raise RuntimeError(f"Unsupported color format: {fmt}")

class OrbbecRGBDCamera:
    def __init__(self, align_mode="HW"):
        self.pipeline = Pipeline()
        self.config = Config()
        self.align_mode = align_mode.upper()
        self.started = False

    def start(self):
        colors = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depths = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        self.color_profile = colors.get_default_video_stream_profile()
        self.depth_profile = depths.get_default_video_stream_profile()
        self.config.enable_stream(self.color_profile)
        self.config.enable_stream(self.depth_profile)

        mode = {"HW": OBAlignMode.HW_MODE, "SW": OBAlignMode.SW_MODE}.get(
            self.align_mode, OBAlignMode.DISABLE
        )
        self.config.set_align_mode(mode)
        try:
            self.pipeline.enable_frame_sync()
        except Exception as e:
            print(f"[Orbbec] frame sync warning: {e}")

        self.pipeline.start(self.config)
        self.started = True
        self.color_intrinsics = self.color_profile.get_intrinsic()

    def read(self, timeout_ms=100):
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None
        cf, df = frames.get_color_frame(), frames.get_depth_frame()
        if cf is None or df is None:
            return None
        color = frame_to_bgr(cf)
        dh, dw = df.get_height(), df.get_width()
        depth = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(dh,dw)
        depth_mm = depth.astype(np.float32) * float(df.get_depth_scale())
        if color.shape[:2] != depth_mm.shape[:2]:
            raise RuntimeError(
                f"RGB/Depth not aligned: color={color.shape[1::-1]}, "
                f"depth={depth_mm.shape[::-1]}. Try align_mode HW/SW."
            )
        return color, depth_mm

    def stop(self):
        if self.started:
            self.pipeline.stop()
            self.started = False
