import cv2
from ultralytics import YOLO
from pyorbbecsdk import Pipeline, Config, OBSensorType
from utils import frame_to_bgr_image

MODEL_PATH = (
    "/home/liar/PiPER_X/piper_pick_handover_ws/"
    "src/piper_pick_handover/models/earbud_case_seg.pt"
)
CONF = 0.50
IMGSZ = 640
DEVICE = "cpu"
ESC_KEY = 27


def main():
    model = YOLO(MODEL_PATH)
    print("Model loaded")
    print("Classes:", model.names)

    pipeline = Pipeline()
    config = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_default_video_stream_profile()
    print("Color profile:", color_profile)
    config.enable_stream(color_profile)

    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_default_video_stream_profile()
    print("Depth profile:", depth_profile)
    config.enable_stream(depth_profile)

    pipeline.start(config)

    print("\nDaBai DC1 + YOLO-Seg started")
    print("Q / ESC : quit\n")

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                continue

            results = model.predict(
                source=color_image,
                conf=CONF,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False,
            )

            result = results[0]
            annotated = result.plot()

            object_count = 0 if result.boxes is None else len(result.boxes)
            mask_status = result.masks is not None

            cv2.putText(
                annotated,
                f"Objects={object_count} Mask={'YES' if mask_status else 'NO'} CPU imgsz={IMGSZ}",
                (20, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("DaBai DC1 + Earbud Case YOLO-Seg", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), ESC_KEY):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("DaBai DC1 closed")


if __name__ == "__main__":
    main()
