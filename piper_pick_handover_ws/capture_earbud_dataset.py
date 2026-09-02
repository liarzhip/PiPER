import os
import cv2
import time
import numpy as np

from pyorbbecsdk import Pipeline, Config, OBSensorType
from utils import frame_to_bgr_image

SAVE_DIR = os.path.expanduser(
    "~/PiPER_X/piper_pick_handover_ws/datasets/earbud_case_raw"
)
IMAGE_PREFIX = "earbud"
JPEG_QUALITY = 95
AUTO_INTERVAL_S = 1.0
MIN_DEPTH_MM = 20.0
MAX_DEPTH_MM = 5000.0
ESC_KEY = 27


def depth_to_colormap(depth_mm):
    valid = np.where(
        (depth_mm >= MIN_DEPTH_MM) & (depth_mm <= MAX_DEPTH_MM),
        depth_mm,
        0.0,
    )
    depth_8u = cv2.normalize(
        valid, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)


def find_next_index(save_dir):
    max_index = -1
    if not os.path.isdir(save_dir):
        return 0

    for name in os.listdir(save_dir):
        if not (name.startswith(f"{IMAGE_PREFIX}_") and name.endswith(".jpg")):
            continue
        try:
            idx = int(os.path.splitext(name)[0].split("_")[-1])
            max_index = max(max_index, idx)
        except ValueError:
            pass

    return max_index + 1


def save_training_image(color_image, index):
    filename = f"{IMAGE_PREFIX}_{index:05d}.jpg"
    path = os.path.join(SAVE_DIR, filename)
    ok = cv2.imwrite(
        path,
        color_image,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError(f"保存失败: {path}")
    print(f"[SAVE] {path}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    image_index = find_next_index(SAVE_DIR)

    # ------------------------------------------------------------
    # 直接使用 pyorbbecsdk，与 collect_rgbd.py 的启动方式一致
    # ------------------------------------------------------------
    pipeline = Pipeline()
    config = Config()

    try:
        color_profiles = pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
        color_profile = color_profiles.get_default_video_stream_profile()
        print("Color profile:", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print("[ERROR] 无法配置 Color stream:", e)
        return

    try:
        depth_profiles = pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )
        depth_profile = depth_profiles.get_default_video_stream_profile()
        print("Depth profile:", depth_profile)
        config.enable_stream(depth_profile)
    except Exception as e:
        print("[ERROR] 无法配置 Depth stream:", e)
        return

    try:
        pipeline.start(config)
    except Exception as e:
        print("[ERROR] Pipeline 启动失败:", e)
        return

    print("\n========================================")
    print("DaBai DC1 耳机盒训练数据采集")
    print("保存目录:", SAVE_DIR)
    print("S / SPACE : 保存一张 RGB 图片")
    print("A         : 开关自动采集（每 %.1f 秒）" % AUTO_INTERVAL_S)
    print("Q / ESC   : 退出")
    print("========================================\n")

    auto_capture = False
    last_auto_time = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None:
                continue

            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                continue

            debug = color_image.copy()
            depth_vis = None

            # 深度只用于现场观察，不作为 YOLO 训练图保存
            if depth_frame is not None:
                dw = depth_frame.get_width()
                dh = depth_frame.get_height()
                scale = depth_frame.get_depth_scale()

                depth_raw = np.frombuffer(
                    depth_frame.get_data(), dtype=np.uint16
                ).reshape((dh, dw))

                depth_mm = depth_raw.astype(np.float32) * scale
                depth_vis = depth_to_colormap(depth_mm)

                cx = dw // 2
                cy = dh // 2
                center_depth = float(depth_mm[cy, cx])

                cv2.circle(depth_vis, (cx, cy), 5, (255, 255, 255), 2)
                cv2.putText(
                    depth_vis,
                    f"Center {center_depth:.1f} mm",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            mode = "AUTO" if auto_capture else "MANUAL"
            text = f"Saved: {image_index} | {mode}"
            cv2.putText(
                debug, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(
                debug, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2, cv2.LINE_AA
            )
            cv2.putText(
                debug,
                "S/SPACE save | A auto | Q quit",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            now = time.monotonic()
            if auto_capture and now - last_auto_time >= AUTO_INTERVAL_S:
                save_training_image(color_image, image_index)
                image_index += 1
                last_auto_time = now

            color_display = cv2.resize(debug, (640, 480))

            if depth_vis is not None:
                depth_display = cv2.resize(depth_vis, (640, 480))
                display = np.hstack((color_display, depth_display))
            else:
                display = color_display

            cv2.imshow("DaBai DC1 Earbud Dataset Collector", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("s"), ord("S"), ord(" ")):
                save_training_image(color_image, image_index)
                image_index += 1
                last_auto_time = time.monotonic()

            elif key in (ord("a"), ord("A")):
                auto_capture = not auto_capture
                last_auto_time = time.monotonic()
                print("[AUTO]", "ON" if auto_capture else "OFF")

            elif key in (ord("q"), ord("Q"), ESC_KEY):
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("[ERROR]", e)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("DaBai DC1 已关闭")
        print("训练图片目录:", SAVE_DIR)


if __name__ == "__main__":
    main()
