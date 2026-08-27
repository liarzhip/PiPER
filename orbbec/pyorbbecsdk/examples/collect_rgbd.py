import os
import cv2
import json
import time
import numpy as np
from pyorbbecsdk import (
    Pipeline,
    Config,
    OBSensorType,
)
from utils import frame_to_bgr_image

# ============================================================
# 参数
# ============================================================

SAVE_ROOT = os.path.expanduser("~/PiPER_X/orbbec/data")
MIN_DEPTH_MM = 20
MAX_DEPTH_MM = 5000
ESC_KEY = 27


def depth_to_colormap(depth_mm):
    """
    将实际毫米深度转换成彩色图。
    仅用于显示，不用于后续深度计算。
    """
    valid = np.where(
        (depth_mm >= MIN_DEPTH_MM) &
        (depth_mm <= MAX_DEPTH_MM),
        depth_mm,
        0
    ) # 过滤调无效深度和超出范围的值
    depth_8u = cv2.normalize(
        valid,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )
    return cv2.applyColorMap(
        depth_8u,
        cv2.COLORMAP_JET
    ) # 将深度图转化为为彩色图


def save_rgbd(
    color_image,
    depth_raw,
    depth_mm,
    depth_scale,
    color_width,
    color_height,
    depth_width,
    depth_height,
):
    """
    保存一组 RGB-D 数据。
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp_ms = int(time.time() * 1000)

    folder_name = f"{timestamp}_{timestamp_ms % 1000:03d}"

    save_dir = os.path.join(
        SAVE_ROOT,
        folder_name
    )
    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. RGB 原图
    # --------------------------------------------------------
    color_path = os.path.join(
        save_dir,
        "color.png"
    )
    cv2.imwrite(
        color_path,
        color_image
    )

    # --------------------------------------------------------
    # 2. 原始深度 uint16
    #
    # 不经过缩放，保留相机原始深度数据。
    # 实际距离 = depth_raw * depth_scale
    # --------------------------------------------------------

    depth_raw_path = os.path.join(
        save_dir,
        "depth_raw.png"
    )

    cv2.imwrite(
        depth_raw_path,
        depth_raw
    )

    # --------------------------------------------------------
    # 3. 实际毫米深度
    #
    # float32 numpy 文件。
    # 后续视觉程序建议直接读取这个。
    # --------------------------------------------------------

    depth_mm_path = os.path.join(
        save_dir,
        "depth_mm.npy"
    )
    np.save(
        depth_mm_path,
        depth_mm.astype(np.float32)
    )

    # --------------------------------------------------------
    # 4. 深度伪彩色图
    #
    # 只用于人工查看。
    # --------------------------------------------------------
    depth_vis = depth_to_colormap(
        depth_mm
    )

    depth_vis_path = os.path.join(
        save_dir,
        "depth_vis.png"
    )

    cv2.imwrite(
        depth_vis_path,
        depth_vis
    )

    # --------------------------------------------------------
    # 5. 元数据
    # --------------------------------------------------------
    metadata = {
        "timestamp_ms": timestamp_ms,

        "depth_scale": float(depth_scale),

        "color": {
            "width": int(color_width),
            "height": int(color_height),
            "file": "color.png",
        },

        "depth": {
            "width": int(depth_width),
            "height": int(depth_height),
            "raw_file": "depth_raw.png",
            "mm_file": "depth_mm.npy",
            "visualization_file": "depth_vis.png",
        },

        "depth_unit": "mm",
    }

    metadata_path = os.path.join(
        save_dir,
        "meta.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n========================================")
    print("RGB-D 数据保存成功")
    print("目录:", save_dir)
    print("========================================")
    print("RGB       :", color_path)
    print("Depth raw :", depth_raw_path)
    print("Depth mm  :", depth_mm_path)
    print("Depth vis :", depth_vis_path)
    print("Metadata  :", metadata_path)
    print("========================================\n")


def main():
    os.makedirs(
        SAVE_ROOT,
        exist_ok=True
    )
    pipeline = Pipeline() # 连接相机
    config = Config() # 创建配置对象

    # ========================================================
    # 1. 获取 Color stream
    # ========================================================
    try:
        color_profiles = (
            pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR #  彩色相机传感器
            )
        )
        color_profile = (
            color_profiles.get_default_video_stream_profile()
        )
        print(
            "Color profile:",
            color_profile
        )
        config.enable_stream(
            color_profile
        )
    except Exception as e:
        print(
            "[ERROR] 无法配置 Color stream:"
        )
        print(e)

        return

    # ========================================================
    # 2. 获取 Depth stream
    # ========================================================
    try:
        depth_profiles = (
            pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR # 深度传感器
            )
        )
        depth_profile = (
            depth_profiles.get_default_video_stream_profile()
        )
        print(
            "Depth profile:",
            depth_profile
        )
        config.enable_stream(
            depth_profile
        )
    except Exception as e:
        print(
            "[ERROR] 无法配置 Depth stream:"
        )
        print(e)

        return

    # ========================================================
    # 3. 启动相机
    # ========================================================
    try:
        pipeline.start( 
            config
        ) # 按照配置启动相机

    except Exception as e:
        print(
            "[ERROR] Pipeline 启动失败:"
        )
        print(e)

        return

    print("\nDaBai DC1 RGB-D 采集已启动")
    print("--------------------------------")
    print("S     : 保存当前 RGB-D")
    print("Q/ESC : 退出")
    print("--------------------------------\n")

    # ========================================================
    # 4. 主循环
    # ========================================================
    while True:
        try:
            frames = pipeline.wait_for_frames(
                1000
            ) # 等待一组帧，最长等待时间1s
            if frames is None:
                continue

            # ------------------------------------------------
            # Color
            # ------------------------------------------------
            color_frame = (
                frames.get_color_frame()
            )

            # ------------------------------------------------
            # Depth
            # ------------------------------------------------
            depth_frame = (
                frames.get_depth_frame()
            )
            if (
                color_frame is None
                or depth_frame is None
            ):
                continue

            # =================================================
            # Color → OpenCV BGR
            # =================================================
            color_image = (
                frame_to_bgr_image(
                    color_frame
                )
            )
            if color_image is None:
                continue

            # =================================================
            # Depth → numpy
            # =================================================
            depth_width = (
                depth_frame.get_width()
            )
            depth_height = (
                depth_frame.get_height()
            )
            depth_scale = (
                depth_frame.get_depth_scale()
            )
            depth_raw = np.frombuffer(
                depth_frame.get_data(),
                dtype=np.uint16
            ).reshape(
                (
                    depth_height,
                    depth_width
                )
            ).copy()

            # 实际毫米距离
            depth_mm = (
                depth_raw.astype(np.float32)
                * depth_scale
            )

            # =================================================
            # 中心点深度
            # =================================================
            center_x = depth_width // 2
            center_y = depth_height // 2

            center_depth = (
                depth_mm[
                    center_y,
                    center_x
                ]
            )

            # =================================================
            # Depth 显示
            # =================================================
            depth_vis = depth_to_colormap(
                depth_mm
            )
            cv2.circle(
                depth_vis,
                (center_x, center_y),
                5,
                (255, 255, 255),
                2
            )
            cv2.putText(
                depth_vis,
                f"{center_depth:.1f} mm",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # =================================================
            # 为显示统一尺寸
            #
            # 注意：
            # 这里只 resize 显示图。
            # 保存的数据保持原始分辨率。
            # =================================================
            display_width = 640
            display_height = 480

            color_display = cv2.resize(
                color_image,
                (
                    display_width,
                    display_height
                )
            )
            depth_display = cv2.resize(
                depth_vis,
                (
                    display_width,
                    display_height
                )
            )
            combined = np.hstack(
                (
                    color_display,
                    depth_display
                )
            )
            cv2.imshow(
                "DaBai DC1 RGB-D Collector",
                combined
            )

            # =================================================
            # 键盘
            # =================================================
            key = cv2.waitKey(1) & 0xFF
            # S：保存
            if key in (
                ord("s"),
                ord("S")
            ):
                save_rgbd(
                    color_image=color_image,
                    depth_raw=depth_raw,
                    depth_mm=depth_mm,
                    depth_scale=depth_scale,
                    color_width=color_frame.get_width(),
                    color_height=color_frame.get_height(),
                    depth_width=depth_width,
                    depth_height=depth_height,
                )

            # Q / ESC：退出
            elif key in (
                ord("q"),
                ord("Q"),
                ESC_KEY
            ):
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(
                "[ERROR]",
                e
            )

    # ========================================================
    # 5. 关闭设备
    # ========================================================
    pipeline.stop()
    cv2.destroyAllWindows()
    print(
        "DaBai DC1 已关闭"
    )

if __name__ == "__main__":
    main()