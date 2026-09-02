#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Labelme Polygon -> YOLO Segmentation converter

Designed for:
    piper_pick_handover_ws
    earbud_case single-class segmentation dataset

Expected Labelme source directory:
    earbud_case_labelme/
        earbud_00000.jpg
        earbud_00000.json
        earbud_00001.jpg
        earbud_00001.json
        ...

Output dataset:
    earbud_case_seg/
        images/
            train/
            val/
        labels/
            train/
            val/
        data.yaml

Supported Labelme shape types:
    polygon
    rectangle   (automatically converted to 4 polygon points)

Default class:
    earbud_case

Usage:
    python3 labelme_to_yolo_seg.py \
        --input ~/PiPER_X/piper_pick_handover_ws/datasets/earbud_case_labelme \
        --output ~/PiPER_X/piper_pick_handover_ws/datasets/earbud_case_seg \
        --val-ratio 0.2

If your JSON/image files are directly inside:
    datasets/earbud_case_raw/
you may use that directory as --input as well.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Labelme polygons to YOLO segmentation format."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing Labelme JSON files and source images.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output YOLO-Seg dataset directory.",
    )

    parser.add_argument(
        "--class-name",
        default="earbud_case",
        help="Target Labelme label / YOLO class name.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation fraction, default 0.20.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/val split.",
    )

    parser.add_argument(
        "--include-negative",
        action="store_true",
        help=(
            "Include images whose JSON contains no target object. "
            "They will receive an empty YOLO label file."
        ),
    )

    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing images/labels under the output directory first.",
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def find_image_for_json(json_path: Path, annotation: dict):
    """
    Resolve the source image corresponding to one Labelme JSON.
    Priority:
      1. Labelme imagePath
      2. Same stem with a known image extension
    """

    image_path_value = annotation.get(
        "imagePath"
    )

    if image_path_value:
        candidate = (
            json_path.parent
            / Path(image_path_value).name
        )

        if candidate.is_file():
            return candidate

    for ext in IMAGE_EXTENSIONS:
        candidate = json_path.with_suffix(
            ext
        )

        if candidate.is_file():
            return candidate

        candidate_upper = json_path.with_suffix(
            ext.upper()
        )

        if candidate_upper.is_file():
            return candidate_upper

    return None


def get_image_size(
    image_path: Path,
    annotation: dict,
):
    """
    Prefer Labelme imageWidth/imageHeight when present.
    Fall back to OpenCV.
    """

    width = annotation.get(
        "imageWidth"
    )
    height = annotation.get(
        "imageHeight"
    )

    if (
        width is not None
        and height is not None
        and int(width) > 0
        and int(height) > 0
    ):
        return int(width), int(height)

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read image: {image_path}"
        )

    height, width = image.shape[:2]

    return int(width), int(height)


def clamp01(value):
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def normalize_polygon(
    points,
    width,
    height,
):
    """
    Convert pixel coordinates to YOLO normalized polygon coordinates.

    YOLO segmentation line:
        class_id x1 y1 x2 y2 x3 y3 ...

    Coordinates must be normalized to [0, 1].
    """

    if len(points) < 3:
        return None

    normalized = []

    for point in points:
        if (
            point is None
            or len(point) < 2
        ):
            return None

        x = clamp01(
            float(point[0])
            / float(width)
        )

        y = clamp01(
            float(point[1])
            / float(height)
        )

        normalized.extend(
            [x, y]
        )

    # Need at least 3 xy points = 6 values.
    if len(normalized) < 6:
        return None

    return normalized


def rectangle_to_polygon(points):
    """
    Labelme rectangle normally stores two diagonal points.
    Convert:
        (x1,y1), (x2,y2)
    into four polygon corners.
    """

    if len(points) != 2:
        return None

    x1, y1 = points[0]
    x2, y2 = points[1]

    xmin = min(x1, x2)
    xmax = max(x1, x2)
    ymin = min(y1, y2)
    ymax = max(y1, y2)

    return [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ]


def shape_to_polygon(shape):
    shape_type = str(
        shape.get(
            "shape_type",
            "polygon",
        )
    )

    points = shape.get(
        "points",
        []
    )

    if shape_type == "polygon":
        return points

    if shape_type == "rectangle":
        return rectangle_to_polygon(
            points
        )

    return None


def convert_one_annotation(
    json_path: Path,
    class_name: str,
):
    """
    Returns:
        image_path,
        list_of_yolo_lines

    Only shapes whose label == class_name are converted.
    """

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        annotation = json.load(f)

    image_path = find_image_for_json(
        json_path,
        annotation,
    )

    if image_path is None:
        raise FileNotFoundError(
            f"No source image found for: {json_path}"
        )

    width, height = get_image_size(
        image_path,
        annotation,
    )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid image size for {image_path}: "
            f"{width}x{height}"
        )

    yolo_lines = []

    for shape in annotation.get(
        "shapes",
        []
    ):
        label = str(
            shape.get(
                "label",
                "",
            )
        ).strip()

        # Single class project:
        # only export earbud_case.
        if label != class_name:
            continue

        polygon = shape_to_polygon(
            shape
        )

        if polygon is None:
            print(
                f"[SKIP SHAPE] Unsupported shape_type "
                f"in {json_path.name}: "
                f"{shape.get('shape_type')}"
            )
            continue

        normalized = normalize_polygon(
            polygon,
            width,
            height,
        )

        if normalized is None:
            print(
                f"[SKIP SHAPE] Invalid polygon "
                f"in {json_path.name}"
            )
            continue

        # Single target class -> class id 0.
        tokens = ["0"]

        tokens.extend(
            f"{value:.6f}"
            for value in normalized
        )

        yolo_lines.append(
            " ".join(tokens)
        )

    return image_path, yolo_lines


def reset_output_dirs(
    output_dir: Path,
    clean_output: bool,
):
    target_dirs = [
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
    ]

    if clean_output:
        for directory in target_dirs:
            if directory.exists():
                shutil.rmtree(
                    directory
                )

    for directory in target_dirs:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def write_data_yaml(
    output_dir: Path,
    class_name: str,
):
    yaml_path = (
        output_dir
        / "data.yaml"
    )

    content = (
        f"path: {output_dir.resolve()}\n"
        f"\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: {class_name}\n"
    )

    yaml_path.write_text(
        content,
        encoding="utf-8",
    )

    return yaml_path


def copy_sample(
    image_path: Path,
    yolo_lines,
    split: str,
    output_dir: Path,
):
    """
    Copy image and create a matching .txt label.
    """

    image_dst_dir = (
        output_dir
        / "images"
        / split
    )

    label_dst_dir = (
        output_dir
        / "labels"
        / split
    )

    image_dst = (
        image_dst_dir
        / image_path.name
    )

    label_dst = (
        label_dst_dir
        / f"{image_path.stem}.txt"
    )

    shutil.copy2(
        image_path,
        image_dst,
    )

    label_text = ""

    if yolo_lines:
        label_text = (
            "\n".join(
                yolo_lines
            )
            + "\n"
        )

    label_dst.write_text(
        label_text,
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    input_dir = Path(
        args.input
    ).expanduser().resolve()

    output_dir = Path(
        args.output
    ).expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(
            f"[ERROR] Input directory does not exist: "
            f"{input_dir}"
        )

    if not (
        0.0
        <= args.val_ratio
        < 1.0
    ):
        raise SystemExit(
            "[ERROR] --val-ratio must satisfy "
            "0 <= val-ratio < 1"
        )

    reset_output_dirs(
        output_dir,
        args.clean_output,
    )

    json_files = sorted(
        input_dir.glob(
            "*.json"
        )
    )

    if not json_files:
        raise SystemExit(
            f"[ERROR] No Labelme JSON files found in: "
            f"{input_dir}"
        )

    samples = []
    failed = []

    print()
    print(
        "============================================"
    )
    print(
        "Labelme -> YOLO Segmentation"
    )
    print(
        "============================================"
    )
    print(
        f"Input       : {input_dir}"
    )
    print(
        f"Output      : {output_dir}"
    )
    print(
        f"Class       : {args.class_name}"
    )
    print(
        f"JSON files  : {len(json_files)}"
    )
    print(
        f"Val ratio   : {args.val_ratio:.2f}"
    )
    print(
        "============================================"
    )
    print()

    for json_path in json_files:
        try:
            (
                image_path,
                yolo_lines,
            ) = convert_one_annotation(
                json_path,
                args.class_name,
            )

            if not yolo_lines:
                if not args.include_negative:
                    print(
                        f"[SKIP] No '{args.class_name}' "
                        f"polygon: {json_path.name}"
                    )
                    continue

            samples.append(
                (
                    image_path,
                    yolo_lines,
                )
            )

            print(
                f"[OK] {json_path.name} "
                f"-> {len(yolo_lines)} object(s)"
            )

        except Exception as exc:
            failed.append(
                (
                    json_path.name,
                    str(exc),
                )
            )

            print(
                f"[FAILED] {json_path.name}: "
                f"{exc}"
            )

    if not samples:
        raise SystemExit(
            "[ERROR] No valid samples were converted."
        )

    # --------------------------------------------------------
    # Deterministic train / val split
    # --------------------------------------------------------

    rng = random.Random(
        args.seed
    )

    rng.shuffle(
        samples
    )

    total = len(
        samples
    )

    if (
        args.val_ratio > 0.0
        and total > 1
    ):
        val_count = int(
            round(
                total
                * args.val_ratio
            )
        )

        # Keep both train and val non-empty.
        val_count = max(
            1,
            val_count,
        )

        val_count = min(
            val_count,
            total - 1,
        )

    else:
        val_count = 0

    val_samples = (
        samples[:val_count]
    )

    train_samples = (
        samples[val_count:]
    )

    for (
        image_path,
        yolo_lines,
    ) in train_samples:
        copy_sample(
            image_path,
            yolo_lines,
            "train",
            output_dir,
        )

    for (
        image_path,
        yolo_lines,
    ) in val_samples:
        copy_sample(
            image_path,
            yolo_lines,
            "val",
            output_dir,
        )

    yaml_path = write_data_yaml(
        output_dir,
        args.class_name,
    )

    print()
    print(
        "============================================"
    )
    print(
        "Conversion complete"
    )
    print(
        "============================================"
    )
    print(
        f"Valid samples : {total}"
    )
    print(
        f"Train         : {len(train_samples)}"
    )
    print(
        f"Val           : {len(val_samples)}"
    )
    print(
        f"Failed JSON   : {len(failed)}"
    )
    print(
        f"data.yaml     : {yaml_path}"
    )
    print(
        "============================================"
    )

    if failed:
        print()
        print(
            "Failed files:"
        )

        for name, error in failed:
            print(
                f"  - {name}: {error}"
            )

    print()
    print(
        "YOLO label example:"
    )

    first_lines = (
        samples[0][1]
    )

    if first_lines:
        print(
            first_lines[0][
                :180
            ]
        )

    print()


if __name__ == "__main__":
    main()
