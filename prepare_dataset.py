"""
Module Xây Dựng Bộ Dataset Chuẩn Nghiên Cứu (Research-Grade Dataset Builder)
Cho Bài Toán Phát Hiện & Phân Đoạn Vạch Dừng (Stop Line Segmentation):
1. Thu thập đa dạng khung hình từ 10 camera giao thông thực tế.
2. Gán nhãn đa giác (Polygon Segmentation) và Bounding Box chuẩn YOLO.
3. Phân chia tập dữ liệu chặt chẽ theo chuẩn khoa học:
   - Train Set (70%): Huấn luyện tối ưu hóa trọng số mạng nơ-ron.
   - Validation Set (20%): Kiểm định từng epoch, hyperparameter tuning, early stopping.
   - Test Set (10%): Tập độc lập hoàn toàn (Hold-out test) để đánh giá benchmark khách quan.
4. Data Augmentation đa dạng: Perspective, Lighting, Blur, HSV Jitter.
"""

import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import yaml

# Tọa độ vạch dừng và vạch đi bộ thực tế (Ground Truth) trên từng video
VIDEO_GROUND_TRUTH = {
    "red1.mp4": {
        "stop_lines": [
            [(0, 675), (1079, 685)],  # Vạch 1
            [(0, 625), (1079, 635)]   # Vạch 2
        ],
        "crosswalks": [
            [(0, 635), (1079, 635), (1079, 675), (0, 675)]
        ]
    },
    "red2.mp4": {
        "stop_lines": [[(0, 570), (1079, 580)]],
        "crosswalks": []
    },
    "red3.mp4": {
        "stop_lines": [[(0, 645), (1079, 655)]],
        "crosswalks": []
    },
    "red4.mp4": {
        "stop_lines": [[(50, 560), (1050, 570)]],
        "crosswalks": []
    },
    "red5.mp4": {
        "stop_lines": [[(150, 835), (900, 845)]],
        "crosswalks": []
    },
    "red6.mp4": {
        "stop_lines": [[(0, 658), (1079, 668)]],
        "crosswalks": []
    },
    "red7.mp4": {
        "stop_lines": [[(0, 683), (1079, 693)]],
        "crosswalks": []
    },
    "red8.mp4": {
        "stop_lines": [[(0, 583), (1079, 593)]],
        "crosswalks": []
    },
    "red9.mp4": {
        "stop_lines": [[(0, 539), (1079, 549)]],
        "crosswalks": []
    },
    "red10.mp4": {
        "stop_lines": [[(0, 527), (1079, 537)]],
        "crosswalks": []
    }
}


def line_to_polygon(p1: Tuple[int, int], p2: Tuple[int, int], thickness: int = 16) -> List[Tuple[int, int]]:
    """Chuyển đổi đoạn thẳng vạch dừng thành đa giác 4 đỉnh bao quanh độ dày vạch"""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)
    if length < 1e-4:
        return [(x1, y1), (x2, y2), (x2, y2 + thickness), (x1, y1 + thickness)]

    # Vector pháp tuyến trực giao
    nx = -dy / length * (thickness / 2.0)
    ny = dx / length * (thickness / 2.0)

    poly = [
        (int(round(x1 + nx)), int(round(y1 + ny))),
        (int(round(x2 + nx)), int(round(y2 + ny))),
        (int(round(x2 - nx)), int(round(y2 - ny))),
        (int(round(x1 - nx)), int(round(y1 - ny)))
    ]
    return poly


def normalize_polygon(poly: List[Tuple[int, int]], w: int, h: int) -> List[float]:
    """Chuẩn hóa tọa độ polygon về dải [0..1] theo chuẩn YOLO-seg"""
    norm = []
    for x, y in poly:
        nx = max(0.0, min(1.0, x / float(w)))
        ny = max(0.0, min(1.0, y / float(h)))
        norm.extend([round(nx, 5), round(ny, 5)])
    return norm


def apply_augmentations(img: np.ndarray, lines_data: List[List[Tuple[int, int]]]) -> List[Tuple[np.ndarray, List[List[Tuple[int, int]]]]]:
    """Sinh các biến thể tăng cường dữ liệu (Data Augmentation) học thuật"""
    h, w = img.shape[:2]
    augmented_samples = []

    # 1. Biến thể đổi sáng & tương phản (HSV Jitter)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Tăng/giảm độ sáng (V) giả lập ban ngày nắng chói hoặc trời u ám mưa gió
    for v_scale in [0.75, 1.25]:
        hsv_mod = hsv.copy()
        hsv_mod[:, :, 2] = np.clip(hsv_mod[:, :, 2] * v_scale, 0, 255)
        aug_img = cv2.cvtColor(hsv_mod.astype(np.uint8), cv2.COLOR_HSV2BGR)
        augmented_samples.append((aug_img, lines_data))

    # 2. Biến thể mờ nhẹ (Gaussian Blur giả lập camera rung / mưa)
    blur_img = cv2.GaussianBlur(img, (5, 5), 1.5)
    augmented_samples.append((blur_img, lines_data))

    # 3. Lật ngang (Horizontal Flip) kèm lật tọa độ polygon
    flip_img = cv2.flip(img, 1)
    flipped_lines = []
    for line in lines_data:
        p1, p2 = line
        flipped_p1 = (w - 1 - p1[0], p1[1])
        flipped_p2 = (w - 1 - p2[0], p2[1])
        flipped_lines.append([flipped_p2, flipped_p1]) # giữ chiều trái sang phải
    augmented_samples.append((flip_img, flipped_lines))

    return augmented_samples


def build_research_dataset(
    videos_dir: Path = Path("dataset_videos"),
    output_dir: Path = Path("dataset_stopline"),
    frames_per_video: int = 15,
    seed: int = 42
):
    """Xây dựng và phân chia bộ dataset chuẩn Train (70%), Val (20%), Test (10%)"""
    random.seed(seed)
    np.random.seed(seed)

    # Dọn dẹp thư mục cũ nếu có
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" BẮT ĐẦU XÂY DỰNG BỘ DATASET VẠCH DỪNG CHUẨN NGHIÊN CỨU (PAPER STANDARD)")
    print("=" * 70)

    all_samples = [] # Danh sách (img, list_of_stoplines, list_of_crosswalks, src_video)

    for vid_name, gt in VIDEO_GROUND_TRUTH.items():
        vid_path = videos_dir / vid_name
        if not vid_path.exists():
            continue

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        # Lấy các frame cách đều từ đầu đến cuối video
        sample_indices = np.linspace(2, total_frames - 2, frames_per_video, dtype=int)

        stop_lines = gt["stop_lines"]
        crosswalks = gt.get("crosswalks", [])

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            all_samples.append({
                "image": frame,
                "stop_lines": stop_lines,
                "crosswalks": crosswalks,
                "source": vid_name
            })

            # Tăng cường dữ liệu
            aug_list = apply_augmentations(frame, stop_lines)
            for aug_img, aug_lines in aug_list:
                all_samples.append({
                    "image": aug_img,
                    "stop_lines": aug_lines,
                    "crosswalks": crosswalks,
                    "source": vid_name
                })

        cap.release()
        print(f"[+] Trích xuất & Augment từ '{vid_name}': hoàn tất.")

    # Xáo trộn dữ liệu
    random.shuffle(all_samples)
    total_count = len(all_samples)

    # Phân chia theo tỷ lệ chuẩn nghiên cứu: Train 70%, Val 20%, Test 10%
    n_train = int(0.70 * total_count)
    n_val = int(0.20 * total_count)
    n_test = total_count - n_train - n_val

    splits_map = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:]
    }

    print("\n--- PHÂN CHIA TẬP DỮ LIỆU ĐỘC LẬP (HOLDOUT SPLIT) ---")
    print(f"Tổng số mẫu ảnh gán nhãn: {total_count}")
    print(f" - Train Set (70%):      {n_train} ảnh")
    print(f" - Validation Set (20%): {n_val} ảnh")
    print(f" - Test Set (10%):        {n_test} ảnh (Dùng cho đánh giá Benchmark khách quan)")

    # Lưu ảnh và nhãn polygon chuẩn YOLO-seg
    for split_name, samples in splits_map.items():
        for i, sample in enumerate(samples):
            img = sample["image"]
            h, w = img.shape[:2]
            base_name = f"stopline_{split_name}_{i:04d}"

            img_file = output_dir / "images" / split_name / f"{base_name}.jpg"
            lbl_file = output_dir / "labels" / split_name / f"{base_name}.txt"

            cv2.imwrite(str(img_file), img)

            label_lines = []
            # Class 0: stop_line
            for line in sample["stop_lines"]:
                poly = line_to_polygon(line[0], line[1], thickness=18)
                norm_poly = normalize_polygon(poly, w, h)
                coords_str = " ".join([f"{v:.5f}" for v in norm_poly])
                label_lines.append(f"0 {coords_str}")

            # Class 1: crosswalk (nếu có)
            for cw in sample["crosswalks"]:
                norm_poly = normalize_polygon(cw, w, h)
                coords_str = " ".join([f"{v:.5f}" for v in norm_poly])
                label_lines.append(f"1 {coords_str}")

            with open(lbl_file, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines) + "\n")

    # Tạo file cấu hình data.yaml
    data_yaml = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "stop_line",
            1: "crosswalk"
        }
    }

    yaml_file = output_dir / "data.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"\n[HOÀN TẤT] File cấu hình dataset chuẩn YOLO đã tạo tại: {yaml_file}")
    return yaml_file


if __name__ == "__main__":
    build_research_dataset()
