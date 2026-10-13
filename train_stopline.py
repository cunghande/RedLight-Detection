"""
Script Huấn Luyện Model AI Phân Đoạn Vạch Dừng (YOLOv8-seg / YOLO11-seg).
- Hỗ trợ tạo cấu trúc dataset chuẩn YOLO-seg (stop_line, zebra_crossing).
- Thiết lập augmentation chuyên biệt cho camera giao thông (đổi sáng, mờ, perspective, xoay nhẹ).
- Hướng dẫn chi tiết gán nhãn dữ liệu bằng CVAT / Roboflow.
"""

import argparse
import logging
from pathlib import Path
import random
import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("TrainStopLine")


def create_sample_dataset_structure(dataset_dir: Path):
    """Tạo cấu trúc thư mục dataset chuẩn YOLO-seg"""
    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    data_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "stop_line",
            1: "zebra_crossing"
        }
    }
    with open(dataset_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f)

    logger.info(f"Đã tạo cấu trúc dataset và file 'data.yaml' tại '{dataset_dir}'")


def generate_synthetic_bootstrap_data(video_files: list, dataset_dir: Path, num_samples: int = 40):
    """
    Trích xuất frame từ các video thực tế và sinh nhãn bootstrap giả lập
    để kiểm thử pipeline huấn luyện trước khi có nhãn thủ công từ CVAT/Roboflow.
    """
    create_sample_dataset_structure(dataset_dir)
    logger.info(f"Đang sinh {num_samples} mẫu dữ liệu bootstrap từ các video thực tế...")

    count = 0
    for vid_path in video_files:
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(5, min(total - 1, 150), 5, dtype=int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            split = "val" if count % 5 == 0 else "train"
            img_name = f"frame_{count:04d}.jpg"
            lbl_name = f"frame_{count:04d}.txt"

            img_path = dataset_dir / "images" / split / img_name
            lbl_path = dataset_dir / "labels" / split / lbl_name

            cv2.imwrite(str(img_path), frame)

            # Tọa độ polygon chuẩn hóa [0..1] giả lập vạch dừng ngang mặt đường (y: 0.55 -> 0.58)
            y_norm1 = 0.55
            y_norm2 = 0.58
            poly = [
                0.15, y_norm1,
                0.85, y_norm1,
                0.85, y_norm2,
                0.15, y_norm2
            ]
            poly_str = " ".join([f"{v:.4f}" for v in poly])
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write(f"0 {poly_str}\n") # Class 0: stop_line

            count += 1
            if count >= num_samples:
                break
        cap.release()
        if count >= num_samples:
            break

    logger.info(f"Đã tạo thành công {count} mẫu ảnh + polygon annotation cho YOLO-seg!")


def train_yolo_seg(
    data_yaml_path: str = "dataset_stopline/data.yaml",
    base_model: str = "yolov8s-seg.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    output_dir: str = "runs/stopline_seg"
):
    """Huấn luyện model YOLOv8-seg / YOLO11-seg với data augmentation chuyên sâu"""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Chưa cài đặt ultralytics. Hãy chạy: pip install ultralytics")
        return

    logger.info(f"Bắt đầu huấn luyện YOLO-seg từ model nền '{base_model}'...")
    model = YOLO(base_model)

    # Cấu hình augmentation chuyên biệt cho giao thông
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=output_dir,
        name="stopline_model",
        # Augmentation
        hsv_h=0.015,       # thay đổi sắc độ nhẹ
        hsv_s=0.6,         # tăng giảm độ bão hòa (giả lập trời mưa/nắng)
        hsv_v=0.4,         # tăng giảm độ sáng (giả lập ngày/đêm/bóng râm)
        degrees=6.0,       # xoay nhẹ camera (+-6 độ)
        perspective=0.0006,# biến đổi phối cảnh đường dốc
        shear=2.0,         # biến dạng trượt
        flipud=0.0,        # KHÔNG lật ngược trên-dưới (mặt đường luôn ở dưới)
        fliplr=0.5,        # lật trái-phải
        mosaic=0.5,        # mosaic ghép ảnh
        verbose=True
    )
    logger.info("Huấn luyện hoàn tất! Trọng số tốt nhất lưu tại: runs/stopline_seg/stopline_model/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện model YOLO-seg nhận diện vạch dừng")
    parser.add_argument("--generate-data", action="store_true", help="Sinh dataset mẫu từ video thực tế để kiểm thử pipeline")
    parser.add_argument("--data", type=str, default="dataset_stopline/data.yaml", help="Đường dẫn file data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s-seg.pt", help="Model nền (yolov8s-seg.pt, yolo11s-seg.pt)")
    parser.add_argument("--epochs", type=int, default=20, help="Số epochs huấn luyện")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    args = parser.parse_args()

    ds_dir = Path("dataset_stopline")
    if args.generate_data:
        video_paths = list(Path("dataset_videos").glob("*.mp4"))
        generate_synthetic_bootstrap_data(video_paths, ds_dir)

    train_yolo_seg(data_yaml_path=args.data, base_model=args.model, epochs=args.epochs, imgsz=args.imgsz)
