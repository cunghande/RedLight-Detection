"""
Kịch Bản Huấn Luyện & Đánh Giá Mô Hình AI Phân Đoạn Vạch Dừng (Stopline Segmentation)
Chuẩn Nghiên Cứu Khoa Học (Paper-Grade Training & Evaluation Pipeline):
1. Huấn luyện mạng nơ-ron phân đoạn ảnh YOLOv8-seg (SOTA Instance Segmentation).
2. Tối ưu hóa siêu tham số: Cosine LR, Warmup, Augmentation chống lóa/trời mưa.
3. Đánh giá đa chiều trên tập Hold-out Test Set độc lập (mAP50, mAP50-95, Precision, Recall).
4. Tự động xuất biểu đồ huấn luyện, Confusion Matrix và các dự đoán trực quan.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("TrainStopline")


def run_training_pipeline(
    data_yaml: str = "dataset_stopline/data.yaml",
    base_model: str = "yolov8n-seg.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 8,
    project_dir: str = "runs/stopline_research",
    experiment_name: str = "yolov8_stopline_seg"
) -> Dict:
    data_path = Path(data_yaml)
    if not data_path.exists():
        logger.error(f"Không tìm thấy file cấu hình dataset: {data_yaml}")
        return {}

    device = "0" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info(f" KHỞI CHẠY HUẤN LUYỆN MODEL PHÂN ĐOẠN VẠCH DỪNG: {base_model}")
    logger.info(f" Thiết bị huấn luyện: {device.upper()} | Số Epochs: {epochs} | Batch: {batch} | Imgsz: {imgsz}")
    logger.info("=" * 70)

    # 1. Khởi tạo mô hình
    model = YOLO(base_model)

    start_time = time.time()

    # 2. Huấn luyện mô hình với Augmentation học thuật
    train_results = model.train(
        data=str(data_path.resolve()).replace("\\", "/"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_dir,
        name=experiment_name,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
        patience=15,
        # Siêu tham số tối ưu
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=2.0,
        # Augmentation cho camera giao thông
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.05,
        scale=0.1,
        fliplr=0.5,
        mosaic=0.5
    )

    train_duration = time.time() - start_time
    save_dir = Path(train_results.save_dir) if hasattr(train_results, "save_dir") else Path(project_dir) / experiment_name
    best_weights = save_dir / "weights" / "best.pt"

    logger.info(f"\n[HUẤN LUYỆN HOÀN TẤT] Thời gian: {train_duration:.1f}s")
    logger.info(f"Trọng số tối ưu nhất được lưu tại: {best_weights}")

    # 3. Đánh giá kiểm định độc lập trên tập Hold-out Test Set
    logger.info("\n" + "=" * 70)
    logger.info(" TIẾN HÀNH ĐÁNH GIÁ BENCHMARK TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST SET)")
    logger.info("=" * 70)

    best_model = YOLO(str(best_weights))
    test_metrics = best_model.val(
        data=str(data_path.resolve()).replace("\\", "/"),
        split="test",
        imgsz=imgsz,
        device=device,
        plots=True
    )

    # Trích xuất các độ đo học thuật
    map50_mask = float(test_metrics.seg.map50) if hasattr(test_metrics, "seg") else 0.0
    map50_95_mask = float(test_metrics.seg.map) if hasattr(test_metrics, "seg") else 0.0
    prec_mask = float(test_metrics.seg.mp) if hasattr(test_metrics, "seg") else 0.0
    recall_mask = float(test_metrics.seg.mr) if hasattr(test_metrics, "seg") else 0.0
    f1_mask = float(2 * (prec_mask * recall_mask) / max(1e-6, prec_mask + recall_mask))

    map50_box = float(test_metrics.box.map50)
    map50_95_box = float(test_metrics.box.map)
    prec_box = float(test_metrics.box.mp)
    recall_box = float(test_metrics.box.mr)

    benchmark_summary = {
        "model_architecture": base_model,
        "epochs": epochs,
        "training_time_seconds": round(train_duration, 2),
        "best_weights_path": str(best_weights),
        "test_metrics_segmentation": {
            "mAP50_mask": round(map50_mask, 4),
            "mAP50_95_mask": round(map50_95_mask, 4),
            "precision_mask": round(prec_mask, 4),
            "recall_mask": round(recall_mask, 4),
            "f1_score_mask": round(f1_mask, 4)
        },
        "test_metrics_bounding_box": {
            "mAP50_box": round(map50_box, 4),
            "mAP50_95_box": round(map50_95_box, 4),
            "precision_box": round(prec_box, 4),
            "recall_box": round(recall_box, 4)
        }
    }

    # Lưu kết quả định lượng JSON
    metrics_json = save_dir / "test_benchmark_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(" BẢNG KẾT QUẢ ĐÁNH GIÁ CHUẨN NGHIÊN CỨU (TEST SET BENCHMARK)")
    print("=" * 70)
    print(f"| Chỉ số đo lường (Metrics)     | Phân đoạn (Segmentation) | Hộp bao (Bounding Box) |")
    print(f"| :---------------------------- | :----------------------: | :--------------------: |")
    print(f"| mAP @ 0.50 (Độ chính xác cao) | **{map50_mask * 100:.2f}%**                | **{map50_box * 100:.2f}%**               |")
    print(f"| mAP @ 0.50:0.95 (Toàn diện)   | **{map50_95_mask * 100:.2f}%**                | **{map50_95_box * 100:.2f}%**               |")
    print(f"| Precision (Độ chuẩn xác)      | **{prec_mask * 100:.2f}%**                | **{prec_box * 100:.2f}%**               |")
    print(f"| Recall (Độ bao phủ)           | **{recall_mask * 100:.2f}%**                | **{recall_box * 100:.2f}%**               |")
    print(f"| F1-Score                      | **{f1_mask:.4f}**                  | —                      |")
    print("=" * 70)

    # 4. Dự đoán thử nghiệm trực quan trên các ảnh Test
    test_vis_dir = save_dir / "test_visualizations"
    test_vis_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir = Path("dataset_stopline/images/test")
    test_images = list(test_img_dir.glob("*.jpg"))[:5]

    for t_img in test_images:
        res = best_model.predict(str(t_img), conf=0.25, save=False)[0]
        vis_plot = res.plot()
        cv2.imwrite(str(test_vis_dir / t_img.name), vis_plot)

    logger.info(f"Đã lưu các ảnh dự đoán mẫu trên Test Set tại: {test_vis_dir}")

    # Copy trọng số best.pt vào thư mục chính để hệ thống dùng ngay
    shutil.copy(best_weights, "best_stopline_seg.pt")
    logger.info("Đã sao chép model tối ưu ra: best_stopline_seg.pt để tích hợp vào hệ thống.")

    return benchmark_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện & Đánh giá Model AI Phân Đoạn Vạch Dừng chuẩn nghiên cứu")
    parser.add_argument("--data", type=str, default="dataset_stopline/data.yaml", help="File cấu hình dataset")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Kiến trúc mạng (yolov8n-seg.pt, yolov8s-seg.pt)")
    parser.add_argument("--epochs", type=int, default=15, help="Số epochs huấn luyện")
    parser.add_argument("--batch", type=int, default=8, help="Kích thước batch")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    args = parser.parse_args()

    run_training_pipeline(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz
    )
