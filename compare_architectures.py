import time
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config
from model_factory import get_detector

MODELS_CONFIG = [
    {
        "id": "faster_rcnn",
        "name": "Faster R-CNN (ResNet-50)",
        "type": "Two-Stage (Baseline)",
        "description": "Mo hinh Two-stage kinh dien, do chinh xac cao nhung latency lon"
    },
    {
        "id": "ssd_mobilenet",
        "name": "SSD-MobileNet",
        "type": "One-Stage (Mobile/Tầm trung)",
        "description": "Mo hinh nhe toi uu cho thiet bi nhung/di dong, danh doi do chinh xac"
    },
    {
        "id": "yolov8s",
        "name": "YOLOv8s (Model Chinh)",
        "type": "Anchor-Free (SOTA Real-time)",
        "description": "Mo hinh chu luc ket hop toc do thoi gian thuc va do chinh xac cao"
    }
]

def run_multi_architecture_benchmark(video_path="red1.mp4", max_frames=30):
    print("=" * 80)
    print("BENCHMARK SO SANH 3 TRUONG PHAI KIEN TRUC OBJECT DETECTION")
    print(" 1. Faster R-CNN (ResNet-50)  | Two-Stage Baseline")
    print(" 2. SSD-MobileNet             | One-Stage Mobile/Tầm trung")
    print(" 3. YOLOv8s                   | SOTA Real-Time (Model Chính)")
    print(f"Video thu nghiem: {video_path} ({max_frames} frames)")
    print("=" * 80)

    results = []

    for m_cfg in MODELS_CONFIG:
        m_id = m_cfg["id"]
        m_name = m_cfg["name"]
        print(f"\n[+] Dang khoi tao va danh gia: {m_name}...")

        detector = get_detector(m_id)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[LOI] Khong the mo video: {video_path}")
            return

        frame_count = 0
        latencies = []
        total_detections = 0
        conf_scores = []

        # Warm up 1 frame
        ret, frame = cap.read()
        if ret:
            _ = detector.detect(frame, conf_thresh=0.35)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            boxes, scores, class_names = detector.detect(frame, conf_thresh=0.35)
            dt_ms = (time.perf_counter() - t0) * 1000
            latencies.append(dt_ms)

            if len(boxes) > 0:
                total_detections += len(boxes)
                conf_scores.extend(scores.tolist())

            frame_count += 1

        cap.release()

        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.0
        dets_per_frame = total_detections / frame_count if frame_count > 0 else 0.0

        item = {
            "model_name": m_name,
            "architecture_type": m_cfg["type"],
            "description": m_cfg["description"],
            "avg_latency_ms": round(avg_latency, 2),
            "avg_fps": round(avg_fps, 1),
            "total_detections": total_detections,
            "detections_per_frame": round(dets_per_frame, 2),
            "avg_confidence": round(avg_conf, 3)
        }
        results.append(item)
        print(f"    -> Latency: {avg_latency:.1f}ms | FPS: {avg_fps:.1f} | Detections: {total_detections} | Avg Conf: {avg_conf:.3f}")

    # Bang ket qua
    print("\n" + "=" * 90)
    print(f"{'Mô hình':<28} | {'Trường phái':<24} | {'Latency (ms)':<13} | {'FPS':<8} | {'Avg Conf'}")
    print("-" * 90)
    for r in results:
        print(f"{r['model_name']:<28} | {r['architecture_type']:<24} | {r['avg_latency_ms']:<13} | {r['avg_fps']:<8} | {r['avg_confidence']}")
    print("=" * 90)

    # Luu JSON
    with open("architecture_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    plot_architecture_comparison(results)
    print("\n[THANH CONG] Da xuat bao cao 'architecture_comparison.json' va bieu do 'architecture_chart.png'")

def plot_architecture_comparison(results):
    names = ["Faster R-CNN", "SSD-MobileNet", "YOLOv8s"]
    fps_vals = [r["avg_fps"] for r in results]
    conf_vals = [r["avg_confidence"] * 100 for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(names))
    width = 0.45

    b1 = ax1.bar(x, fps_vals, width, label="FPS", color="#2ca02c")
    ax1.set_ylabel("Frames Per Second (FPS)")
    ax1.set_title("So sanh Toc do xu ly (FPS)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontweight="bold")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in b1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    b2 = ax2.bar(x, conf_vals, width, label="Confidence (%)", color="#1f77b4")
    ax2.set_ylabel("Do tin cay (%)")
    ax2.set_title("So sanh Do tin cay trung binh (Avg Confidence)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontweight="bold")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in b2:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("SO SANH 3 TRUONG PHAI: FASTER R-CNN vs SSD-MOBILENET vs YOLOV8S", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("architecture_chart.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sanh 3 kien truc Object Detection")
    parser.add_argument("--source", type=str, default="red1.mp4", help="Video thu nghiem")
    parser.add_argument("--frames", type=int, default=30, help="So frame toi da")
    args = parser.parse_args()

    run_multi_architecture_benchmark(video_path=args.source, max_frames=args.frames)
