import argparse
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Optional

import cv2
import numpy as np
import yaml

from src.camera_stabilizer import CameraStabilizer
from src.report_generator import ReportGenerator
from src.stopline_detector import StopLineDetector
from src.traffic_light_detector import TrafficLightDetector
from src.violation_engine import ViolationEngine

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("TrafficSystem")


def load_yaml_config(config_path: str = "config.yaml") -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def draw_hud(
    frame: np.ndarray,
    traffic_state: str,
    total_violations: int,
    fps: float,
    stop_lines_info: str,
    camera_id: str
):
    """Vẽ bảng điều khiển HUD thông tin trực quan ở góc trên"""
    h, w = frame.shape[:2]

    # Panel nền tối kính mờ ở góc trên
    panel_h = 55
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 1. Tiêu đề
    cv2.putText(frame, "AI RED LIGHT SYSTEM", (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)

    # 2. Camera ID & Vạch dừng
    cv2.putText(frame, f"CAM: {camera_id} | {stop_lines_info}", (270, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # 3. Đèn tín hiệu giao thông
    state_colors = {
        "RED": (0, 0, 255),
        "YELLOW": (0, 255, 255),
        "GREEN": (0, 255, 0)
    }
    cur_color = state_colors.get(traffic_state, (0, 255, 0))
    cv2.circle(frame, (w - 280, 27), 10, cur_color, -1)
    cv2.putText(frame, f"DEN: {traffic_state}", (w - 260, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cur_color, 2)

    # 4. Tổng vi phạm
    cv2.putText(frame, f"VI PHAM: {total_violations}", (w - 140, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255) if total_violations > 0 else (255, 255, 255), 2)


def run_pipeline(
    video_source: str,
    camera_id: str = "cam_01",
    redetect: bool = False,
    max_lines: Optional[int] = None,
    output_video: Optional[str] = "output_violation.mp4",
    stabilize: bool = False,
    headless: bool = False,
    debug: bool = False
):
    cfg = load_yaml_config("config.yaml")

    # Mở nguồn video
    cap = cv2.VideoCapture(int(video_source) if str(video_source).isdigit() else str(video_source))
    if not cap.isOpened():
        logger.error(f"Không thể mở nguồn video: {video_source}")
        return

    # 1. KHỞI TẠO BỘ PHÁT HIỆN VẠCH DỪNG (CHẠY 1 LẦN HOẶC ĐỌC CACHE)
    line_detector = StopLineDetector(config_path="config.yaml")
    stop_lines = line_detector.detect(
        video_source,
        camera_id=camera_id,
        redetect=redetect,
        max_lines=max_lines,
        debug=debug
    )

    if not stop_lines:
        logger.error(f"[LỖI] Không thể phát hiện hoặc đọc vạch dừng cho video: {video_source}")
        return

    # 2. KHỞI TẠO BỘ CHỐNG RUNG CAMERA
    stabilizer = CameraStabilizer(
        enabled=stabilize or cfg.get("camera", {}).get("stabilization_enabled", False),
        method=cfg.get("stabilization", {}).get("method", "ORB"),
        max_features=cfg.get("stabilization", {}).get("max_features", 1000)
    )

    ret, first_frame = cap.read()
    if ret:
        stabilizer.set_reference_frame(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 3. KHỞI TẠO MODEL YOLO VÀ TRACKER (BYTETRACK)
    from ultralytics import YOLO
    model_weights = cfg.get("models", {}).get("vehicle_detector", {}).get("weights", "weights/yolo11x.pt")
    if not Path(model_weights).is_file():
        raise FileNotFoundError(f"Không tìm thấy trọng số xe: {model_weights}")
    logger.info(f"Đang tải model YOLO ({model_weights}) kèm ByteTrack...")
    yolo_model = YOLO(model_weights)

    # 4. KHỞI TẠO BỘ NHẬN DIỆN ĐÈN TÍN HIỆU
    light_cfg = cfg.get("models", {}).get("traffic_light", {})
    light_weights = light_cfg.get("weights", "weights/yolov8s.pt")
    if not Path(light_weights).is_file():
        raise FileNotFoundError(f"Không tìm thấy trọng số đèn: {light_weights}")
    light_model = YOLO(light_weights)
    if light_model.names.get(9) != "traffic light":
        raise ValueError("Model đèn cần là checkpoint COCO có class 9 = traffic light")
    tl_detector = TrafficLightDetector(yolo_model=light_model,
                                       conf_thresh=light_cfg.get("conf", 0.25),
                                       max_missing_frames=light_cfg.get("max_missing_frames", 10))

    # 5. KHỞI TẠO LOGIC VI PHẠM (VIOLATION ENGINE)
    output_dir = Path(cfg.get("general", {}).get("output_dir", "violations")) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    violation_engine = ViolationEngine(
        stop_lines=stop_lines,
        output_dir=output_dir,
        min_violation_frames=cfg.get("violation_logic", {}).get("min_violation_frames", 3),
        allow_right_turn=cfg.get("violation_logic", {}).get("allow_right_turn", False),
        right_turn_x_ratio=cfg.get("violation_logic", {}).get("right_turn_x_ratio", 0.75)
    )

    # Thiết lập VideoWriter nếu cần lưu video
    writer = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps_in, (w_in, h_in))
        logger.info(f"Đang thiết lập lưu video kết quả ra: {output_video}")

    logger.info("\n" + "=" * 55)
    logger.info(" HỆ THỐNG PHÁT HIỆN VƯỢT ĐÈN ĐỎ ĐANG HOẠT ĐỘNG")
    logger.info(" Phím tắt: [q]: Thoát | [t]: Đổi màu đèn | [space]: Tạm dừng")
    logger.info("=" * 55 + "\n")

    frame_idx = 0
    paused = False
    prev_time = time.time()
    fps = 0.0

    class_aliases = {"person": "Nguoi di bo", "bicycle": "Xe dap", "car": "O to",
                     "motorcycle": "Xe may", "bus": "Xe buyt", "truck": "Xe tai"}
    vehicle_cfg = cfg.get("models", {}).get("vehicle_detector", {})
    selected_names = vehicle_cfg.get("track_names", list(class_aliases))
    track_classes = [i for i, name in yolo_model.names.items() if name in selected_names]
    if not track_classes:
        raise ValueError(f"No selected classes in checkpoint: {yolo_model.names}")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                logger.info("Đã xử lý hết video.")
                break
            frame_idx += 1

            # Ổn định hóa vạch dừng nếu camera rung lắc
            current_stop_lines = stabilizer.stabilize_lines(stop_lines, frame)
            violation_engine.update_stop_lines(current_stop_lines)

            # Chạy YOLO Tracking với ByteTrack
            results = yolo_model.track(
                frame,
                persist=True,
                tracker=cfg.get("models", {}).get("vehicle_detector", {}).get("tracker", "bytetrack.yaml"),
                classes=track_classes,
                conf=cfg.get("models", {}).get("vehicle_detector", {}).get("conf", 0.35),
                verbose=False
            )

            # Thu thập tracks
            tracks = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes_data = results[0].boxes
                for i in range(len(boxes_data)):
                    box = boxes_data.xyxy[i].cpu().numpy().astype(int)
                    cls_id = int(boxes_data.cls[i].cpu().numpy())
                    conf = float(boxes_data.conf[i].cpu().numpy())
                    track_id = int(boxes_data.id[i].cpu().numpy()) if boxes_data.id is not None and boxes_data.id[i] is not None else None

                    if track_id is None:
                        continue
                    tracks.append({
                        "id": track_id,
                        "box": box,
                        "class_name": class_aliases.get(yolo_model.names[cls_id], yolo_model.names[cls_id]),
                        "conf": conf
                    })

            # Nhận diện trạng thái đèn giao thông
            traffic_state, _ = tl_detector.detect_state(frame)

            # Xử lý cắt vạch & vi phạm
            annotated_frame, frame_violations = violation_engine.process_tracks(
                frame=frame,
                tracks=tracks,
                traffic_state=traffic_state,
                frame_idx=frame_idx
            )

            # Tính toán FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, now - prev_time))
            prev_time = now

            # Vẽ HUD Dashboard
            line_desc = f"{len(current_stop_lines)} vach (CV)"
            draw_hud(
                frame=annotated_frame,
                traffic_state=traffic_state,
                total_violations=len(violation_engine.violated_ids),
                fps=fps,
                stop_lines_info=line_desc,
                camera_id=camera_id
            )

            if writer:
                writer.write(annotated_frame)

        if not headless:
            cv2.imshow("He Thong Phat Hien Vuot Den Do", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                new_state = tl_detector.toggle_state()
                logger.info(f"Đã chuyển trạng thái đèn thủ công sang: {new_state}")
            elif key == 32:
                paused = not paused
                logger.info(f"{'Tạm dừng' if paused else 'Tiếp tục'}")

    cap.release()
    if writer:
        writer.release()
    if not headless:
        cv2.destroyAllWindows()

    # Xuất các sự kiện cần người dùng kiểm tra.
    report_gen = ReportGenerator(output_dir=output_dir)
    report_gen.export_reports(violation_engine.violations_log)

    logger.info(f"\n[HOÀN TẤT] Số sự kiện nghi vượt đèn đỏ: {len(violation_engine.violated_ids)}")
    logger.info(f"Báo cáo JSON: {output_dir / 'events.json'}")
    logger.info(f"Báo cáo CSV:  {output_dir / 'events.csv'}")
    logger.info(f"Báo cáo Markdown: {output_dir / 'events.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hệ thống phát hiện phương tiện vượt đèn đỏ tự động định vị vạch dừng")
    parser.add_argument("--video", "--source", dest="video", type=str, default="dataset_videos/red1.mp4", help="Đường dẫn file video hoặc ID camera")
    parser.add_argument("--camera-id", type=str, default="cam_01", help="ID định danh camera (cam_01, cam_red1...)")
    parser.add_argument("--redetect", action="store_true", help="Bỏ qua cache và ép buộc quét lại vạch mới")
    parser.add_argument("--max-lines", type=int, default=None, help="Số lượng vạch dừng tối đa cần phát hiện (1 hoặc 2)")
    parser.add_argument("--output", type=str, default="output_violation.mp4", help="Đường dẫn lưu video kết quả")
    parser.add_argument("--stabilize", action="store_true", help="Bật chống rung lắc camera bằng Homography")
    parser.add_argument("--headless", action="store_true", help="Chạy chế độ ngầm không mở cửa sổ giao diện")
    parser.add_argument("--debug", action="store_true", help="Xuất ảnh debug chi tiết: dòng xe, Road ROI, cụm xe dừng, ứng viên kèm bảng điểm")
    args = parser.parse_args()

    run_pipeline(
        video_source=args.video,
        camera_id=args.camera_id,
        redetect=args.redetect,
        max_lines=args.max_lines,
        output_video=args.output,
        stabilize=args.stabilize,
        headless=args.headless,
        debug=args.debug
    )
