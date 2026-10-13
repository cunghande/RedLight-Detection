import argparse
import sys
import time
from pathlib import Path
import cv2

import config
from traffic_light import TrafficLightDetector
from red_light_detector import RedLightViolationDetector

def draw_hud(frame, traffic_state, total_violations, fps, recent_violations):
    """Ve giao dien thong tin (HUD Dashboard) hien dai, truc quan len video"""
    h, w = frame.shape[:2]
    
    # 1. Thanh Top Bar thong so tren cung
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, 55), (w, 55), (60, 60, 60), 1)

    # Tieu de he thong
    cv2.putText(frame, "AI RED LIGHT VIOLATION SYSTEM", (20, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 255), 2)

    # Trang thai den giao thong
    state_color = (0, 0, 255) if traffic_state == "RED" else (
        (0, 255, 255) if traffic_state == "YELLOW" else (0, 255, 0)
    )
    cv2.putText(frame, "TIN HIEU DEN:", (460, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
    cv2.putText(frame, traffic_state, (610, 37),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, state_color, 2)
    cv2.circle(frame, (735, 30), 12, state_color, -1)
    cv2.circle(frame, (735, 30), 12, (255, 255, 255), 1)

    # Tong so vi pham
    cv2.putText(frame, "TONG VI PHAM:", (780, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
    cv2.putText(frame, str(total_violations), (950, 37),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255) if total_violations > 0 else (0, 255, 0), 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # 2. Bang canh bao vi pham moi nhat o goc duoi ben trai
    if recent_violations:
        panel_w, panel_h = 360, min(160, 35 + len(recent_violations) * 30)
        panel_x, panel_y = 20, h - panel_h - 20
        
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 255), 2)
        
        cv2.putText(frame, "DANH SACH VI PHAM GAN NHAT", (panel_x + 12, panel_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
        
        for i, viol in enumerate(reversed(recent_violations[-4:])):
            tid = viol.get("track_id") or viol.get("thong_tin_phuong_tien", {}).get("ma_dinh_danh_track_id", "?")
            vtype = viol.get("vehicle_type") or viol.get("thong_tin_phuong_tien", {}).get("loai_phuong_tien", "Xe")
            vcode = viol.get("ma_bien_ban", f"ID:{tid}")
            txt = f"[!] {vcode}: {vtype}"
            cv2.putText(frame, txt, (panel_x + 12, panel_y + 52 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

def run_detection(source, output_video=None, model_path=config.MODEL_PATH, headless=False):
    # Kiem tra nguon video (so camera hoac file path)
    if str(source).isdigit():
        source = int(source)
    else:
        source_path = Path(source)
        if not source_path.exists():
            print(f"[CANH BAO] File video '{source}' khong ton tai. Dang tao video demo mau...")
            from generate_demo_video import create_traffic_demo
            create_traffic_demo(output_path=str(source))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[LOI] Khong the mo nguon video: {source}")
        return

    # Khoi tao bo phat hien den & bo phat hien vi pham
    tl_detector = TrafficLightDetector(
        mode=config.TRAFFIC_LIGHT_MODE,
        roi=config.TRAFFIC_LIGHT_ROI,
        timer_cycle=config.TIMER_CYCLE
    )
    detector = RedLightViolationDetector(model_path=model_path, source_name=str(source))

    # Thiet lap VideoWriter neu nguoi dung muon luu video ket qua
    writer = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps_in, (w_in, h_in))
        print(f"[INFO] Se luu video ket qua ra: {output_video}")

    prev_time = time.time()
    fps = 0.0
    paused = False
    frame_idx = 0

    print("\n" + "="*50)
    print(" HE THONG PHAT HIEN VUOT DEN DO DANG CHAY")
    if not headless:
        print(" Phim tat:")
        print("   [q] : Thoat")
        print("   [t] : Chuyen trang thai den (Do <-> Vang <-> Xanh)")
        print("   [space] : Tam dung / Tiep tuc")
    print("="*50 + "\n")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Da xem het video.")
                break
            frame_idx += 1

            # 1. Cap nhat trang thai den
            traffic_state = tl_detector.update(frame)

            # 2. Xu ly tracking & phat hien vuot den do
            annotated_frame, new_violations = detector.process_frame(frame, traffic_state, frame_idx=frame_idx)

            # 3. Ve hop den giao thong
            tl_detector.draw(annotated_frame)

            # 4. Tinh FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else fps
            prev_time = curr_time

            # 5. Ve HUD Dashboard
            draw_hud(
                frame=annotated_frame,
                traffic_state=traffic_state,
                total_violations=len(detector.violated_ids),
                fps=fps,
                recent_violations=detector.violations_log
            )

            # Ghi video neu duoc yeu cau
            if writer:
                writer.write(annotated_frame)

            if not headless:
                cv2.imshow("Red Light Violation Detection", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    new_s = tl_detector.toggle_state()
                    print(f"[INFO] Da chuyen thu cong sang den: {new_s}")
                elif key == 32:  # phim space
                    paused = not paused
                    print(f"[INFO] {'Tam dung' if paused else 'Tiep tuc'}")

    cap.release()
    if writer:
        writer.release()
    if not headless:
        cv2.destroyAllWindows()
    print(f"\n[HOAN TAT] Tong so phuong tien vi pham: {len(detector.violated_ids)}")
    print(f"File bao cao phat nguoi JSON: {config.OUTPUT_DIR / 'traffic_citations.json'}")
    print(f"File bao cao tong hop Markdown: {config.OUTPUT_DIR / 'traffic_citations_report.md'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="He thong phat hien vuot den do bang YOLO")
    parser.add_argument("--source", type=str, default="demo.mp4", help="Duong dan video hoac ID camera (0, 1...)")
    parser.add_argument("--model", type=str, default=config.MODEL_PATH, help="Duong dan model YOLO (yolov8s.pt / best.pt)")
    parser.add_argument("--output", type=str, default="output_violation.mp4", help="Duong dan luu video ket qua")
    parser.add_argument("--headless", action="store_true", help="Chay khong hien cua so giao dien")
    args = parser.parse_args()

    run_detection(source=args.source, output_video=args.output, model_path=args.model, headless=args.headless)
