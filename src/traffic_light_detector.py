import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("TrafficLightDetector")


class TrafficLightDetector:
    """
    Module Nhận Diện Trạng Thái Đèn Tín Hiệu Giao Thông (Đỏ / Vàng / Xanh).
    - Sử dụng YOLO để tự động phát hiện cụm đèn (class 'traffic light' - COCO class 9) trên toàn khung hình.
    - Cắt vùng đèn (Crop ROI) và phân tích không gian màu HSV để xác định bóng đèn đang bật.
    - Hỗ trợ cơ chế giữ trạng thái (State Persistence) khi đèn tạm thời bị che khuất và phím tắt chuyển thủ công [t].
    """

    def __init__(self, yolo_model=None, conf_thresh: float = 0.25, hsv_ranges: Optional[Dict] = None,
                 max_missing_frames: int = 10):
        self.model = yolo_model
        self.conf_thresh = conf_thresh
        self.current_state = "UNKNOWN"
        self.max_missing_frames = max_missing_frames
        self.missing_frames = 0
        self.consecutive_count = 0
        self.detected_boxes: List[Tuple[int, int, int, int]] = []

        # Khoảng màu HSV mặc định nếu không truyền
        self.hsv_ranges = hsv_ranges or {
            "RED": {
                "lower1": np.array([0, 100, 100]),
                "upper1": np.array([10, 255, 255]),
                "lower2": np.array([160, 100, 100]),
                "upper2": np.array([180, 255, 255])
            },
            "YELLOW": {
                "lower": np.array([15, 100, 100]),
                "upper": np.array([35, 255, 255])
            },
            "GREEN": {
                "lower": np.array([40, 100, 100]),
                "upper": np.array([90, 255, 255])
            }
        }

    def detect_state(self, frame: np.ndarray, yolo_detections: Optional[List] = None) -> Tuple[str, List[Tuple[int, int, int, int]]]:
        """
        Nhận diện trạng thái đèn từ khung hình:
        - yolo_detections: nếu pipeline đã chạy YOLO detect trước, truyền vào để tái sử dụng bbox class 9 mà không cần chạy lại YOLO.
        """
        boxes = []

        if yolo_detections is not None:
            # yolo_detections là danh sách (box, cls_id, conf)
            for box, cls_id, conf in yolo_detections:
                if int(cls_id) == 9 and conf >= self.conf_thresh:
                    boxes.append(tuple(map(int, box)))
        elif self.model is not None:
            results = self.model.predict(frame, conf=self.conf_thresh, classes=[9], verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None:
                for b in results[0].boxes.xyxy.cpu().numpy():
                    boxes.append(tuple(map(int, b)))

        self.detected_boxes = boxes

        if not boxes:
            self.missing_frames += 1
            if self.missing_frames >= self.max_missing_frames:
                self.current_state = "UNKNOWN"
            return self.current_state, []

        state_votes = {"RED": 0, "YELLOW": 0, "GREEN": 0}

        for (x1, y1, x2, y2) in boxes:
            h, w = frame.shape[:2]
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            if cx2 - cx1 < 5 or cy2 - cy1 < 10:
                continue

            crop = frame[cy1:cy2, cx1:cx2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # Tính mật độ pixel màu đỏ
            mask_r1 = cv2.inRange(hsv, self.hsv_ranges["RED"]["lower1"], self.hsv_ranges["RED"]["upper1"])
            mask_r2 = cv2.inRange(hsv, self.hsv_ranges["RED"]["lower2"], self.hsv_ranges["RED"]["upper2"])
            red_pixels = cv2.countNonZero(cv2.bitwise_or(mask_r1, mask_r2))

            # Tính mật độ pixel màu vàng
            yellow_pixels = cv2.countNonZero(cv2.inRange(hsv, self.hsv_ranges["YELLOW"]["lower"], self.hsv_ranges["YELLOW"]["upper"]))

            # Tính mật độ pixel màu xanh
            green_pixels = cv2.countNonZero(cv2.inRange(hsv, self.hsv_ranges["GREEN"]["lower"], self.hsv_ranges["GREEN"]["upper"]))

            max_val = max(red_pixels, yellow_pixels, green_pixels)
            if max_val >= 10:
                if max_val == red_pixels:
                    state_votes["RED"] += 1
                elif max_val == yellow_pixels:
                    state_votes["YELLOW"] += 1
                else:
                    state_votes["GREEN"] += 1

        # Quyết định trạng thái theo số phiếu
        if not any(state_votes.values()):
            self.missing_frames += 1
            if self.missing_frames >= self.max_missing_frames:
                self.current_state = "UNKNOWN"
            return self.current_state, boxes
        self.missing_frames = 0
        if state_votes["RED"] > 0:
            self.current_state = "RED"
        elif state_votes["YELLOW"] > 0 and state_votes["YELLOW"] >= state_votes["GREEN"]:
            self.current_state = "YELLOW"
        elif state_votes["GREEN"] > 0:
            self.current_state = "GREEN"

        return self.current_state, boxes

    def toggle_state(self) -> str:
        """Chuyển trạng thái đèn thủ công khi nhấn phím [t]"""
        cycle = {"RED": "GREEN", "GREEN": "YELLOW", "YELLOW": "RED"}
        self.current_state = cycle.get(self.current_state, "GREEN")
        return self.current_state
