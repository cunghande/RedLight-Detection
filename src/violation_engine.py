from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger("ViolationEngine")


class ViolationEngine:
    """
    Logic Phát Hiện Phương Tiện Vượt Đèn Đỏ (Red Light Violation Engine):
    - Sử dụng khoảng cách có dấu: d = (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)
    - Nhận biết đổi dấu của d khi xe cắt qua vạch dừng TRONG LÚC ĐÈN ĐỎ.
    - Bộ lọc giảm báo sai (Anti-False-Alarm):
        1. Bỏ qua xe đã vượt qua vạch trước khi đèn đỏ bật.
        2. Yêu cầu vi phạm duy trì ít nhất K frame liên tiếp (mặc định K=3).
        3. Hỗ trợ cấu hình ngoại lệ rẽ phải (Right-turn exception).
        4. Mỗi ID phương tiện (Track ID) chỉ ghi nhận một sự kiện trong một lượt chạy.
    """

    def __init__(
        self,
        stop_lines: List[Dict],
        output_dir: Path,
        min_violation_frames: int = 3,
        allow_right_turn: bool = False,
        right_turn_x_ratio: float = 0.75
    ):
        self.stop_lines = stop_lines
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_violation_frames = min_violation_frames
        self.allow_right_turn = allow_right_turn
        self.right_turn_x_ratio = right_turn_x_ratio

        # Quản lý trạng thái từng phương tiện (Track ID)
        self.previous_positions: Dict[int, Tuple[int, int]] = {}
        self.previous_signs: Dict[int, Dict[str, float]] = {} # {track_id: {line_id: sign}}
        self.violated_ids: Set[int] = set()
        self.violation_consecutive_counts: Dict[Tuple[int, str], int] = {}
        self.violation_last_frame: Dict[Tuple[int, str], int] = {}
        self.passed_before_red: Set[int] = set() # Các xe đã qua vạch trước khi đèn đỏ
        self.violations_log: List[Dict] = []

    def update_stop_lines(self, new_lines: List[Dict]):
        """Cập nhật tọa độ vạch dừng khi có chống rung hoặc thay đổi động"""
        self.stop_lines = new_lines

    def _compute_signed_distance(self, pt: Tuple[int, int], line: List[Tuple[int, int]]) -> float:
        """
        Khoảng cách có dấu từ điểm pt(x, y) tới đường thẳng nối line(p1 -> p2):
        d = (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)
        """
        x, y = pt
        (x1, y1), (x2, y2) = line
        return float((x - x1) * (y2 - y1) - (y - y1) * (x2 - x1))

    def process_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        traffic_state: str,
        frame_idx: int = 0
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Xử lý từng phương tiện trong khung hình:
        tracks: danh sách dict gồm: {'id': int, 'box': [x1,y1,x2,y2], 'class_name': str, 'conf': float}
        """
        annotated_frame = frame.copy()
        current_frame_violations = []
        h, w = frame.shape[:2]

        # 1. Vẽ các vạch dừng lên khung hình
        line_color = (0, 0, 255) if traffic_state == "RED" else (
            (0, 255, 255) if traffic_state == "YELLOW" else (0, 255, 0)
        )
        for s in self.stop_lines:
            p1, p2 = s["line"]
            cv2.line(annotated_frame, p1, p2, line_color, 4)
            cv2.putText(
                annotated_frame, s["name"], (p1[0], max(25, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2
            )

        # 2. Xử lý từng phương tiện
        for tr in tracks:
            track_id = tr["id"]
            x1, y1, x2, y2 = tr["box"]
            cls_name = tr.get("class_name", "O to")
            conf = tr.get("conf", 0.8)

            # Điểm tham chiếu: trung điểm cạnh đáy bbox
            ref_pt = (int((x1 + x2) / 2), int(y2))

            # Tính toán khoảng cách có dấu tới từng vạch dừng
            if track_id not in self.previous_signs:
                self.previous_signs[track_id] = {}

            for line_cfg in self.stop_lines:
                line_id = line_cfg["id"]
                current_d = self._compute_signed_distance(ref_pt, line_cfg["line"])
                start, end = line_cfg["line"]
                line_sign = 1 if end[0] >= start[0] else -1
                direction = line_cfg.get("direction", "TOP_TO_BOTTOM")
                current_after = current_d * line_sign * (1 if direction == "BOTTOM_TO_TOP" else -1) > 0
                prev_d = self.previous_signs[track_id].get(line_id)
                prev_after = (prev_d * line_sign * (1 if direction == "BOTTOM_TO_TOP" else -1) > 0
                              if prev_d is not None else None)
                key = (track_id, line_id)

                if traffic_state != "RED":
                    self.violation_consecutive_counts.pop(key, None)
                    if current_after:
                        self.passed_before_red.add(track_id)
                elif track_id not in self.violated_ids and track_id not in self.passed_before_red:
                    in_segment = min(start[0], end[0]) <= ref_pt[0] <= max(start[0], end[0])
                    crossing = prev_after is False and current_after and in_segment
                    if crossing and self.allow_right_turn and ref_pt[0] >= w * self.right_turn_x_ratio:
                        crossing = False
                    continuing = (key in self.violation_consecutive_counts and current_after
                                  and self.violation_last_frame.get(key) == frame_idx - 1)
                    if crossing or continuing:
                        count = self.violation_consecutive_counts.get(key, 0) + 1
                        self.violation_consecutive_counts[key] = count
                        self.violation_last_frame[key] = frame_idx
                        if count >= self.min_violation_frames:
                            self.violated_ids.add(track_id)
                            current_frame_violations.append(self._record_violation(
                                frame=frame, box=[x1, y1, x2, y2], track_id=track_id,
                                cls_name=cls_name, conf=conf, frame_idx=frame_idx, line_cfg=line_cfg))
                    elif not current_after or key in self.violation_consecutive_counts:
                        self.violation_consecutive_counts.pop(key, None)

                self.previous_signs[track_id][line_id] = current_d

            self.previous_positions[track_id] = ref_pt

            # 3. Vẽ Bounding Box và Nhãn Phương Tiện
            is_violation = track_id in self.violated_ids
            box_color = (0, 0, 255) if is_violation else (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2 if not is_violation else 3)
            cv2.circle(annotated_frame, ref_pt, 4, (0, 255, 255), -1) # Điểm tham chiếu

            label = f"ID:{track_id} {cls_name} {conf:.2f}"
            if is_violation:
                label = f"[VI PHAM] {label}"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated_frame,
                (x1, max(0, y1 - lh - 6)),
                (x1 + lw + 6, y1),
                box_color, -1
            )
            cv2.putText(
                annotated_frame, label,
                (x1 + 3, max(lh + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return annotated_frame, current_frame_violations

    def _record_violation(
        self,
        frame: np.ndarray,
        box: List[int],
        track_id: int,
        cls_name: str,
        conf: float,
        frame_idx: int,
        line_cfg: Dict
    ) -> Dict:
        """Lưu ảnh của sự kiện nghi vượt đèn đỏ để người dùng kiểm tra."""
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        event_id = f"EV-{now.strftime('%Y%m%d-%H%M%S-%f')}-{track_id:04d}"
        line_name = line_cfg.get("name", "Vach dung")

        # 1. Lưu ảnh hiện trường
        scene_filename = self.output_dir / f"{event_id}_scene.jpg"
        scene_img = frame.copy()
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(scene_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            scene_img,
            f"CAN KIEM TRA: {event_id} - {cls_name} ID:{track_id}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        cv2.imwrite(str(scene_filename), scene_img)

        # 2. Lưu ảnh crop xe
        h, w = frame.shape[:2]
        crop_img = frame[max(0, y1 - 10):min(h, y2 + 10), max(0, x1 - 10):min(w, x2 + 10)]
        crop_filename = self.output_dir / f"{event_id}_vehicle.jpg"
        if crop_img.size > 0:
            cv2.imwrite(str(crop_filename), crop_img)

        entry = {
            "event_id": event_id,
            "track_id": track_id,
            "vehicle_type": cls_name,
            "confidence": round(float(conf), 2),
            "frame_index": frame_idx,
            "timestamp": timestamp_str,
            "stop_line_violated": line_name,
            "scene_image": str(scene_filename.name),
            "vehicle_crop_image": str(crop_filename.name)
        }

        self.violations_log.append(entry)
        logger.warning("[CẦN KIỂM TRA] %s | %s (ID:%s) | %s", event_id, cls_name, track_id, line_name)
        return entry
