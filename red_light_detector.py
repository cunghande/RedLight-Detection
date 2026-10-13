import cv2
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np

import config
from model_factory import get_detector

class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=90):
        self.next_object_id = 1
        self.objects = {}  # {id: centroid}
        self.disappeared = {}  # {id: count}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, rects):
        """rects: np.ndarray shape (N, 4) -> dict {id: centroid}"""
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return {}

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2) / 2), int(y2))  # diem tiep dat

        if len(self.objects) == 0:
            for i in range(len(rects)):
                obj_id = self.next_object_id
                self.objects[obj_id] = input_centroids[i]
                self.disappeared[obj_id] = 0
                self.next_object_id += 1
            return self.objects.copy()

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_cols = set(range(input_centroids.shape[0])).difference(used_cols)
        for col in unused_cols:
            obj_id = self.next_object_id
            self.objects[obj_id] = input_centroids[col]
            self.disappeared[obj_id] = 0
            self.next_object_id += 1

        unused_rows = set(range(len(object_centroids))).difference(used_rows)
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                del self.objects[obj_id]
                del self.disappeared[obj_id]

        return self.objects.copy()

class RedLightViolationDetector:
    def __init__(self, model_path=config.MODEL_PATH, conf=config.CONF_THRESHOLD):
        self.detector = get_detector(model_path)
        self.model_name = self.detector.name
        print(f"[INFO] He thong khoi tao thanh cong voi model: {self.model_name}")
        self.conf = conf
        self.stop_line = config.STOP_LINE
        self.direction = config.MOVE_DIRECTION
        
        self.tracker = CentroidTracker()
        self.previous_positions = {}
        self.violated_ids = set()
        self.violations_log = []
        
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def _is_crossing_line(self, pt_prev, pt_curr):
        p1, p2 = self.stop_line
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B, C, D = p1, p2, pt_prev, pt_curr
        line_intersect = (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))
        
        if not line_intersect:
            y_line = (p1[1] + p2[1]) / 2
            if self.direction == "TOP_TO_BOTTOM":
                line_intersect = (pt_prev[1] <= y_line and pt_curr[1] > y_line)
            else:
                line_intersect = (pt_prev[1] >= y_line and pt_curr[1] < y_line)
        return line_intersect

    def process_frame(self, frame, traffic_state: str):
        current_violations = []
        boxes, scores, class_names = self.detector.detect(frame, conf_thresh=self.conf)

        annotated_frame = frame.copy()
        line_color = (0, 0, 255) if traffic_state == "RED" else (
            (0, 255, 255) if traffic_state == "YELLOW" else (0, 255, 0)
        )
        p1, p2 = self.stop_line
        cv2.line(annotated_frame, p1, p2, line_color, 4)
        cv2.putText(
            annotated_frame,
            "VACH DUNG (STOP LINE)",
            (p1[0], max(30, p1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            line_color,
            2
        )

        if len(boxes) > 0:
            tracked_objects = self.tracker.update(boxes)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                curr_pt = (int((x1 + x2) / 2), int(y2))
                cls_name = class_names[i] if i < len(class_names) else "Phuong tien"
                conf = scores[i] if i < len(scores) else 0.5

                # Tim track_id phu hop nhat tu tracker
                matched_id = None
                min_dist = float("inf")
                for t_id, t_centroid in tracked_objects.items():
                    dist = np.linalg.norm(np.array(curr_pt) - np.array(t_centroid))
                    if dist < min_dist and dist < 40:
                        min_dist = dist
                        matched_id = t_id
                        
                track_id = matched_id if matched_id is not None else (i + 1)

                is_violation = track_id in self.violated_ids

                if not is_violation and track_id in self.previous_positions:
                    prev_pt = self.previous_positions[track_id]
                    if traffic_state == "RED" and self._is_crossing_line(prev_pt, curr_pt):
                        is_violation = True
                        self.violated_ids.add(track_id)
                        violation_info = self._record_violation(
                            frame=frame,
                            box=box,
                            track_id=track_id,
                            cls_name=cls_name,
                            conf=float(conf)
                        )
                        current_violations.append(violation_info)

                self.previous_positions[track_id] = curr_pt

                if is_violation:
                    box_color = (0, 0, 255)
                    label = f"[VUOT DEN DO] ID:{track_id} {cls_name}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
                    cv2.circle(annotated_frame, curr_pt, 6, (0, 0, 255), -1)
                else:
                    box_color = (0, 220, 220)
                    label = f"ID:{track_id} {cls_name} {conf:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.circle(annotated_frame, curr_pt, 4, (0, 255, 0), -1)

                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(
                    annotated_frame,
                    (x1, max(0, y1 - lh - 8)),
                    (x1 + lw + 6, max(lh + 8, y1)),
                    box_color,
                    -1
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 3, max(lh + 3, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

        return annotated_frame, current_violations

    def _record_violation(self, frame, box, track_id, cls_name, conf):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        v_id = f"VIOLATION_{timestamp}_ID{track_id}"
        
        scene_filename = self.output_dir / f"{v_id}_scene.jpg"
        scene_img = frame.copy()
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(scene_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            scene_img,
            f"VIOLATION: RED LIGHT RUNNER - ID:{track_id} ({cls_name})",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        cv2.imwrite(str(scene_filename), scene_img)

        h, w = frame.shape[:2]
        crop_x1 = max(0, x1 - 10)
        crop_y1 = max(0, y1 - 10)
        crop_x2 = min(w, x2 + 10)
        crop_y2 = min(h, y2 + 10)
        crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_filename = self.output_dir / f"{v_id}_vehicle.jpg"
        if crop_img.size > 0:
            cv2.imwrite(str(crop_filename), crop_img)

        info = {
            "violation_id": v_id,
            "track_id": int(track_id),
            "vehicle_type": cls_name,
            "confidence": round(float(conf), 3),
            "model_used": self.model_name,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scene_image": str(scene_filename.name),
            "vehicle_image": str(crop_filename.name)
        }
        self.violations_log.append(info)
        
        log_file = self.output_dir / "violations.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.violations_log, f, ensure_ascii=False, indent=2)

        print(f"[CANH BAO] PHAT HIEN VUOT DEN DO! ID: {track_id} | Loai xe: {cls_name} | Model: {self.model_name}")
        return info
