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
    def __init__(self, model_path=config.MODEL_PATH, conf=config.CONF_THRESHOLD, stop_lines=None, source_name="video", auto_detect_lines=True):
        self.detector = get_detector(model_path)
        self.model_name = self.detector.name
        self.conf = conf
        self.source_name = source_name
        
        # Danh sach cac vach dung (ho tro tu dong nhan dien thuc te theo video)
        if stop_lines is not None:
            raw_lines = stop_lines
        else:
            raw_lines = config.get_stop_lines(source_name, auto_detect=auto_detect_lines)
            
        # Chuan hoa ve danh sach dict neu nguoi dung truyen dang list cu
        if isinstance(raw_lines, list) and len(raw_lines) > 0 and isinstance(raw_lines[0], tuple):
            self.stop_lines = [{"id": "line_1", "name": "Vach dung", "line": raw_lines, "direction": config.MOVE_DIRECTION}]
        else:
            self.stop_lines = raw_lines or []

        print(f"[INFO] Khoi tao thanh cong voi model: {self.model_name}")
        if len(self.stop_lines) > 0:
            print(f"[INFO] He thong nhan dien va theo doi {len(self.stop_lines)} vach dung thuc te:")
            for s in self.stop_lines:
                print(f"   -> {s['name']}: {s['line']} (Huong: {s['direction']})")
        else:
            print(f"[CANH BAO] Video khong co hoac chua xac dinh duoc vach dung mat duong!")
        
        self.tracker = CentroidTracker()
        self.previous_positions = {}
        self.violated_ids = set()
        self.violations_log = []
        
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def _is_crossing_line(self, pt_prev, pt_curr, line_cfg):
        p1, p2 = line_cfg["line"]
        direction = line_cfg.get("direction", "TOP_TO_BOTTOM")

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B, C, D = p1, p2, pt_prev, pt_curr
        line_intersect = (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))
        
        if not line_intersect:
            y_line = (p1[1] + p2[1]) / 2
            x_min = min(p1[0], p2[0]) - 50
            x_max = max(p1[0], p2[0]) + 50
            
            # Kiem tra trong khoang chieu ngang cua vach
            if x_min <= pt_curr[0] <= x_max:
                if direction == "TOP_TO_BOTTOM":
                    line_intersect = (pt_prev[1] <= y_line and pt_curr[1] > y_line)
                elif direction == "BOTTOM_TO_TOP":
                    line_intersect = (pt_prev[1] >= y_line and pt_curr[1] < y_line)
                else:
                    line_intersect = abs(pt_prev[1] - y_line) <= 15 and abs(pt_curr[1] - y_line) <= 15
        return line_intersect

    def process_frame(self, frame, traffic_state: str, frame_idx: int = 0):
        current_violations = []
        boxes, scores, class_names = self.detector.detect(frame, conf_thresh=self.conf)

        annotated_frame = frame.copy()
        
        # Mau sac vach dung tuy theo trang thai den
        line_color = (0, 0, 255) if traffic_state == "RED" else (
            (0, 255, 255) if traffic_state == "YELLOW" else (0, 255, 0)
        )

        # Ve tat ca cac vach dung tren khung hinh
        for s_idx, s_cfg in enumerate(self.stop_lines):
            p1, p2 = s_cfg["line"]
            cv2.line(annotated_frame, p1, p2, line_color, 4)
            label_text = f"{s_cfg['name']}"
            cv2.putText(
                annotated_frame,
                label_text,
                (p1[0], max(25, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
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
                    if dist < min_dist and dist < 45:
                        min_dist = dist
                        matched_id = t_id
                        
                track_id = matched_id if matched_id is not None else (i + 1)
                is_violation = track_id in self.violated_ids

                # Kiem tra cat qua bat ky vach dung nao
                if not is_violation and track_id in self.previous_positions:
                    prev_pt = self.previous_positions[track_id]
                    if traffic_state == "RED":
                        for s_cfg in self.stop_lines:
                            if self._is_crossing_line(prev_pt, curr_pt, s_cfg):
                                is_violation = True
                                self.violated_ids.add(track_id)
                                violation_info = self._record_violation(
                                    frame=frame,
                                    box=box,
                                    track_id=track_id,
                                    cls_name=cls_name,
                                    conf=float(conf),
                                    frame_idx=frame_idx,
                                    line_cfg=s_cfg
                                )
                                current_violations.append(violation_info)
                                break

                self.previous_positions[track_id] = curr_pt

                if is_violation:
                    box_color = (0, 0, 255)
                    label = f"[PHAT NGUOI] ID:{track_id} {cls_name}"
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

    def _record_violation(self, frame, box, track_id, cls_name, conf, frame_idx, line_cfg):
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        citation_code = f"PN-{datetime.now().strftime('%Y%m%d')}-{track_id:04d}"
        line_name = line_cfg.get("name", "Vach dung")
        
        # 1. Luu anh hien truong toan canh
        scene_filename = self.output_dir / f"{citation_code}_scene.jpg"
        scene_img = frame.copy()
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(scene_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            scene_img,
            f"BIEN BAN PHAT NGUOI: {citation_code} - {cls_name} ID:{track_id} tai {line_name}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        cv2.imwrite(str(scene_filename), scene_img)

        # 2. Luu anh crop phuong tien
        h, w = frame.shape[:2]
        crop_x1 = max(0, x1 - 10)
        crop_y1 = max(0, y1 - 10)
        crop_x2 = min(w, x2 + 10)
        crop_y2 = min(h, y2 + 10)
        crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_filename = self.output_dir / f"{citation_code}_vehicle.jpg"
        if crop_img.size > 0:
            cv2.imwrite(str(crop_filename), crop_img)

        # 3. Tra cuu quy dinh phat nguoi theo Nghi dinh 100/123
        reg_info = config.PENALTY_REGULATIONS.get(cls_name, {
            "fine_range": "800.000 - 1.000.000 VNĐ",
            "suspension": "Tước GPLX 1 - 3 tháng",
            "legal_code": "Nghị định 100/2019/NĐ-CP"
        })

        # 4. Dong goi Thong bao phat nguoi chuan JSON
        citation_entry = {
            "ma_bien_ban": citation_code,
            "tieu_de": "THÔNG BÁO VI PHẠM TRẬT TỰ AN TOÀN GIAO THÔNG ĐƯỜNG BỘ (PHẠT NGUỘI)",
            "loai_vi_pham": "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông (Vượt đèn đỏ)",
            "can_cu_phap_ly": reg_info["legal_code"],
            "vach_vi_pham": line_name,
            "thong_tin_phuong_tien": {
                "ma_dinh_danh_track_id": int(track_id),
                "loai_phuong_tien": cls_name,
                "do_tin_cay_ai": f"{conf * 100:.1f}%",
                "mo_hinh_nhan_dien": self.model_name
            },
            "thoi_gian_dia_diem": {
                "thoi_diem_ghi_nhan": timestamp_str,
                "nguon_video": str(Path(self.source_name).name),
                "khung_hinh_frame": int(frame_idx),
                "tin_hieu_den": "RED (ĐÈN ĐỎ)",
                "toa_do_vach_vi_pham": line_cfg["line"]
            },
            "muc_xu_phat_du_kien": {
                "tien_phat": reg_info["fine_range"],
                "hinh_thuc_bo_sung": reg_info["suspension"],
                "bien_phap_khac": "Trừ điểm Giấy phép lái xe theo Luật TTATGTĐB mới"
            },
            "hinh_anh_bang_chung": {
                "anh_hien_truong": str(scene_filename.name),
                "anh_phuong_tien": str(crop_filename.name)
            },
            "trang_thai_xu_ly": "ĐÃ LẬP HỒ SƠ - CHỜ GỬI THÔNG BÁO PHẠT NGUỘI",
            # Shortcut fields cho giao dien
            "track_id": int(track_id),
            "vehicle_type": cls_name,
            "time": timestamp_str
        }
        
        self.violations_log.append(citation_entry)
        
        # Luu file JSON thong bao phat nguoi
        json_file = self.output_dir / "traffic_citations.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.violations_log, f, ensure_ascii=False, indent=2)

        # Xuat them bao cao Markdown de doc de dang
        self._export_markdown_report()

        print(f"\n[PHAT NGUOI] {citation_code} | {cls_name} (ID:{track_id}) | Tai: {line_name} | Phat: {reg_info['fine_range']}")
        return citation_entry

    def _export_markdown_report(self):
        """Xuat file bao cao tong hop cac bien ban phat nguoi ra Markdown"""
        md_file = self.output_dir / "traffic_citations_report.md"
        lines = [
            "# 🚨 BÁO CÁO TỔNG HỢP VI PHẠM VƯỢT ĐÈN ĐỎ (PHẠT NGUỘI)",
            f"*Thời gian xuất báo cáo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Tổng số phương tiện vi phạm ghi nhận: **{len(self.violations_log)}** vụ*",
            "",
            "---",
            "### 📋 Danh sách Biên bản Phạt nguội chi tiết:",
            ""
        ]

        for item in self.violations_log:
            c_code = item["ma_bien_ban"]
            v_info = item["thong_tin_phuong_tien"]
            p_info = item["muc_xu_phat_du_kien"]
            t_info = item["thoi_gian_dia_diem"]
            e_info = item["hinh_anh_bang_chung"]
            line_name = item.get("vach_vi_pham", "Vạch dừng")

            lines.extend([
                f"#### 📄 Biên bản: `{c_code}`",
                f"- **Phương tiện:** **{v_info['loai_phuong_tien']}** (ID: `{v_info['ma_dinh_danh_track_id']}`, Độ tin cậy AI: `{v_info['do_tin_cay_ai']}`)",
                f"- **Địa điểm vi phạm:** **{line_name}**",
                f"- **Thời điểm:** `{t_info['thoi_diem_ghi_nhan']}` (Frame: `{t_info['khung_hinh_frame']}` từ `{t_info['nguon_video']}`)",
                f"- **Hành vi:** {item['loai_vi_pham']}",
                f"- **Mức phạt dự kiến:** **{p_info['tien_phat']}** | {p_info['hinh_thuc_bo_sung']}",
                f"- **Căn cứ:** *{item['can_cu_phap_ly']}*",
                f"- **Bằng chứng:** Ảnh hiện trường: `{e_info['anh_hien_truong']}` | Ảnh xe: `{e_info['anh_phuong_tien']}`",
                f"- **Trạng thái:** `{item['trang_thai_xu_ly']}`",
                "",
                "---"
            ])

        with open(md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
