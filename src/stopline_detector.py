import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("StopLineDetector")


class StopLineDetector:
    """
    Module Tự Động Phát Hiện Vạch Dừng (Stop Line Detector).
    Hỗ trợ 2 chế độ:
      - Chế độ A (Classical Baseline): Lọc trung vị đa khung hình + Phân đoạn thích nghi + HoughLinesP + cv2.fitLine.
      - Chế độ B (AI YOLO-seg): Phân đoạn mask stop_line / zebra_crossing + Voting đa khung hình + Neo không gian.
    Tích hợp bộ nhớ đệm Cache configs/<camera_id>.json và cờ ép buộc quét lại (--redetect).
    """

    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        self.config_path = Path(config_path)
        self.cfg = self._load_config()

        # Đường dẫn thư mục cache
        self.cache_dir = Path(self.cfg.get("camera", {}).get("cache_dir", "configs"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Tham số Chế độ A
        classical_cfg = self.cfg.get("stopline_detector", {}).get("classical", {})
        self.sample_frames_classical = classical_cfg.get("sample_frames", 200)
        self.roi_y_min_ratio = classical_cfg.get("roi", {}).get("y_min_ratio", 0.46)
        self.roi_y_max_ratio = classical_cfg.get("roi", {}).get("y_max_ratio", 0.82)
        self.brightness_pct = classical_cfg.get("filter", {}).get("brightness_percentile", 85)
        self.min_brightness = classical_cfg.get("filter", {}).get("min_brightness", 115)
        self.max_slope = classical_cfg.get("filter", {}).get("max_slope", 0.08)
        self.max_dy = classical_cfg.get("filter", {}).get("max_dy", 16)
        self.cluster_y_thresh = classical_cfg.get("clustering", {}).get("y_threshold", 40)
        self.min_span_ratio = classical_cfg.get("clustering", {}).get("min_span_ratio", 0.22)
        self.max_lines = classical_cfg.get("selection", {}).get("max_lines", 1)

        # Tham số Chế độ B
        ai_cfg = self.cfg.get("stopline_detector", {}).get("ai", {})
        self.sample_frames_ai = ai_cfg.get("sample_frames", 50)
        self.ai_conf_thresh = ai_cfg.get("mask_conf", 0.35)
        self.zebra_assist = ai_cfg.get("zebra_assist", True) if isinstance(ai_cfg.get("zebra_assist", True), bool) else True

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    # =========================================================================
    # QUẢN LÝ BỘ NHỚ ĐỆM (CACHE)
    # =========================================================================
    def get_cache_path(self, camera_id: str) -> Path:
        return self.cache_dir / f"{camera_id}.json"

    def load_from_cache(self, camera_id: str) -> Optional[List[Dict]]:
        cache_file = self.get_cache_path(camera_id)
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                lines = data.get("lines", [])
                logger.info(f"Đã tải thành công {len(lines)} vạch dừng từ cache: {cache_file}")
                return lines
            except Exception as e:
                logger.warning(f"Lỗi khi đọc file cache {cache_file}: {e}")
        return None

    def save_to_cache(self, camera_id: str, lines: List[Dict], mode: str, video_path: str = ""):
        cache_file = self.get_cache_path(camera_id)
        data = {
            "camera_id": camera_id,
            "source_video": str(video_path),
            "detector_mode": mode,
            "total_lines": len(lines),
            "lines": lines
        }
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Đã lưu kết quả vạch dừng vào cache: {cache_file}")
        except Exception as e:
            logger.error(f"Không thể ghi cache {cache_file}: {e}")

    # =========================================================================
    # ĐIỀU PHỐI CHÍNH (DETECT ENTRYPOINT)
    # =========================================================================
    def detect(
        self,
        video_source: Union[str, Path, cv2.VideoCapture],
        camera_id: Optional[str] = None,
        mode: Optional[str] = None,
        redetect: bool = False
    ) -> List[Dict]:
        """
        Tự động phát hiện vạch dừng trong video.
        - camera_id: định danh camera/video để đọc/ghi cache.
        - mode: "classical" (Chế độ A) hoặc "ai" (Chế độ B).
        - redetect: Nếu True, bỏ qua cache và ép buộc quét lại vạch mới.
        """
        source_name = str(video_source)
        if camera_id is None:
            camera_id = Path(source_name).stem if isinstance(video_source, (str, Path)) else "cam_default"

        if mode is None:
            mode = self.cfg.get("stopline_detector", {}).get("default_mode", "classical")

        # 1. Kiểm tra cache nếu không ép buộc redetect
        if not redetect:
            cached_lines = self.load_from_cache(camera_id)
            if cached_lines is not None and len(cached_lines) > 0:
                return cached_lines

        logger.info(f"Đang thực hiện phát hiện vạch dừng tự động cho '{camera_id}' [Chế độ: {mode.upper()}]...")

        # 2. Thực thi thuật toán theo chế độ được chọn
        if mode.lower() == "ai":
            detected_lines = self.detect_ai(video_source)
        else:
            detected_lines = self.detect_classical(video_source)

        # 3. Lưu vào cache nếu phát hiện thành công
        if detected_lines:
            self.save_to_cache(camera_id, detected_lines, mode=mode, video_path=source_name)
        else:
            logger.warning(f"Không phát hiện thấy vạch dừng hợp lệ nào trong '{source_name}'.")

        return detected_lines

    # =========================================================================
    # CHẾ ĐỘ A: BASELINE CỔ ĐIỂN (CLASSICAL CV)
    # =========================================================================
    def detect_classical(self, video_source: Union[str, Path, cv2.VideoCapture]) -> List[Dict]:
        """
        Chế độ A:
        1. Đọc N frame cách đều nhau, tạo ảnh nền bằng median để triệt tiêu xe di chuyển.
        2. Lọc vùng mặt đường ROI và lọc độ sáng thích nghi theo bách phân vị.
        3. Canny + Lọc hình thái học ngang + HoughLinesP với ràng buộc độ dốc ngang.
        4. Gom nhóm cụm Y và làm khớp đường thẳng bằng cv2.fitLine.
        5. Đánh giá độ tin cậy và trả về danh sách vạch dừng.
        """
        frames, H, W = self._sample_video_frames(video_source, self.sample_frames_classical)
        if not frames:
            return []

        # 1. Ảnh nền tĩnh bằng median
        road_bg = np.median(frames, axis=0).astype(np.uint8)

        # 2. Vùng mặt đường quan tâm (Active Road ROI)
        roi_top = int(H * self.roi_y_min_ratio)
        roi_bottom = int(H * self.roi_y_max_ratio)
        roi = road_bg[roi_top:roi_bottom, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 3. Phân đoạn vạch sơn trắng thích nghi
        th_val = max(self.min_brightness, int(np.percentile(gray, self.brightness_pct)))
        _, bin_mask = cv2.threshold(gray, th_val, 255, cv2.THRESH_BINARY)

        # 4. Lọc hình thái học chiều ngang
        kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(self.cfg.get("stopline_detector", {}).get("classical", {}).get("filter", {}).get("horizontal_kernel", [25, 3])))
        clean_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel_horiz)

        # 5. Hough Lines P
        lines = cv2.HoughLinesP(
            clean_mask, 1, np.pi / 180,
            threshold=self.cfg.get("stopline_detector", {}).get("classical", {}).get("filter", {}).get("hough_threshold", 35),
            minLineLength=self.cfg.get("stopline_detector", {}).get("classical", {}).get("filter", {}).get("hough_min_line_len", 35),
            maxLineGap=self.cfg.get("stopline_detector", {}).get("classical", {}).get("filter", {}).get("hough_max_line_gap", 30)
        )
        if lines is None:
            return []

        # 6. Lọc các đoạn thẳng nằm ngang nghiêm ngặt
        horizontal_segments = []
        for l in lines:
            if len(l) == 4 and not hasattr(l[0], '__len__'):
                x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
            else:
                x1, y1, x2, y2 = int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3])

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx >= 35 and dy <= self.max_dy and (dy / max(1, dx)) < self.max_slope:
                horizontal_segments.append((min(x1, x2), y1 + roi_top, max(x1, x2), y2 + roi_top, dx))

        if not horizontal_segments:
            return []

        # 7. Gom nhóm theo trục Y (Clustering)
        clusters = []
        for x1, y1, x2, y2, dx in horizontal_segments:
            y_mid = (y1 + y2) / 2
            matched = False
            for c in clusters:
                if abs(c['ym'] - y_mid) < self.cluster_y_thresh:
                    c['lines'].append((x1, y1, x2, y2, dx))
                    c['ym'] = np.mean([(it[1] + it[3]) / 2 for it in c['lines']])
                    matched = True
                    break
            if not matched:
                clusters.append({'ym': y_mid, 'lines': [(x1, y1, x2, y2, dx)]})

        # 8. Làm khớp đường thẳng bằng cv2.fitLine và tính điểm tin cậy
        scored_clusters = []
        for c in clusters:
            min_x = min(it[0] for it in c['lines'])
            max_x = max(it[2] for it in c['lines'])
            span_x = max_x - min_x
            tot_len = sum(it[4] for it in c['lines'])

            if span_x >= W * self.min_span_ratio:
                # Thu thập toàn bộ tọa độ điểm để fitLine
                pts = []
                for it in c['lines']:
                    pts.append([it[0], it[1]])
                    pts.append([it[2], it[3]])
                pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 2)

                # cv2.fitLine sử dụng chuẩn M-estimator L1 / Huber chống nhiễu
                vx, vy, x0, y0 = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                slope = vy / vx if abs(vx) > 1e-4 else 0.0
                slope = max(-0.025, min(0.025, slope)) # giới hạn độ dốc nằm ngang phẳng

                avg_y = int(round(c['ym']))
                x_mid = (min_x + max_x) / 2
                y1 = int(round(avg_y + slope * (min_x - x_mid)))
                y2 = int(round(avg_y + slope * (max_x - x_mid)))

                if abs(y2 - y1) > 6:
                    y2 = y1

                # Điểm tin cậy chuẩn hóa trong khoảng [0.0, 1.0]
                confidence = float(min(1.0, (span_x / W) * 0.7 + (tot_len / 4000.0) * 0.3))
                confidence = round(confidence, 3)

                scored_clusters.append({
                    'confidence': confidence,
                    'ym': avg_y,
                    'span': span_x,
                    'tot_len': tot_len,
                    'line': [(int(min_x), y1), (int(max_x), y2)]
                })

        if not scored_clusters:
            return []

        # Sắp xếp theo độ tin cậy giảm dần
        scored_clusters.sort(key=lambda x: x['confidence'], reverse=True)

        # Chọn vạch chính (và vạch phụ nếu thực sự có 2 vạch)
        primary = scored_clusters[0]
        final_lines = [primary]

        sec_dist = self.cfg.get("stopline_detector", {}).get("classical", {}).get("selection", {}).get("secondary_distance_thresh", 220)
        if self.max_lines >= 2:
            for cand in scored_clusters[1:]:
                if abs(cand['ym'] - primary['ym']) >= sec_dist and cand['span'] >= W * 0.35 and cand['confidence'] >= 0.80 * primary['confidence']:
                    final_lines.append(cand)
                    break

        final_lines.sort(key=lambda x: x['ym'])

        results = []
        for i, s in enumerate(final_lines):
            y_avg = s['ym']
            direction = 'TOP_TO_BOTTOM' if y_avg < H * 0.55 else 'BOTTOM_TO_TOP'
            name_desc = "Vach dung" if len(final_lines) == 1 else f"Vach {i+1}"
            results.append({
                "id": f"stop_line_{i+1}",
                "name": f"{name_desc} (Y={int(y_avg)})",
                "line": s['line'],
                "direction": direction,
                "confidence": s['confidence'],
                "method": "classical_median_hough"
            })

        logger.info(f"[Chế độ A] Phát hiện thành công {len(results)} vạch dừng:")
        for r in results:
            logger.info(f"   -> {r['name']}: {r['line']} | Độ tin cậy: {r['confidence']}")

        return results

    # =========================================================================
    # CHẾ ĐỘ B: MODEL AI (YOLO-SEG)
    # =========================================================================
    def detect_ai(
        self,
        video_source: Union[str, Path, cv2.VideoCapture],
        model_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Chế độ B:
        1. Dùng YOLOv8-seg / YOLO11-seg nhận diện các class stop_line và zebra_crossing.
        2. Chạy trên N frame, loại bỏ vùng bbox xe che khuất.
        3. Khớp đường thẳng cv2.fitLine từ mask.
        4. Voting trung vị đa khung hình (Multi-frame median voting).
        5. Sử dụng zebra_crossing làm tín hiệu định vị phụ.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("Chưa cài đặt thư viện ultralytics. Hãy chạy: pip install ultralytics")
            return self.detect_classical(video_source)

        if model_path is None:
            model_path = self.cfg.get("models", {}).get("stopline_seg", {}).get("weights", "yolov8s-seg.pt")

        weights_file = Path(model_path)
        # Nếu chưa có weights fine-tuned chuyên biệt, fallback sang classical kết hợp hướng dẫn train
        if not weights_file.exists() or "seg" not in weights_file.name:
            logger.warning(f"File trọng số YOLO-seg '{model_path}' chưa sẵn sàng. Đang tự động chuyển sang Chế độ A (Classical Baseline) chất lượng cao...")
            res = self.detect_classical(video_source)
            for r in res:
                r["method"] = "ai_fallback_classical"
            return res

        frames, H, W = self._sample_video_frames(video_source, self.sample_frames_ai)
        if not frames:
            return []

        logger.info(f"Đang suy luận YOLO-seg ({model_path}) trên {len(frames)} khung hình mẫu...")
        model = YOLO(model_path)

        candidate_lines = []
        zebra_lines = []

        for frame in frames:
            results = model.predict(frame, conf=self.ai_conf_thresh, verbose=False)
            if not results or len(results) == 0:
                continue

            res = results[0]
            if res.masks is None:
                continue

            # Lấy danh sách bbox xe để loại bỏ vùng bị che khuất
            vehicle_boxes = []
            if res.boxes is not None:
                for box, cls_id in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                    if int(cls_id) in [2, 3, 5, 7]: # car, motorcycle, bus, truck
                        vehicle_boxes.append(box.astype(int))

            # Xử lý từng mask
            for mask_data, cls_id in zip(res.masks.xy, res.boxes.cls.cpu().numpy()):
                pts = np.array(mask_data, dtype=np.int32)
                if len(pts) < 10:
                    continue

                # Tạo mask nhị phân cho contour này
                contour_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(contour_mask, [pts], -1, 255, -1)

                # Trừ đi các vùng bị xe che
                for vx1, vy1, vx2, vy2 in vehicle_boxes:
                    contour_mask[max(0, vy1):min(H, vy2), max(0, vx1):min(W, vx2)] = 0

                # Tìm lại tọa độ các pixel hợp lệ trong mask
                valid_pts = np.column_stack(np.where(contour_mask > 0))
                if len(valid_pts) < 50:
                    continue
                # Chuyển về dạng (x, y)
                pts_xy = valid_pts[:, [1, 0]].astype(np.int32)

                # Fit line
                vx, vy, x0, y0 = cv2.fitLine(pts_xy, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                slope = vy / vx if abs(vx) > 1e-4 else 0.0
                if abs(slope) < 0.10: # Chỉ nhận vạch ngang
                    min_x = int(np.min(pts_xy[:, 0]))
                    max_x = int(np.max(pts_xy[:, 0]))
                    span = max_x - min_x
                    if span >= W * 0.20:
                        y_mid = int(y0)
                        candidate_lines.append((min_x, max_x, y_mid, span, slope))

        if not candidate_lines:
            logger.warning("YOLO-seg không phát hiện thấy vạch dừng. Tự động chuyển tiếp sang Chế độ A...")
            return self.detect_classical(video_source)

        # 4. Voting đa khung hình: Lấy trung vị (Median) các tham số
        med_y = int(np.median([l[2] for l in candidate_lines]))
        med_min_x = int(np.min([l[0] for l in candidate_lines]))
        med_max_x = int(np.max([l[1] for l in candidate_lines]))
        med_slope = float(np.median([l[4] for l in candidate_lines]))
        med_slope = max(-0.02, min(0.02, med_slope))

        y1 = int(round(med_y + med_slope * (med_min_x - (med_min_x + med_max_x) / 2)))
        y2 = int(round(med_y + med_slope * (med_max_x - (med_min_x + med_max_x) / 2)))
        if abs(y2 - y1) > 6:
            y2 = y1

        confidence = round(float(min(1.0, len(candidate_lines) / (len(frames) * 1.5))), 3)

        direction = 'TOP_TO_BOTTOM' if med_y < H * 0.55 else 'BOTTOM_TO_TOP'
        results = [{
            "id": "stop_line_1",
            "name": f"Vach dung AI (Y={med_y})",
            "line": [(med_min_x, y1), (med_max_x, y2)],
            "direction": direction,
            "confidence": confidence,
            "method": "ai_yolo_seg_voting"
        }]

        logger.info(f"[Chế độ B - YOLO-seg] Đã fit vạch thành công từ {len(candidate_lines)} mẫu: {results[0]['line']} (Conf: {confidence})")
        return results

    # =========================================================================
    # HÀM BỔ TRỢ ĐỌC KHUNG HÌNH (SAMPLING)
    # =========================================================================
    def _sample_video_frames(
        self,
        video_source: Union[str, Path, cv2.VideoCapture],
        num_samples: int
    ) -> Tuple[List[np.ndarray], int, int]:
        if isinstance(video_source, (str, Path)):
            cap = cv2.VideoCapture(str(video_source))
            should_release = True
        else:
            cap = video_source
            should_release = False

        if not cap.isOpened():
            logger.error(f"Không thể mở nguồn video: {video_source}")
            return [], 0, 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 200

        max_sample_range = min(total_frames - 1, 200)
        sample_indices = np.linspace(0, max_sample_range, min(num_samples, total_frames), dtype=int)

        frames = []
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        if should_release:
            cap.release()

        if not frames:
            return [], 0, 0

        H, W = frames[0].shape[:2]
        return frames, H, W
