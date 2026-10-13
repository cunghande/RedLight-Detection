import cv2
import numpy as np
from pathlib import Path


class AutoStopLineDetector:
    """
    Module tu dong phat hien vach dung den do / vach sang duong tren mat duong (Autonomous Stop Line Detector)
    su dung thi giac may tinh & xu ly anh:
    - Thu thap anh mat duong bang phep loc trung vi da khung hinh (Median Temporal Filter) loai bo xe di chuyen.
    - Xac dinh vung mat duong hoat dong (Active Road Surface ROI, y: 46% -> 82% H) loai bo ho lan, cay coi, watermark.
    - Trich xuat vach son trang bang nguong thich nghi do sang mat duong (Adaptive Road Brightness Percentile).
    - Trich xuat duong nam ngang: Hough Line Transform voi rang buoc chat che (|dy| <= 16px, do doc |dy/dx| < 0.08).
    - Gom nhom cum Y va danh gia do tin cay (Span & Total Paint Density).
    - Mac dinh chi lay DUY NHAT 1 vach dung chinh thuc te. Khong tu y them vach 2 neu video chi co 1 vach!
    - Vach duoc ke nam ngang thang tap theo mat duong, khong lech cheo hay nghieng dốc.
    """

    def __init__(self, num_samples=10, max_lines=1):
        self.num_samples = num_samples
        self.max_lines = max_lines

    def detect_from_video(self, video_source, max_lines=None):
        if max_lines is None:
            max_lines = self.max_lines

        if isinstance(video_source, (str, Path)):
            cap = cv2.VideoCapture(str(video_source))
            should_release = True
        else:
            cap = video_source
            should_release = False

        if not cap.isOpened():
            print(f"[CANH BAO] Khong the mo video de tu dong phat hien vach: {video_source}")
            return []

        # 1. Thu thap cac frame dai dien doc theo video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 100

        max_sample_range = min(total_frames - 1, 100)
        sample_indices = np.linspace(0, max_sample_range, self.num_samples, dtype=int)

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
            return []

        H, W = frames[0].shape[:2]

        # 2. Trich xuat anh nen mat duong bang phep loc Median (loai bo phuong tien dang chay)
        if len(frames) >= 5:
            road_background = np.median(frames[:7], axis=0).astype(np.uint8)
        else:
            road_background = frames[0]

        # 3. Vung mat duong hoat dong: tu 46% den 82% chieu cao khung hinh
        roi_top = int(H * 0.46)
        roi_bottom = int(H * 0.82)
        roi = road_background[roi_top:roi_bottom, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 4. Trich xuat vach son trang bang nguong do sang thich nghi mat duong
        th_val = max(115, int(np.percentile(gray, 85)))
        _, bin_mask = cv2.threshold(gray, th_val, 255, cv2.THRESH_BINARY)

        # 5. Bo loc hinh thai hoc chieu ngang (Horizontal Opening)
        kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        clean_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel_horiz)

        # 6. Phat hien cac doan thang bang HoughLinesP
        lines = cv2.HoughLinesP(clean_mask, 1, np.pi / 180, threshold=35, minLineLength=35, maxLineGap=30)
        if lines is None:
            return []

        # 7. Loc nghiem ngat: Vach dung phai hoan toan nam ngang
        horizontal_segments = []
        for l in lines:
            if len(l) == 4 and not hasattr(l[0], '__len__'):
                x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
            else:
                x1, y1, x2, y2 = int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3])

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            # RANG BUOC CHAT CHE:
            # - dx >= 35 pixel
            # - dy <= 16 pixel (khong lech qua 16 pixel doc)
            # - do doc |dy/dx| < 0.08 (< 4.5 do) -> tuyet doi khong bao gio bi cheo dốc!
            if dx >= 35 and dy <= 16 and (dy / max(1, dx)) < 0.08:
                horizontal_segments.append((min(x1, x2), y1 + roi_top, max(x1, x2), y2 + roi_top, dx))

        if not horizontal_segments:
            return []

        # 8. Gom nhom cac doan thang theo truc Y (Clustering)
        clusters = []
        for x1, y1, x2, y2, dx in horizontal_segments:
            y_mid = (y1 + y2) / 2
            matched = False
            for c in clusters:
                if abs(c['ym'] - y_mid) < 40:
                    c['lines'].append((x1, y1, x2, y2, dx))
                    c['ym'] = np.mean([(it[1] + it[3]) / 2 for it in c['lines']])
                    matched = True
                    break
            if not matched:
                clusters.append({'ym': y_mid, 'lines': [(x1, y1, x2, y2, dx)]})

        # 9. Cham diem tung cum vach
        scored_clusters = []
        for c in clusters:
            min_x = min(it[0] for it in c['lines'])
            max_x = max(it[2] for it in c['lines'])
            span_x = max_x - min_x
            tot_len = sum(it[4] for it in c['lines'])

            # Vach dung thuc te phai trai rong it nhat 22% chieu rong man hinh
            if span_x >= W * 0.22:
                avg_y = int(round(c['ym']))

                # Lam phang vach nam ngang theo mat duong (do lech dy toi da <= 6px)
                slopes = [(it[3] - it[1]) / max(1, it[2] - it[0]) for it in c['lines']]
                med_slope = float(np.median(slopes)) if slopes else 0.0
                med_slope = max(-0.02, min(0.02, med_slope))

                x_mid = (min_x + max_x) / 2
                y1 = int(round(avg_y + med_slope * (min_x - x_mid)))
                y2 = int(round(avg_y + med_slope * (max_x - x_mid)))

                if abs(y2 - y1) > 8:
                    y2 = y1

                score = span_x * 2.0 + tot_len
                scored_clusters.append({
                    'score': score,
                    'ym': avg_y,
                    'span': span_x,
                    'tot_l': tot_len,
                    'line': [(int(min_x), y1), (int(max_x), y2)]
                })

        if not scored_clusters:
            return []

        # Sap xep theo do tin cay (score) giam dan
        scored_clusters.sort(key=lambda x: x['score'], reverse=True)

        # 10. XAC DINH CHINH XAC SO LUONG VACH (Mac dinh 1 vach duy nhat)
        primary = scored_clusters[0]
        final_lines = [primary]

        # Chi them vach 2 neu nguoi dung cho phep (max_lines >= 2) VA thuc su co vach 2
        # o phia doi dien cua nga tu (cach xa it nhat 220px, span >= 35%W, score >= 85% vach 1)
        if max_lines >= 2:
            for cand in scored_clusters[1:]:
                if abs(cand['ym'] - primary['ym']) >= 220 and cand['span'] >= W * 0.35 and cand['score'] >= 0.85 * primary['score']:
                    final_lines.append(cand)
                    break

        final_lines.sort(key=lambda x: x['ym'])

        detected_stop_lines = []
        for i, s in enumerate(final_lines):
            y_avg = s['ym']
            direction = 'TOP_TO_BOTTOM' if y_avg < H * 0.55 else 'BOTTOM_TO_TOP'
            name_desc = "Vach dung" if len(final_lines) == 1 else f"Vach {i+1}"
            
            detected_stop_lines.append({
                'id': f'auto_line_{i+1}',
                'name': f"{name_desc} (Y={int(y_avg)})",
                'line': s['line'],
                'direction': direction,
                'score': float(round(s['score'], 1))
            })

        return detected_stop_lines


def get_auto_stop_lines(video_path, max_lines=1):
    detector = AutoStopLineDetector(max_lines=max_lines)
    return detector.detect_from_video(video_path)
