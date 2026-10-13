import cv2
import numpy as np
from pathlib import Path


class AutoStopLineDetector:
    """
    Module tu dong phat hien vach dung den do / vach son mat duong (Autonomous Stop Line Detector)
    su dung thi giac may tinh & xu ly anh da khung hinh:
    - Thu thap bo loc da khung hinh (Median Temporal Filter) de loai bo xe dang di chuyen, giu lai mat duong tinh.
    - Trich xuat vung mat duong (Road ROI) & dac trung vach son trang (HSV White Filter + Top-Hat Morphology).
    - Trich xuat cac duong nam ngang (Horizontal Morphological Opening + Hough Line Transform).
    - Gom nhom cum toa do Y va cham diem do tin cay (Length, Span, Continuity Score).
    - Tra ve toa do [(x1, y1), (x2, y2)] thuc te cua video, khong phu thuoc vao bat ky toa do mac dinh nao!
    """

    def __init__(self, num_samples=12, max_lines=2):
        self.num_samples = num_samples
        self.max_lines = max_lines

    def detect_from_video(self, video_source, max_lines=None):
        """
        Nhan vao duong dan file video hoac doi tuong cv2.VideoCapture.
        Tra ve danh sach cac vach dung tu dong duoc phat hien:
        [
            {
                'id': 'line_1',
                'name': 'Vach 1 (Tu dong nhan dien)',
                'line': [(x1, y1), (x2, y2)],
                'direction': 'TOP_TO_BOTTOM' hoac 'BOTTOM_TO_TOP',
                'confidence': float
            }, ...
        ]
        Neu video khong co vach son mat duong, tra ve [] (danh sach rong).
        """
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

        max_sample_range = min(total_frames - 1, 120)
        sample_indices = np.linspace(0, max_sample_range, self.num_samples, dtype=int)

        frames = []
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                
        # Tra lai vi tri frame ban dau
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        if should_release:
            cap.release()

        if not frames:
            return []

        H, W = frames[0].shape[:2]

        # 2. Trich xuat anh nen mat duong bang phep loc Median (loai bo phuong tien dang chay)
        if len(frames) >= 5:
            road_background = np.median(frames[:8], axis=0).astype(np.uint8)
        else:
            road_background = frames[0]

        # 3. Vung quan tam mat duong (Road ROI: tu 35% den 92% chieu cao khung hinh)
        roi_top = int(H * 0.35)
        roi_bottom = int(H * 0.92)
        roi = road_background[roi_top:roi_bottom, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 4. Trich xuat vach son trang:
        # 4.1. Bo loc mau trang HSV (Saturation thap, Brightness cao)
        lower_white = np.array([0, 0, 130])
        upper_white = np.array([180, 70, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # 4.2. Phep bien doi Top-hat tren Grayscale (nhan dien vach son sang tren nen duong toi)
        tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, tophat_kernel)
        _, mask_tophat = cv2.threshold(tophat, 28, 255, cv2.THRESH_BINARY)

        combined_mask = cv2.bitwise_or(mask_white, mask_tophat)

        # 5. Bo loc hinh thai hoc chieu ngang (Horizontal Opening) de loai bo vach doc va hat nhieu
        kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_horiz)

        # 6. Phat hien cac doan thang bang HoughLinesP
        lines = cv2.HoughLinesP(clean_mask, 1, np.pi / 180, threshold=40, minLineLength=35, maxLineGap=25)
        if lines is None:
            return []

        # 7. Loc cac doan thang nam ngang (slope |dx| > 2.0 * |dy|)
        horizontal_segments = []
        for l in lines:
            if len(l) == 4 and not hasattr(l[0], '__len__'):
                x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
            else:
                x1, y1, x2, y2 = int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3])

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx > 2.0 * dy and dx >= 30:
                # Quy doi toa do ve khung hinh goc
                horizontal_segments.append((min(x1, x2), y1 + roi_top, max(x1, x2), y2 + roi_top, dx))

        if not horizontal_segments:
            return []

        # 8. Gom nhom cac doan thang theo truc Y (Clustering)
        clusters = []
        for x1, y1, x2, y2, length in horizontal_segments:
            y_mid = (y1 + y2) / 2
            matched = False
            for c in clusters:
                if abs(c['y_mid'] - y_mid) < 35:
                    c['lines'].append((x1, y1, x2, y2, length))
                    c['y_mid'] = np.mean([(it[1] + it[3]) / 2 for it in c['lines']])
                    matched = True
                    break
            if not matched:
                clusters.append({'y_mid': y_mid, 'lines': [(x1, y1, x2, y2, length)]})

        # 9. Cham diem va tinh toan duong thang khop (Line Fitting & Scoring)
        candidates = []
        for c in clusters:
            min_x = min(it[0] for it in c['lines'])
            max_x = max(it[2] for it in c['lines'])
            span_x = max_x - min_x
            tot_len = sum(it[4] for it in c['lines'])

            # Vach dung thuc te phai trai rong qua cac lan duong (span it nhat 20% chieu rong)
            if span_x < W * 0.20 and tot_len < W * 0.18:
                continue

            # Hoi quy tuyen tinh de tao duong thang lien mach
            pts_x, pts_y = [], []
            for it in c['lines']:
                pts_x.extend([it[0], it[2]])
                pts_y.extend([it[1], it[3]])
            fit = np.polyfit(pts_x, pts_y, 1)
            y_start = int(fit[0] * min_x + fit[1])
            y_end = int(fit[0] * max_x + fit[1])

            # Tinh diem tin cay: Do rong span + Tong chieu dai cac phan tu vach
            score = (span_x / W) * 1.5 + (tot_len / W)
            candidates.append({
                'score': score,
                'y_mid': c['y_mid'],
                'line': [(int(min_x), y_start), (int(max_x), y_end)],
                'span': span_x
            })

        if not candidates:
            return []

        # Sap xep theo do tin cay cao nhat
        candidates.sort(key=lambda item: item['score'], reverse=True)

        # Chon toi da `max_lines` vach co score cao nhat va cach xa nhau it nhat 70px
        selected = []
        for cand in candidates:
            if len(selected) >= max_lines:
                break
            if not selected or all(abs(cand['y_mid'] - s['y_mid']) > 70 for s in selected):
                selected.append(cand)

        # Sap xep lai theo thu tu tu tren xuong duoi (Y tang dan)
        selected.sort(key=lambda item: item['y_mid'])

        # Tao danh sach cau hinh vach dung chuan
        detected_stop_lines = []
        for i, s in enumerate(selected):
            y_avg = s['y_mid']
            # Huong di chuyen: vach nua tren ngã tư thiet lap TOP_TO_BOTTOM, nua duoi BOTTOM_TO_TOP
            direction = 'TOP_TO_BOTTOM' if y_avg < H * 0.55 else 'BOTTOM_TO_TOP'
            name_desc = f"Vach {i+1} (Tu dong phat hien - Y={int(y_avg)})"
            
            detected_stop_lines.append({
                'id': f'auto_line_{i+1}',
                'name': name_desc,
                'line': s['line'],
                'direction': direction,
                'score': float(round(s['score'], 2))
            })

        return detected_stop_lines


# Ham tien ich de goi nhanh
def get_auto_stop_lines(video_path, max_lines=2):
    detector = AutoStopLineDetector(max_lines=max_lines)
    return detector.detect_from_video(video_path)
