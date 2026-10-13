import cv2
import numpy as np
import time

class TrafficLightDetector:
    def __init__(self, mode="TIMER", roi=None, timer_cycle=None):
        """
        mode: "COLOR_ROI" hoac "TIMER"
        roi: (x, y, w, h) toa do hop den giao thong
        timer_cycle: dict {"RED": sec, "GREEN": sec, "YELLOW": sec}
        """
        self.mode = mode
        self.roi = roi
        self.timer_cycle = timer_cycle or {"RED": 8, "GREEN": 8, "YELLOW": 3}
        self.current_state = "RED"
        self.state_start_time = time.time()
        
        # Mau BGR tuong ung de ve giao dien
        self.STATE_COLORS = {
            "RED": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "GREEN": (0, 255, 0)
        }

    def update(self, frame=None) -> str:
        """Cap nhat va tra ve trang thai den: 'RED', 'YELLOW', 'GREEN'"""
        if self.mode == "TIMER":
            return self._update_by_timer()
        elif self.mode == "COLOR_ROI" and frame is not None and self.roi is not None:
            return self._detect_from_roi(frame)
        return self.current_state

    def _update_by_timer(self) -> str:
        elapsed = time.time() - self.state_start_time
        target_duration = self.timer_cycle.get(self.current_state, 8)
        
        if elapsed >= target_duration:
            if self.current_state == "RED":
                self.current_state = "GREEN"
            elif self.current_state == "GREEN":
                self.current_state = "YELLOW"
            elif self.current_state == "YELLOW":
                self.current_state = "RED"
            self.state_start_time = time.time()
            
        return self.current_state

    def _detect_from_roi(self, frame) -> str:
        x, y, w, h = self.roi
        h_frame, w_frame = frame.shape[:2]
        
        # Kiem tra toa do hop le
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_frame, x + w)
        y2 = min(h_frame, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return self.current_state
            
        roi_img = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        
        # Nguong mau do trong HSV (mau do nam o 2 dai gia tri Hue)
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 70])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Nguong mau xanh la
        lower_green = np.array([40, 70, 70])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Nguong mau vang
        lower_yellow = np.array([15, 70, 70])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        red_count = cv2.countNonZero(mask_red)
        green_count = cv2.countNonZero(mask_green)
        yellow_count = cv2.countNonZero(mask_yellow)
        
        counts = {"RED": red_count, "GREEN": green_count, "YELLOW": yellow_count}
        max_state = max(counts, key=counts.get)
        
        # Nguong toi thieu de xac nhan den sang
        if counts[max_state] > 30:
            self.current_state = max_state
            
        return self.current_state

    def toggle_state(self):
        """Cho phep chuyen trang thai thu cong (phim tat)"""
        states = ["RED", "YELLOW", "GREEN"]
        idx = (states.index(self.current_state) + 1) % len(states)
        self.current_state = states[idx]
        self.state_start_time = time.time()
        return self.current_state

    def draw(self, frame):
        """Ve hien thi den tin hieu len frame"""
        # Hien thi hop den o goc tren ben phai
        box_x, box_y = 30, 30
        box_w, box_h = 170, 60
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (200, 200, 200), 2)
        
        # Ve 3 bong den (Do, Vang, Xanh)
        r = 16
        center_red = (box_x + 30, box_y + 30)
        center_yellow = (box_x + 75, box_y + 30)
        center_green = (box_x + 120, box_y + 30)
        
        # Den tat: mau xam toi; Den bat: mau ruc ro
        c_red = (0, 0, 255) if self.current_state == "RED" else (30, 30, 80)
        c_yellow = (0, 255, 255) if self.current_state == "YELLOW" else (30, 80, 80)
        c_green = (0, 255, 0) if self.current_state == "GREEN" else (30, 80, 30)
        
        cv2.circle(frame, center_red, r, c_red, -1)
        cv2.circle(frame, center_yellow, r, c_yellow, -1)
        cv2.circle(frame, center_green, r, c_green, -1)
        
        cv2.circle(frame, center_red, r, (220, 220, 220), 1)
        cv2.circle(frame, center_yellow, r, (220, 220, 220), 1)
        cv2.circle(frame, center_green, r, (220, 220, 220), 1)
        
        # Neu co ROI tren frame that, ve khung vien ROI
        if self.roi and self.mode == "COLOR_ROI":
            x, y, w, h = self.roi
            color = self.STATE_COLORS.get(self.current_state, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"TL: {self.current_state}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
