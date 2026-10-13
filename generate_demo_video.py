import cv2
import numpy as np
from pathlib import Path

def create_traffic_demo(output_path="demo.mp4", duration_sec=12, fps=25):
    """
    Tao video mo phong nga tu giao thong voi:
    - Duong chay, vach dung, lan duong.
    - Den tin hieu giao thong doi mau (Xanh -> Vang -> Do).
    - Cac phuong tien (O to, xe may) di chuyen; co phuong tien vuot den do.
    """
    width, height = 1280, 720
    total_frames = duration_sec * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Danh sach xe mo phong: [type, x, y, speed, width, height, color, stops_at_red]
    vehicles = [
        {"type": "car", "x": 380, "y": 50, "speed": 8, "w": 90, "h": 160, "color": (50, 50, 220), "stops": False},
        {"type": "motorcycle", "x": 580, "y": 80, "speed": 6, "w": 40, "h": 90, "color": (30, 180, 50), "stops": True},
        {"type": "motorcycle_violator", "x": 750, "y": -50, "speed": 7, "w": 40, "h": 90, "color": (200, 120, 20), "stops": False},
        {"type": "car_2", "x": 920, "y": -180, "speed": 9, "w": 95, "h": 170, "color": (210, 80, 50), "stops": True},
    ]

    stop_line_y = 480

    for f in range(total_frames):
        # 1. Ve mat duong (mau xam toi)
        frame = np.full((height, width, 3), (60, 60, 60), dtype=np.uint8)
        
        # Ve via he hai ben
        cv2.rectangle(frame, (0, 0), (220, height), (100, 120, 100), -1)
        cv2.rectangle(frame, (1080, 0), (width, height), (100, 120, 100), -1)
        
        # Ve vach ke lan duong dut doan
        for lane_x in [480, 700, 900]:
            for dash_y in range(0, height, 60):
                cv2.rectangle(frame, (lane_x - 3, dash_y), (lane_x + 3, dash_y + 35), (255, 255, 255), -1)
                
        # Ve vach di bo qua duong (Crosswalk)
        for cw_x in range(250, 1050, 60):
            cv2.rectangle(frame, (cw_x, stop_line_y + 40), (cw_x + 35, stop_line_y + 110), (220, 220, 220), -1)

        # Ve vach dung xe (Stop Line)
        cv2.rectangle(frame, (230, stop_line_y - 6), (1070, stop_line_y + 6), (255, 255, 255), -1)
        
        # 2. Xac dinh mau den giao thong theo thoi gian
        # 0-4s: Green | 4-6s: Yellow | 6-12s: Red
        t_sec = f / fps
        if t_sec < 4.0:
            light_state = "GREEN"
        elif t_sec < 6.0:
            light_state = "YELLOW"
        else:
            light_state = "RED"

        # 3. Ve hop den giao thong o ben phai
        tl_x, tl_y = 1120, 100
        tl_w, tl_h = 60, 150
        cv2.rectangle(frame, (tl_x, tl_y), (tl_x + tl_w, tl_y + tl_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (tl_x, tl_y), (tl_x + tl_w, tl_y + tl_h), (180, 180, 180), 2)
        
        c_red = (0, 0, 255) if light_state == "RED" else (20, 20, 60)
        c_yellow = (0, 255, 255) if light_state == "YELLOW" else (20, 60, 60)
        c_green = (0, 255, 0) if light_state == "GREEN" else (20, 60, 20)
        
        cv2.circle(frame, (tl_x + 30, tl_y + 28), 16, c_red, -1)
        cv2.circle(frame, (tl_x + 30, tl_y + 75), 16, c_yellow, -1)
        cv2.circle(frame, (tl_x + 30, tl_y + 122), 16, c_green, -1)

        # 4. Cap nhat va ve cac phuong tien
        for v in vehicles:
            # Logic dung truoc vach khi den do
            should_stop = (light_state == "RED" or light_state == "YELLOW") and v["stops"]
            at_stop_line = (v["y"] + v["h"] >= stop_line_y - 25 and v["y"] + v["h"] <= stop_line_y)
            
            if not (should_stop and at_stop_line):
                v["y"] += v["speed"]

            # Ve hinh chu nhat bieu dien xe
            vx, vy, vw, vh = int(v["x"]), int(v["y"]), int(v["w"]), int(v["h"])
            if vy + vh > 0 and vy < height:
                cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), v["color"], -1)
                cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), (255, 255, 255), 2)
                # Kinh xe / chi tiet
                cv2.rectangle(frame, (vx + 6, vy + 20), (vx + vw - 6, vy + 50), (220, 220, 240), -1)
                label = "MOTO" if "motor" in v["type"] else "CAR"
                cv2.putText(frame, label, (vx + 4, vy + vh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        out.write(frame)

    out.release()
    print(f"[THANH CONG] Da tao video demo mo phong: {output_path}")

if __name__ == "__main__":
    create_traffic_demo()
