from pathlib import Path

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent

# Cấu hình Model YOLO
# Dùng yolov8n.pt hoặc yolov8s.pt để nhận diện đa dạng phương tiện (xe máy, ô tô, xe buýt, xe tải, người...)
# Hoặc đổi thành best.pt nếu muốn dùng model riêng
MODEL_PATH = "yolov8s.pt"

# Các class COCO cần theo dõi: 
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
TRACK_CLASSES = [0, 1, 2, 3, 5, 7]

CLASS_NAMES_VI = {
    0: "Nguoi",
    1: "Xe dap",
    2: "O to",
    3: "Xe may",
    5: "Xe buyt",
    7: "Xe tai"
}

CONF_THRESHOLD = 0.4

# ================= CẤU HÌNH VẠCH DỪNG (STOP LINE) =================
# Tọa độ 2 điểm tạo thành đoạn thẳng vạch dừng: [(x1, y1), (x2, y2)]
# Phương tiện di chuyển từ trên xuống dưới (y nhỏ -> y lớn) hoặc ngược lại
STOP_LINE = [(150, 480), (1130, 480)]

# Hướng di chuyển vượt đèn đỏ:
# "TOP_TO_BOTTOM": xe đi từ trên xuống dưới (vượt vạch khi y_new > y_line)
# "BOTTOM_TO_TOP": xe đi từ dưới lên trên (vượt vạch khi y_new < y_line)
MOVE_DIRECTION = "TOP_TO_BOTTOM"

# ================= CẤU HÌNH ĐÈN TÍN HIỆU (TRAFFIC LIGHT) =================
# Vùng ROI của đèn tín hiệu trên khung hình: (x, y, w, h)
# Nếu None hoặc False, có thể dùng chế độ mô phỏng chu kỳ (TIMER)
TRAFFIC_LIGHT_ROI = (1120, 100, 60, 150)  # (x, y, width, height)

# Chế độ nhận diện đèn:
# "COLOR_ROI": Nhận diện màu RED/YELLOW/GREEN trong vùng TRAFFIC_LIGHT_ROI
# "TIMER": Tự động đổi màu đèn theo chu kỳ thời gian (giây) phục vụ kiểm thử
TRAFFIC_LIGHT_MODE = "TIMER"

# Chu kỳ đèn khi dùng chế độ "TIMER" (tính bằng giây)
TIMER_CYCLE = {
    "RED": 8,     # Đèn đỏ trong 8 giây
    "GREEN": 8,   # Đèn xanh trong 8 giây
    "YELLOW": 3   # Đèn vàng trong 3 giây
}

# ================= LƯU TRỮ BẰNG CHỨNG =================
OUTPUT_DIR = BASE_DIR / "violations"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
