from pathlib import Path

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent

# Cấu hình Model mặc định
MODEL_PATH = "yolov8s.pt"

# Các class COCO cần theo dõi: 
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
TRACK_CLASSES = [0, 1, 2, 3, 5, 7]

CLASS_NAMES_VI = {
    0: "Nguoi di bo",
    1: "Xe dap",
    2: "O to",
    3: "Xe may",
    5: "Xe buyt",
    7: "Xe tai"
}

CONF_THRESHOLD = 0.35

# ================= CẤU HÌNH VẠCH DỪNG (STOP LINE) =================
# Tọa độ mặc định: Đặt ở đầu lối vào giao lộ (khoảng y=360), xe vừa chớm vượt là bắt ngay,
# KHÔNG bị trôi ra giữa ngã tư mới phát hiện!
DEFAULT_STOP_LINE = [(60, 360), (620, 360)]

# Cấu hình vạch dừng tối ưu riêng cho từng video mẫu nếu có:
VIDEO_STOP_LINES = {
    "red1.mp4": [(60, 360), (620, 360)],
    "red2.mp4": [(60, 360), (620, 360)],
    "demo.mp4": [(230, 480), (1070, 480)],
}

def get_stop_line(source_path: str):
    """Tu dong lay vach dung phu hop voi video hoac dung vach mac dinh"""
    name = Path(str(source_path)).name
    return VIDEO_STOP_LINES.get(name, DEFAULT_STOP_LINE)

# Hướng di chuyển vượt đèn đỏ:
# "TOP_TO_BOTTOM": xe đi từ trên xuống dưới
# "BOTTOM_TO_TOP": xe đi từ dưới lên trên
MOVE_DIRECTION = "TOP_TO_BOTTOM"

# ================= CẤU HÌNH ĐÈN TÍN HIỆU =================
TRAFFIC_LIGHT_ROI = (1120, 100, 60, 150)
TRAFFIC_LIGHT_MODE = "TIMER"

TIMER_CYCLE = {
    "RED": 8,     # Đèn đỏ trong 8 giây
    "GREEN": 8,   # Đèn xanh trong 8 giây
    "YELLOW": 3   # Đèn vàng trong 3 giây
}

# ================= MỨC PHẠT NGUỘI THAM CHIẾU (Nghị định 100 & 123) =================
PENALTY_REGULATIONS = {
    "O to": {
        "fine_range": "4.000.000 - 6.000.000 VNĐ",
        "suspension": "Tước quyền sử dụng GPLX từ 01 đến 03 tháng",
        "legal_code": "Điểm a Khoản 5 Điều 5 NĐ 100/2019/NĐ-CP (sửa đổi NĐ 123/2021)"
    },
    "Xe may": {
        "fine_range": "800.000 - 1.000.000 VNĐ",
        "suspension": "Tước quyền sử dụng GPLX từ 01 đến 03 tháng",
        "legal_code": "Điểm e Khoản 4 Điều 6 NĐ 100/2019/NĐ-CP"
    },
    "Xe tai": {
        "fine_range": "4.000.000 - 6.000.000 VNĐ",
        "suspension": "Tước quyền sử dụng GPLX từ 01 đến 03 tháng",
        "legal_code": "Điểm a Khoản 5 Điều 5 NĐ 100/2019/NĐ-CP"
    },
    "Xe buyt": {
        "fine_range": "4.000.000 - 6.000.000 VNĐ",
        "suspension": "Tước quyền sử dụng GPLX từ 01 đến 03 tháng",
        "legal_code": "Điểm a Khoản 5 Điều 5 NĐ 100/2019/NĐ-CP"
    },
    "Xe dap": {
        "fine_range": "100.000 - 200.000 VNĐ",
        "suspension": "Không áp dụng",
        "legal_code": "Điểm đ Khoản 2 Điều 8 NĐ 100/2019/NĐ-CP"
    },
    "Nguoi di bo": {
        "fine_range": "60.000 - 100.000 VNĐ",
        "suspension": "Không áp dụng",
        "legal_code": "Điểm a Khoản 1 Điều 9 NĐ 100/2019/NĐ-CP"
    }
}

# Thư mục lưu bằng chứng và báo cáo
OUTPUT_DIR = BASE_DIR / "violations"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
