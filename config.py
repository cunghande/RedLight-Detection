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

# ================= CẤU HÌNH ĐA VẠCH DỪNG (MULTIPLE STOP LINES) =================
# Cấu hình chuẩn xác 100% khớp với vạch sơn trắng thực tế trên mặt đường:
# - Vạch 1: Làn phía trên (đầu ngã tư phía xa, xe đi từ trên xuống)
# - Vạch 2: Làn đi thẳng phía dưới (chạy chéo theo đúng vạch sơn trắng thực tế từ x=350 sang mép phải x=1050)

DEFAULT_STOP_LINES = [
    {
        "id": "line_1",
        "name": "Vach 1 (Lan xa - Huong tren xuong)",
        "line": [(20, 440), (580, 440)],
        "direction": "TOP_TO_BOTTOM"
    },
    {
        "id": "line_2",
        "name": "Vach 2 (Lan gan - Huong duoi len)",
        "line": [(350, 640), (1050, 675)],
        "direction": "BOTTOM_TO_TOP"
    }
]

# Cấu hình vạch dừng theo từng video:
VIDEO_STOP_LINES_MAP = {
    "red1.mp4": [
        {
            "id": "line_1",
            "name": "Vach 1 (Lan xa - Huong tren xuong)",
            "line": [(20, 440), (580, 440)],
            "direction": "TOP_TO_BOTTOM"
        },
        {
            "id": "line_2",
            "name": "Vach 2 (Lan gan - Huong duoi len)",
            "line": [(350, 640), (1050, 675)],
            "direction": "BOTTOM_TO_TOP"
        }
    ],
    "red2.mp4": [
        {
            "id": "line_1",
            "name": "Vach 1 (Lan xa)",
            "line": [(20, 440), (580, 440)],
            "direction": "TOP_TO_BOTTOM"
        },
        {
            "id": "line_2",
            "name": "Vach 2 (Lan gan)",
            "line": [(350, 640), (1050, 675)],
            "direction": "BOTTOM_TO_TOP"
        }
    ],
    "demo.mp4": [
        {
            "id": "line_1",
            "name": "Vach dung nga tu",
            "line": [(230, 480), (1070, 480)],
            "direction": "TOP_TO_BOTTOM"
        }
    ]
}

def get_stop_lines(source_path: str):
    """Tu dong lay danh sach tat ca cac vach dung cho video"""
    name = Path(str(source_path)).name
    return VIDEO_STOP_LINES_MAP.get(name, DEFAULT_STOP_LINES)

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
