from pathlib import Path
import yaml

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent
YAML_PATH = BASE_DIR / "config.yaml"

def _load_yaml():
    if YAML_PATH.exists():
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

_CFG = _load_yaml()

# Cấu hình Model mặc định
MODEL_PATH = _CFG.get("models", {}).get("vehicle_detector", {}).get("weights", "yolov8s.pt")
CONF_THRESHOLD = _CFG.get("models", {}).get("vehicle_detector", {}).get("conf", 0.35)
TRACK_CLASSES = _CFG.get("models", {}).get("vehicle_detector", {}).get("track_classes", [0, 1, 2, 3, 5, 7])

CLASS_NAMES_VI = {
    0: "Nguoi di bo",
    1: "Xe dap",
    2: "O to",
    3: "Xe may",
    5: "Xe buyt",
    7: "Xe tai"
}

OUTPUT_DIR = BASE_DIR / _CFG.get("general", {}).get("output_dir", "violations")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Tương thích ngược get_stop_lines
def get_stop_lines(source_path: str, auto_detect: bool = True):
    from src.stopline_detector import StopLineDetector
    detector = StopLineDetector(config_path=YAML_PATH)
    cam_id = Path(str(source_path)).stem
    return detector.detect(source_path, camera_id=cam_id, mode="classical")

# Đèn tín hiệu
TRAFFIC_LIGHT_ROI = (1120, 100, 60, 150)
TRAFFIC_LIGHT_MODE = "TIMER"
TIMER_CYCLE = {"RED": 8, "GREEN": 8, "YELLOW": 3}

PENALTY_REGULATIONS = _CFG.get("penalties", {})
