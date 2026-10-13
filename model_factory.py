import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

TORCHVISION_COCO_CLASSES = {
    1: "Nguoi",
    2: "Xe dap",
    3: "O to",
    4: "Xe may",
    6: "Xe buyt",
    8: "Xe tai",
    10: "Den tin hieu"
}

YOLO_COCO_CLASSES = {
    0: "Nguoi",
    1: "Xe dap",
    2: "O to",
    3: "Xe may",
    5: "Xe buyt",
    7: "Xe tai",
    9: "Den tin hieu"
}

class BaseDetector:
    def detect(self, frame_bgr: np.ndarray, conf_thresh: float = 0.4):
        raise NotImplementedError

class YOLOv8Detector(BaseDetector):
    def __init__(self, model_path="yolov8s.pt"):
        self.model = YOLO(model_path)
        self.name = f"YOLO ({model_path})"

    def detect(self, frame_bgr: np.ndarray, conf_thresh: float = 0.4):
        results = self.model(
            frame_bgr,
            classes=list(YOLO_COCO_CLASSES.keys()),
            conf=conf_thresh,
            verbose=False
        )
        if not results or len(results) == 0 or results[0].boxes is None:
            return np.empty((0, 4)), np.empty((0,)), []
            
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().numpy()
        class_names = [YOLO_COCO_CLASSES.get(c, "Phuong tien") for c in classes]
        return boxes, scores, class_names

class FasterRCNNDetector(BaseDetector):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Khoi tao Faster R-CNN (ResNet-50-FPN) tren {self.device}...")
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        self.name = "Faster R-CNN (ResNet-50)"

    def detect(self, frame_bgr: np.ndarray, conf_thresh: float = 0.4):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb).to(self.device).unsqueeze(0)

        with torch.no_grad():
            preds = self.model(tensor)[0]

        boxes = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()

        keep = []
        class_names = []
        for i, (score, label) in enumerate(zip(scores, labels)):
            if score >= conf_thresh and label in TORCHVISION_COCO_CLASSES:
                keep.append(i)
                class_names.append(TORCHVISION_COCO_CLASSES[label])

        if not keep:
            return np.empty((0, 4)), np.empty((0,)), []

        return boxes[keep], scores[keep], class_names

class SSDMobileNetDetector(BaseDetector):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Khoi tao SSD-MobileNet tren {self.device}...")
        weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        self.name = "SSD-MobileNet"

    def detect(self, frame_bgr: np.ndarray, conf_thresh: float = 0.4):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb).to(self.device).unsqueeze(0)

        with torch.no_grad():
            preds = self.model(tensor)[0]

        boxes = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()

        keep = []
        class_names = []
        for i, (score, label) in enumerate(zip(scores, labels)):
            if score >= conf_thresh and label in TORCHVISION_COCO_CLASSES:
                keep.append(i)
                class_names.append(TORCHVISION_COCO_CLASSES[label])

        if not keep:
            return np.empty((0, 4)), np.empty((0,)), []

        return boxes[keep], scores[keep], class_names

def get_detector(model_type: str = "yolov8s") -> BaseDetector:
    m = model_type.lower().strip()
    if "faster" in m or "rcnn" in m:
        return FasterRCNNDetector()
    elif "ssd" in m or "mobile" in m:
        return SSDMobileNetDetector()
    elif "yolov8n" in m:
        return YOLOv8Detector("yolov8n.pt")
    elif "yolo11" in m:
        return YOLOv8Detector("yolo11s.pt")
    elif "best" in m:
        return YOLOv8Detector("best.pt")
    else:
        return YOLOv8Detector("yolov8s.pt")
