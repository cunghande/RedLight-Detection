import os
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDHead
import numpy as np

# ==============================================================================
# SCRIPT HUAN LUYEN / FINE-TUNE CHO CA 3 TRUONG PHAI:
# 1. Faster R-CNN (ResNet-50)
# 2. SSD-MobileNet
# 3. YOLOv8s
# ==============================================================================

class DummyTrafficDataset(Dataset):
    """
    Dataset mau tao du lieu mo phong / loader chuan format PyTorch Detection
    Format: image tensor (3, H, W), targets: {'boxes': [N, 4], 'labels': [N]}
    """
    def __init__(self, num_samples=50):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Image ngau nhien (3, 640, 640)
        img = torch.rand((3, 640, 640), dtype=torch.float32)
        # Bounding boxes gia dinh cho cac phuong tien (x1, y1, x2, y2)
        boxes = torch.tensor([
            [100.0, 150.0, 250.0, 400.0],
            [300.0, 200.0, 420.0, 380.0]
        ], dtype=torch.float32)
        # Labels: 1 (O to), 2 (Xe may)
        labels = torch.tensor([1, 2], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def train_faster_rcnn(epochs=3, batch_size=2, lr=0.005):
    print("\n" + "="*60)
    print("BAT DAU HUAN LUYEN: Faster R-CNN (ResNet-50-FPN)")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Thiet bi huan luyen: {device}")

    # 1. Khoi tao model voi pretrained backbone
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Thay the Box Predictor cho so class bai toan (vi du: 4 classes: background, car, moto, person)
    num_classes = 4
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # 2. DataLoader & Optimizer
    dataset = DummyTrafficDataset(num_samples=20)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 3. Training Loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        lr_scheduler.step()
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Luu checkpoint
    save_path = "faster_rcnn_custom.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[HOAN TAT] Da luu trong so Faster R-CNN vao: {save_path}\n")

def train_ssd(epochs=3, batch_size=2, lr=0.001):
    print("\n" + "="*60)
    print("BAT DAU HUAN LUYEN: SSDLite320 (MobileNetV3)")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Thiet bi huan luyen: {device}")

    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
    model.to(device)

    dataset = DummyTrafficDataset(num_samples=20)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    save_path = "ssd_mobilenet_custom.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[HOAN TAT] Da luu trong so SSD-MobileNet vao: {save_path}\n")

def train_yolo(epochs=10, model_name="yolov8s.pt"):
    print("\n" + "="*60)
    print(f"BAT DAU HUAN LUYEN: YOLO ({model_name})")
    print("="*60)
    from ultralytics import YOLO
    model = YOLO(model_name)
    # Su dung ham train cua ultralytics
    print(f"[INFO] Huong dan train YOLO qua file Train.py hoac lenh CLI:")
    print(f"  yolo train model={model_name} data=data.yaml epochs={epochs} imgsz=640")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huan luyen Faster R-CNN, SSD hoac YOLO")
    parser.add_argument("--model", type=str, default="faster_rcnn", choices=["faster_rcnn", "ssd", "yolo", "all"])
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    if args.model == "faster_rcnn" or args.model == "all":
        train_faster_rcnn(epochs=args.epochs)
    if args.model == "ssd" or args.model == "all":
        train_ssd(epochs=args.epochs)
    if args.model == "yolo" or args.model == "all":
        train_yolo(epochs=args.epochs)
