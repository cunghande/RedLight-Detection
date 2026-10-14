import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class StoplineSegmentationDataset(Dataset):
    """
    Dataset cho bài toán phân đoạn ảnh vạch kẻ đường:
    Đọc ảnh và nhãn đa giác từ dataset_stopline.
    Trả về ảnh (Tensor C,H,W) và mặt nạ ngữ nghĩa (Tensor H,W) với nhãn [0: nền, 1: stop_line, 2: crosswalk].
    """
    def __init__(self, root_dir: str, split: str = "train", img_size: int = 480):
        self.root = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.img_dir = self.root / "images" / split
        self.lbl_dir = self.root / "labels" / split
        
        self.img_files = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))
        print(f"[{split.upper()}] Tìm thấy {len(self.img_files)} ảnh tại {self.img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {img_path}")
        h_orig, w_orig = img.shape[:2]

        mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])  # 0: stop_line, 1: crosswalk
                    coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
                    coords[:, 0] *= w_orig
                    coords[:, 1] *= h_orig
                    poly = coords.astype(np.int32)
                    # Gán nhãn mặt nạ: 1 = stop_line, 2 = crosswalk
                    cv2.fillPoly(mask, [poly], color=cls_id + 1)

        # Resize về kích thước chuẩn hóa
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        mask_resized = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Chuẩn hóa ảnh [0, 1] và chuyển định dạng CHW
        img_tensor = torch.from_numpy(img_resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        # Chuẩn hóa theo ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        mask_tensor = torch.from_numpy(mask_resized).long()

        return img_tensor, mask_tensor, str(img_path)


class StoplineDetectionDataset(Dataset):
    """
    Dataset cho Mask R-CNN (Two-stage Instance Segmentation):
    Đọc ảnh và nhãn đa giác từ dataset_stopline.
    Trả về image (Tensor [3, H, W]) và dict target chứa boxes, labels, masks.
    """
    def __init__(self, root_dir: str, split: str = "train", img_size: int = 480):
        self.root = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.img_dir = self.root / "images" / split
        self.lbl_dir = self.root / "labels" / split
        self.img_files = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {img_path}")
        h_orig, w_orig = img.shape[:2]

        scale_x = self.img_size / w_orig
        scale_y = self.img_size / h_orig

        boxes = []
        labels = []
        masks = []

        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0]) + 1  # 1: stop_line, 2: crosswalk (0 là background)
                    coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
                    coords[:, 0] *= self.img_size
                    coords[:, 1] *= self.img_size

                    poly = coords.astype(np.int32)
                    x1, y1 = float(np.min(coords[:, 0])), float(np.min(coords[:, 1]))
                    x2, y2 = float(np.max(coords[:, 0])), float(np.max(coords[:, 1]))

                    # Đảm bảo box có kích thước tối thiểu > 2px để Mask R-CNN không lỗi
                    if x2 <= x1 + 2:
                        x2 = x1 + 4
                    if y2 <= y1 + 2:
                        y2 = y1 + 4

                    inst_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                    cv2.fillPoly(inst_mask, [poly], 1)

                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)
                    masks.append(inst_mask)

        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_tensor = torch.from_numpy(img_resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, self.img_size, self.img_size), dtype=torch.uint8)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx])
        }

        return img_tensor, target, str(img_path)


def collate_fn(batch):
    return tuple(zip(*batch))
