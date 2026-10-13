import os
from pathlib import Path

import modal

# ======================= CAU HINH (sua o day) =======================
GPU = "H100"  # doi thanh "L40S" / "A10" / "L4" neu muon tiet kiem credit

# Dataset Roboflow Universe. Mo trang dataset -> "Download Dataset" -> chon
# format "YOLOv8" -> "show download code" de lay dung workspace/project/version.
RF_WORKSPACE = "kmutt-eghvb"
RF_PROJECT = "datamodel"
RF_VERSION = 1  # <-- KIEM TRA lai so version tren Roboflow

BASE_MODEL = "yolov8s.pt"  # trong so pretrained, tu tai ve
RUN_NAME = "plate_helmet"
# ====================================================================

app = modal.App("traffic-yolo-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics", "roboflow")
)

# Noi luu ket qua train (best.pt, bieu do...) giua cac lan chay
volume = modal.Volume.from_name("traffic-yolo-vol", create_if_missing=True)

# Tao bang: modal secret create roboflow-secret ROBOFLOW_API_KEY=<key cua ban>
rf_secret = modal.Secret.from_name("roboflow-secret")

DATA_DIR = "/root/dataset"
VOL_DIR = "/vol"


def prepare_dataset() -> str:
    """Tai dataset tu Roboflow va sua data.yaml de dung duong dan tuyet doi."""
    import yaml
    from roboflow import Roboflow

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    ds = (
        rf.workspace(RF_WORKSPACE)
        .project(RF_PROJECT)
        .version(RF_VERSION)
        .download("yolov8", location=DATA_DIR)
    )
    root = Path(ds.location)
    yaml_path = root / "data.yaml"
    cfg = yaml.safe_load(yaml_path.read_text())

    val_dir = next((d for d in ("valid", "val") if (root / d).exists()), None)
    if not (root / "train").exists() or val_dir is None:
        raise RuntimeError(f"Khong thay thu muc train/valid trong {root}: {list(root.iterdir())}")

    cfg["path"] = str(root)
    cfg["train"] = "train/images"
    cfg["val"] = f"{val_dir}/images"
    if (root / "test").exists():
        cfg["test"] = "test/images"
    yaml_path.write_text(yaml.safe_dump(cfg))

    print("Classes:", cfg.get("names"))
    for split in ("train", val_dir, "test"):
        imgs = root / split / "images"
        if imgs.exists():
            print(f"  {split}: {len(list(imgs.iterdir()))} anh")
    return str(yaml_path)


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOL_DIR: volume},
    secrets=[rf_secret],
    timeout=60 * 60 * 3,  # toi da 3 gio
)
def train(epochs: int, imgsz: int, batch: int, model_name: str) -> dict:
    from ultralytics import YOLO

    data_yaml = prepare_dataset()

    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=f"{VOL_DIR}/runs",
        name=RUN_NAME,
        exist_ok=True,  # ghi de neu chay lai cung ten
        patience=20,  # dung som neu 20 epoch khong cai thien
        plots=True,
    )

    metrics = model.val()
    volume.commit()  # luu ket qua vao Volume

    best = f"{VOL_DIR}/runs/{RUN_NAME}/weights/best.pt"
    return {
        "best_weights": best,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "classes": metrics.names,
    }


@app.local_entrypoint()
def main(
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 32,
    model: str = BASE_MODEL,
):
    result = train.remote(epochs, imgsz, batch, model)
    print("\n===== KET QUA =====")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(
        "\nTai trong so ve may:\n"
        f"  modal volume get traffic-yolo-vol /runs/{RUN_NAME}/weights/best.pt ."
    )