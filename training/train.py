# training/train.py

import argparse
from pathlib import Path
import shutil

import torch
from ultralytics import YOLO

from augmentations import get_train_augmentations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    # Proje kökü: .../Edge-AI-Video-Analytics-System
    root = Path(__file__).resolve().parents[1]

    data_path = root / "training" / "dataset.yaml"
    logs_dir = root / "training" / "logs"
    models_dir = root / "models"

    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using dataset config: {data_path}")
    print(f"Logs dir         : {logs_dir}")
    print(f"Models dir       : {models_dir}")

    # Device seçimi: GPU varsa 0, yoksa CPU
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # YOLO11n modelini yükle
    model = YOLO("yolo11n.pt")  # weights otomatik indirilecek

    # Albumentations pipeline
    custom_augs = get_train_augmentations(img_size=args.imgsz)

    # Eğitim
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(logs_dir),
        name="coco_5cls_yolo11n",
        exist_ok=True,
        multi_scale=True,   # Multi-scale training
        cos_lr=True,        # Cosine LR schedule
        amp=True,           # Mixed precision
        mosaic=1.0,         # Mosaic açık
        mixup=0.2,          # MixUp açık
        close_mosaic=10,    # Son 10 epoch'ta mosaic kapanıyor
        augmentations=custom_augs,  # Albumentations pipeline
        val=True,
        plots=True,         # loss/mAP grafiklerini üret
        save=True,          # weights/best.pt ve last.pt
        device=device,      # <--- KRİTİK KISIM
    )

    # En iyi modeli models/latest.pt olarak kopyala
    run_dir = Path(results.save_dir)  # training/logs/coco_5cls_yolo11n
    best_pt = run_dir / "weights" / "best.pt"
    latest_pt = models_dir / "latest.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, latest_pt)
        print(f"[INFO] Saved best model to {latest_pt}")
    else:
        print("[WARN] best.pt not found, latest.pt not updated")


if __name__ == "__main__":
    main()
