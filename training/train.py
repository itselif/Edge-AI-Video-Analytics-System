# training/train.py

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from pathlib import Path
import shutil
import argparse

from training.augmentations import get_train_augmentations  # <- Albumentations pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / "training" / "dataset.yaml"
LOG_DIR = ROOT / "training" / "logs"
MODELS_DIR = ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")  # Colab GPU: "0"
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    return parser.parse_args()


def main():
    args = parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    # Albumentations augmentations
    custom_augs = get_train_augmentations(img_size=args.imgsz)

    run_name = "coco_5cls_yolo11n"

    LOGGER.info(f"Using dataset config: {DATA_CONFIG}")
    LOGGER.info(f"Logs dir         : {LOG_DIR}")
    LOGGER.info(f"Models dir       : {MODELS_DIR}")

    #Train the model
    results = model.train(
        data=str(DATA_CONFIG),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,

        project=str(LOG_DIR),
        name=run_name,
        exist_ok=True,


        multi_scale=True,  
        cos_lr=True,        
        amp=True,           
        ema=True,          

        # YOLO built-in augmentations
        mosaic=1.0,         
        mixup=0.2,          
        close_mosaic=10,    # Close the mosaic last 10 epoch
        augmentations=custom_augs,

        # Log & plotting
        val=True,
        plots=True,         # Loss curves, mAP curves, confusion matrix etc.
        save=True,
    )

    # Copy the best model as models/latest.pt
    run_dir = LOG_DIR / run_name
    best_ckpt = run_dir / "weights" / "best.pt"
    last_ckpt = run_dir / "weights" / "last.pt"

    src = best_ckpt if best_ckpt.exists() else last_ckpt
    dst = MODELS_DIR / "latest.pt"

    if src.exists():
        shutil.copy2(src, dst)
        LOGGER.info(f"[INFO] Saved best model to {dst}")
    else:
        LOGGER.warning("[WARN] No best/last checkpoint found!")


if __name__ == "__main__":
    main()
