from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def export_to_onnx(
    pt_path: Path,
    onnx_path: Path,
    imgsz: int = 640,
    opset: int = 13,
    device: str = "cuda:0",
) -> None:
    model = YOLO(str(pt_path))

    export_args = dict(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=True,
        simplify=True,
        device=device,
        half=False,
        verbose=False,
    )

    if str(device).startswith("cuda"):
        export_args["optimize"] = False
    else:
        export_args["optimize"] = True

    onnx_file = model.export(**export_args)
    onnx_file = Path(onnx_file)

    if onnx_file != onnx_path:
        onnx_path.write_bytes(onnx_file.read_bytes())
        onnx_file.unlink()

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    print(f"[OK] Exported ONNX model to {onnx_path}")


def validate_onnx_vs_pytorch(
    pt_path: Path,
    onnx_path: Path,
    imgsz: int = 640,
    device: str = "cuda:0",
) -> None:
    device_torch = torch.device(device if torch.cuda.is_available() else "cpu")

    model = YOLO(str(pt_path))
    model.to(device_torch)
    model.eval()

    x = torch.rand(1, 3, imgsz, imgsz, device=device_torch)

    with torch.no_grad():
        pt_raw = model.model(x)[0].detach().cpu().numpy()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    ort_outs = sess.run(None, {input_name: x.detach().cpu().numpy()})
    onnx_raw = ort_outs[0]

    if pt_raw.shape != onnx_raw.shape:
        print(f"[WARN] Shape mismatch: torch {pt_raw.shape} vs onnx {onnx_raw.shape}")
    else:
        diff = np.abs(pt_raw - onnx_raw).mean()
        max_diff = np.abs(pt_raw - onnx_raw).max()
        print(f"[INFO] ONNX vs PyTorch mean diff: {diff:.6f}, max diff: {max_diff:.6f}")

    print("[OK] ONNX validation finished.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, default=str(MODELS_DIR / "latest.pt"))
    ap.add_argument("--onnx", type=str, default=str(MODELS_DIR / "model.onnx"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--device", type=str, default="cuda:0")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pt_path = Path(args.pt)
    onnx_path = Path(args.onnx)

    export_to_onnx(
        pt_path=pt_path,
        onnx_path=onnx_path,
        imgsz=args.imgsz,
        opset=args.opset,
        device=args.device,
    )

    validate_onnx_vs_pytorch(
        pt_path=pt_path,
        onnx_path=onnx_path,
        imgsz=args.imgsz,
        device=args.device,
    )
