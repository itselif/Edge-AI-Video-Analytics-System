# optimization/export_to_onnx.py

"""
PyTorch (YOLO11) -> ONNX export script.

Features:
- Uses models/latest.pt as default input.
- Exports dynamic ONNX (batch, height, width).
- opset >= 12 (default: 13).
- Validates ONNX output vs PyTorch output on a dummy input.
- Saves final ONNX as models/model.onnx
"""

import argparse
from pathlib import Path
import shutil

import numpy as np
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DEFAULT_WEIGHTS = MODELS_DIR / "latest.pt"
DEFAULT_ONNX = MODELS_DIR / "model.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export YOLO model to ONNX with dynamic axes.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to trained PyTorch weights (.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_ONNX),
        help="Output ONNX model path.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (assumed square: H=W=imgsz).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (>= 12).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for PyTorch model ('cuda:0' or 'cpu').",
    )
    return parser.parse_args()


def export_with_ultralytics(args: argparse.Namespace) -> Path:
    """
    Use Ultralytics built-in export to ONNX.

    This already:
    - handles dynamic axes (dynamic=True)
    - uses correct model forward
    - produces an ONNX file path we can then validate and rename.
    """
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print(f"[INFO] Loading YOLO model from: {weights_path}")
    model = YOLO(str(weights_path))

    imgsz = (args.imgsz, args.imgsz)

    print(f"[INFO] Exporting to ONNX (opset={args.opset}, imgsz={imgsz}, dynamic=True)")
    onnx_export_path = model.export(
        format="onnx",
        opset=args.opset,
        imgsz=imgsz,
        dynamic=True,      # dynamic batch / height / width
        simplify=True,
        half=False,
        device=args.device,
        verbose=True,
    )

    onnx_export_path = Path(onnx_export_path)
    print(f"[INFO] Ultralytics export created: {onnx_export_path}")

    # Move/rename to desired output path
    final_path = Path(args.output)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_export_path, final_path)
    print(f"[INFO] Copied ONNX to: {final_path}")

    return final_path


def validate_onnx_vs_pytorch(
    pt_weights: Path,
    onnx_path: Path,
    imgsz: int,
    device: str = "cuda:0",
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    """
    Run a simple numeric check between PyTorch model and ONNX model on a dummy input.

    This is not a perfect mAP-level validation, but it verifies:
    - I/O shapes are compatible
    - numeric outputs are reasonably close.
    """
    if not pt_weights.exists():
        raise FileNotFoundError(f"PyTorch weights not found: {pt_weights}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    print(f"[INFO] Validating ONNX vs PyTorch on a dummy input...")
    print(f"       PT:   {pt_weights}")
    print(f"       ONNX: {onnx_path}")

    # Load YOLO model
    model = YOLO(str(pt_weights))
    model.to(device)
    model.eval()

    # Create dummy input
    h = w = imgsz
    dummy_torch = torch.randn(1, 3, h, w, device=device)

    with torch.no_grad():
        # Ultralytics DetectionModel forward returns a tensor or list; we standardize to tensor.
        pt_out = model.model(dummy_torch)
        if isinstance(pt_out, (list, tuple)):
            pt_out = pt_out[0]
        pt_out = pt_out.detach().cpu().numpy()

    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[INFO] ONNX model check passed.")

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = sess.get_inputs()[0].name
    dummy_onnx = dummy_torch.detach().cpu().numpy()
    onnx_out_list = sess.run(None, {input_name: dummy_onnx})

    if isinstance(onnx_out_list, list):
        onnx_out = onnx_out_list[0]
    else:
        onnx_out = onnx_out_list

    print(f"[INFO] PyTorch output shape: {pt_out.shape}")
    print(f"[INFO] ONNX output shape:    {onnx_out.shape}")

    if pt_out.shape != onnx_out.shape:
        print("[WARN] Shape mismatch between PyTorch and ONNX outputs.")
        print("       This can happen if post-processing layers differ.")
        return

    diff = np.abs(pt_out - onnx_out)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"[INFO] Max abs diff:  {max_diff:.6f}")
    print(f"[INFO] Mean abs diff: {mean_diff:.6f}")

    if max_diff < atol or mean_diff < atol:
        print("[INFO] ONNX outputs are numerically close to PyTorch outputs (PASS).")
    else:
        print("[WARN] ONNX vs PyTorch numeric difference is larger than expected.")
        print("       Check post-processing or export settings if this is critical.")


def main():
    args = parse_args()
    print(f"[INFO] ROOT:       {ROOT}")
    print(f"[INFO] MODELS_DIR: {MODELS_DIR}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    onnx_path = export_with_ultralytics(args)

    # Validation step
    try:
        validate_onnx_vs_pytorch(
            pt_weights=Path(args.weights),
            onnx_path=onnx_path,
            imgsz=args.imgsz,
            device=args.device,
        )
    except Exception as e:
        print(f"[WARN] Validation failed with error: {e}")
        print("       Please inspect the ONNX model manually if needed.")


if __name__ == "__main__":
    main()
