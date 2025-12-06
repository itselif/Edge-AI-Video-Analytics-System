import argparse
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO


def parse_args():
    root = Path(__file__).resolve().parents[1]
    default_model = root / "models" / "latest.pt"
    default_onnx = root / "models" / "model.onnx"

    parser = argparse.ArgumentParser(description="Export YOLO11 model to ONNX with dynamic shapes.")
    parser.add_argument("--model-path", type=str, default=str(default_model), help="Path to trained .pt file")
    parser.add_argument("--onnx-path", type=str, default=str(default_onnx), help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size (square)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version (>=12)")
    parser.add_argument("--dynamic", action="store_true", default=True, help="Enable dynamic axes")
    parser.add_argument("--validate", action="store_true", help="Validate ONNX vs PyTorch outputs")
    return parser.parse_args()


def export_to_onnx(args):
    model_path = Path(args.model_path)
    onnx_path = Path(args.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading YOLO model from {model_path} on {device} ...")
    yolo = YOLO(str(model_path))

    print("[INFO] Exporting to ONNX...")
    exported_path = yolo.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,   # dynamic batch + hw
        simplify=True,
        half=False,
        device=device,
    )

    exported_path = Path(exported_path)
    if exported_path.resolve() != onnx_path.resolve():
        print(f"[INFO] Moving exported ONNX from {exported_path} to {onnx_path}")
        onnx_path.write_bytes(exported_path.read_bytes())

    print(f"[INFO] ONNX model saved to {onnx_path}")
    return yolo, onnx_path


def validate_onnx(onnx_path: Path, yolo_model: YOLO, imgsz: int = 640, num_trials: int = 3):
    """
    Numeric sanity check: run random tensor through PyTorch vs ONNXRuntime and
    compare max/mean absolute difference.
    """
    print("[INFO] Validating ONNX vs PyTorch outputs...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model.model.to(device).eval()

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[INFO] ONNX model is structurally valid.")

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name

    max_diffs = []
    mean_diffs = []

    for i in range(num_trials):
        dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
        with torch.no_grad():
            pt_out = yolo_model.model(dummy)
        if isinstance(pt_out, (list, tuple)):
            pt_out = pt_out[0]
        pt_out = pt_out.detach().cpu().numpy()

        ort_out = sess.run(None, {input_name: dummy.cpu().numpy()})[0]

        if pt_out.shape != ort_out.shape:
            print(f"[WARN] Shape mismatch: torch {pt_out.shape} vs onnx {ort_out.shape}")
            # still try to compare overlapping dims
            common_shape = tuple(min(a, b) for a, b in zip(pt_out.shape, ort_out.shape))
            pt_view = pt_out.reshape(-1)[: np.prod(common_shape)]
            ort_view = ort_out.reshape(-1)[: np.prod(common_shape)]
        else:
            pt_view = pt_out.reshape(-1)
            ort_view = ort_out.reshape(-1)

        diff = np.abs(pt_view - ort_view)
        max_diffs.append(diff.max())
        mean_diffs.append(diff.mean())
        print(f"[INFO] Trial {i+1}: max diff={diff.max():.6f}, mean diff={diff.mean():.6f}")

    print(
        f"[INFO] ONNX validation summary: "
        f"max diff={max(max_diffs):.6e}, mean diff={np.mean(mean_diffs):.6e}"
    )
    print("[INFO] If max diff is ~1e-3 or lower, export is numerically consistent.")


def main():
    args = parse_args()
    yolo, onnx_path = export_to_onnx(args)
    if args.validate:
        validate_onnx(onnx_path, yolo_model=yolo, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
