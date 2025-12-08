# tests/test_onnx_shapes.py

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pytest

from inference.detector import Detector, Detection

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ONNX_PATH = MODELS_DIR / "model.onnx"
FP16_ENGINE = MODELS_DIR / "model_fp16.engine"
INT8_ENGINE = MODELS_DIR / "model_int8.engine"


@pytest.mark.skipif(not ONNX_PATH.exists(), reason="ONNX model not found")
def test_onnx_single_image_dynamic_shapes_and_bounds() -> None:
    """
    ONNX backend farklı giriş boyutlarında sorunsuz çalışıyor mu
    ve bbox koordinatları görüntü sınırları içinde mi?
    Bu, dynamic height/width + pre/post-process consistency testidir.
    """
    det = Detector(
        backend="onnx",
        model_path=ONNX_PATH,
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        device="cpu",
    )

    # Farklı HxW kombinasyonları ile dene
    sizes = [(320, 480), (480, 800), (640, 640)]
    for h, w in sizes:
        img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        dets = det(img)

        assert isinstance(dets, list)
        for d in dets:
            assert isinstance(d, Detection)
            # bbox koordinatları görüntü sınırlarını aşmamalı
            assert 0.0 <= d.x1 <= w
            assert 0.0 <= d.y1 <= h
            assert 0.0 <= d.x2 <= w
            assert 0.0 <= d.y2 <= h
            # class id makul aralıkta
            assert 0 <= d.cls <= 4


@pytest.mark.skipif(not ONNX_PATH.exists(), reason="ONNX model not found")
def test_onnx_batch_inference_io_shape() -> None:
    """
    ONNX backend batch input (liste halinde frame) aldığında
    çıktı formatı [List[Detection], ...] şeklinde mi?
    Bu, dynamic batch size + I/O shape validation testidir.
    """
    det = Detector(
        backend="onnx",
        model_path=ONNX_PATH,
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        device="cpu",
    )

    h, w = 480, 640
    imgs: Sequence[np.ndarray] = [
        (np.random.rand(h, w, 3) * 255).astype(np.uint8) for _ in range(3)
    ]

    outputs = det(imgs)
    assert isinstance(outputs, list)
    assert len(outputs) == len(imgs)

    for per_img in outputs:
        assert isinstance(per_img, list)
        for d in per_img:
            assert isinstance(d, Detection)


@pytest.mark.skipif(
    not FP16_ENGINE.exists(), reason="TensorRT FP16 engine file not found"
)
def test_tensorrt_fp16_engine_loadable() -> None:
    """
    TensorRT FP16 engine deserialize edilebiliyor mu?
    TensorRT kütüphanesi veya engine dosyası yoksa test skip edilir.
    """
    try:
        import tensorrt as trt  # type: ignore
    except Exception:
        pytest.skip("TensorRT not installed in this environment")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    engine_path = FP16_ENGINE
    with engine_path.open("rb") as f:
        engine_bytes = f.read()

    engine = runtime.deserialize_cuda_engine(engine_bytes)
    assert engine is not None


@pytest.mark.skipif(
    not INT8_ENGINE.exists(), reason="TensorRT INT8 engine file not found"
)
def test_tensorrt_int8_engine_loadable() -> None:
    """
    INT8 TensorRT engine deserialize testi.
    Yine TensorRT ya da dosya yoksa otomatik skip.
    """
    try:
        import tensorrt as trt  # type: ignore
    except Exception:
        pytest.skip("TensorRT not installed in this environment")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    engine_path = INT8_ENGINE
    with engine_path.open("rb") as f:
        engine_bytes = f.read()

    engine = runtime.deserialize_cuda_engine(engine_bytes)
    assert engine is not None
