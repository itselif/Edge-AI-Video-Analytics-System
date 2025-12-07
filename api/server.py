import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from inference.detector import Detector, Detection  # type: ignore
from api.schemas import BBox, DetectResponse, HealthResponse, MetricsResponse


# ================================
# Simple latency / FPS tracker
# ================================
class SimpleMetrics:
    def __init__(self, max_history: int = 1000) -> None:
        self.latencies_ms: List[float] = []
        self.max_history = max_history
        self.total_requests: int = 0
        self.start_time: float = time.time()

    def record_latency(self, ms: float) -> None:
        self.latencies_ms.append(ms)
        if len(self.latencies_ms) > self.max_history:
            self.latencies_ms.pop(0)
        self.total_requests += 1

    def get_stats(self) -> dict:
        if not self.latencies_ms:
            return {
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "fps": 0.0,
                "total_requests": self.total_requests,
            }

        lat_sorted = sorted(self.latencies_ms)
        n = len(lat_sorted)

        def percentile(p: float) -> float:
            if n == 1:
                return lat_sorted[0]
            k = int(round((p / 100.0) * (n - 1)))
            return lat_sorted[k]

        avg = mean(lat_sorted)
        p50 = percentile(50)
        p95 = percentile(95)

        elapsed = max(time.time() - self.start_time, 1e-6)
        fps = len(self.latencies_ms) / elapsed

        return {
            "avg_latency_ms": float(avg),
            "p50_latency_ms": float(p50),
            "p95_latency_ms": float(p95),
            "fps": float(fps),
            "total_requests": self.total_requests,
        }


# ================================
# GPU info (if NVML is available)
# ================================
def get_gpu_stats() -> dict:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        used_mb = mem.used / (1024 * 1024)
        total_mb = mem.total / (1024 * 1024)

        return {
            "gpu_name": name,
            "gpu_memory_used_mb": float(used_mb),
            "gpu_memory_total_mb": float(total_mb),
            "gpu_utilization": float(util.gpu),
        }
    except Exception:
        return {
            "gpu_name": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_utilization": None,
        }


# ================================
# FastAPI app & global objects
# ================================
app = FastAPI(title="Edge AI Video Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_BACKEND = os.getenv("DETECTOR_BACKEND", "onnx")  # "onnx" or "torch"
MODEL_DIR = ROOT / "models"
DEFAULT_MODEL_ONNX = MODEL_DIR / "model.onnx"
DEFAULT_MODEL_PT = MODEL_DIR / "latest.pt"

DEFAULT_ONNX_PROVIDERS = os.getenv(
    "ONNX_PROVIDERS",
    "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider",
).split(",")

COCO_5CLS = ["person", "bicycle", "car", "motorcycle", "bus"]

DETECTOR: Optional[Detector] = None
METRICS = SimpleMetrics()


@app.on_event("startup")
def load_model_on_startup() -> None:
    global DETECTOR

    backend = DEFAULT_BACKEND
    model_path = DEFAULT_MODEL_PT if backend == "torch" else DEFAULT_MODEL_ONNX

    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")

    if backend == "onnx":
        DETECTOR = Detector(
            backend="onnx",
            model_path=model_path,
            imgsz=640,
            conf_thres=0.25,
            iou_thres=0.45,
            device="cpu",
            onnx_providers=[p.strip() for p in DEFAULT_ONNX_PROVIDERS],
        )
    elif backend == "torch":
        DETECTOR = Detector(
            backend="torch",
            model_path=model_path,
            imgsz=640,
            conf_thres=0.25,
            iou_thres=0.45,
            device="cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu",
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    print(f"[API] Loaded detector backend={backend}, model={model_path}")


def detections_to_bboxes(dets: List[Detection]) -> List[BBox]:
    bboxes: List[BBox] = []

    for d in dets:
        cls_id = int(d.cls)
        label = COCO_5CLS[cls_id] if 0 <= cls_id < len(COCO_5CLS) else None

        bboxes.append(
            BBox(
                x1=float(d.x1),
                y1=float(d.y1),
                x2=float(d.x2),
                y2=float(d.y2),
                score=float(d.score),
                cls_id=cls_id,
                label=label,
            )
        )

    return bboxes


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    backend = DEFAULT_BACKEND
    model_path = str(DEFAULT_MODEL_PT if backend == "torch" else DEFAULT_MODEL_ONNX)

    status = "ok" if DETECTOR is not None else "error"
    detail = "model loaded" if DETECTOR is not None else "model not loaded"

    return HealthResponse(
        status=status,
        backend=backend,
        model_path=model_path,
        detail=detail,
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    if DETECTOR is None:
        raise RuntimeError("Detector is not initialized.")

    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode input image.")

    t0 = time.time()
    dets = DETECTOR(img)  # type: ignore
    t1 = time.time()

    if isinstance(dets, list) and dets and isinstance(dets[0], Detection):
        det_list: List[Detection] = dets  # type: ignore
    else:
        det_list = dets[0]  # type: ignore

    latency_ms = (t1 - t0) * 1000.0
    METRICS.record_latency(latency_ms)

    bboxes = detections_to_bboxes(det_list)

    backend = DEFAULT_BACKEND
    model_path = str(DEFAULT_MODEL_PT if backend == "torch" else DEFAULT_MODEL_ONNX)

    return DetectResponse(
        backend=backend,
        model_path=model_path,
        inference_time_ms=float(latency_ms),
        num_detections=len(bboxes),
        detections=bboxes,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    backend = DEFAULT_BACKEND
    model_path = str(DEFAULT_MODEL_PT if backend == "torch" else DEFAULT_MODEL_ONNX)

    stats = METRICS.get_stats()
    gpu_stats = get_gpu_stats()

    return MetricsResponse(
        backend=backend,
        model_path=model_path,
        avg_latency_ms=stats["avg_latency_ms"],
        p50_latency_ms=stats["p50_latency_ms"],
        p95_latency_ms=stats["p95_latency_ms"],
        fps=stats["fps"],
        total_requests=stats["total_requests"],
        gpu_name=gpu_stats["gpu_name"],
        gpu_memory_used_mb=gpu_stats["gpu_memory_used_mb"],
        gpu_memory_total_mb=gpu_stats["gpu_memory_total_mb"],
        gpu_utilization=gpu_stats["gpu_utilization"],
    )
