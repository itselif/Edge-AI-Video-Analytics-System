# api/server.py
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference.detector import Detector, Detection
from monitoring.logger import LatencyMeter, JsonLogger, get_gpu_stats
from api.schemas import (
    HealthResponse,
    DetectResponse,
    BBox,
    MetricsResponse,
)

# Sadece 5 sınıf: COCO_5CLS
COCO_5CLS = ["person", "bicycle", "car", "motorcycle", "bus"]

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

DEFAULT_BACKEND = "onnx"
DEFAULT_MODEL_ONNX = MODELS_DIR / "model.onnx"

app = FastAPI(title="Edge AI Video Analytics API")

# CORS (frontend vs. bağlanmak isterse rahat olsun diye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monitoring objeleri (global)
METRICS = LatencyMeter(window_size=100)
EVENT_LOGGER = JsonLogger(str(LOG_DIR / "api_events.jsonl"))

# Global detector
detector: Detector | None = None


@app.on_event("startup")
def startup_event() -> None:
    """
    Uygulama ayağa kalkarken ONNX modelini yükler.
    """
    global detector
    model_path = DEFAULT_MODEL_ONNX

    detector = Detector(
        backend=DEFAULT_BACKEND,
        model_path=model_path,
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        device="cuda",  # CUDA yoksa ORT zaten CPU'ya düşecek
        class_names=COCO_5CLS,  # 👈 LABEL İÇİN ÖNEMLİ
    )

    print(f"[API] Loaded detector backend={DEFAULT_BACKEND}, model={model_path}")


@app.on_event("shutdown")
def shutdown_event() -> None:
    """
    Uygulama kapanırken log dosyasını kapat.
    """
    EVENT_LOGGER.close()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Basit health-check endpoint'i.
    """
    backend = DEFAULT_BACKEND
    model_path = str(DEFAULT_MODEL_ONNX)
    return HealthResponse(
        status="ok",
        backend=backend,
        model_path=model_path,
        detail="Service healthy",
    )


def _detections_to_bboxes(det_list: List[Detection]) -> List[BBox]:
    """
    Detection -> BBox (API schema) dönüşümü.
    Burada cls_id ve (varsa) label alanını dolduruyoruz.
    """
    boxes: List[BBox] = []

    for d in det_list:
        cls_id = int(d.cls)
        label = None

        # Güvenli şekilde class name çek
        if detector is not None and getattr(detector, "class_names", None) is not None:
            if 0 <= cls_id < len(detector.class_names):
                label = detector.class_names[cls_id]

        boxes.append(
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

    return boxes


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    """
    Görsel dosya alır → modelle inference yapar → bbox + skor + latency döner.
    """
    assert detector is not None, "Detector is not initialized"

    # Dosyayı oku → PIL → numpy BGR
    raw_bytes = await file.read()
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    frame = np.array(img)[:, :, ::-1]  # RGB -> BGR

    t0 = time.time()
    det_list: List[Detection] = detector(frame)
    t1 = time.time()

    latency_ms = (t1 - t0) * 1000.0
    METRICS.record_latency(latency_ms)

    bboxes = _detections_to_bboxes(det_list)

    # JSON logging (monitoring/dashboard için)
    EVENT_LOGGER.log(
        "detect",
        {
            "latency_ms": float(latency_ms),
            "num_detections": len(bboxes),
        },
    )

    return DetectResponse(
        backend=DEFAULT_BACKEND,
        model_path=str(DEFAULT_MODEL_ONNX),
        inference_time_ms=float(latency_ms),
        num_detections=len(bboxes),
        detections=bboxes,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """
    Latency istatistikleri + GPU bilgisi dönen endpoint.
    """
    stats = METRICS.get_stats()
    gpu = get_gpu_stats()

    return MetricsResponse(
        backend=DEFAULT_BACKEND,
        model_path=str(DEFAULT_MODEL_ONNX),
        avg_latency_ms=stats["avg_latency_ms"],
        moving_avg_latency_ms=stats["moving_avg_latency_ms"],
        p50_latency_ms=stats["p50_latency_ms"],
        p90_latency_ms=stats["p90_latency_ms"],
        p95_latency_ms=stats["p95_latency_ms"],
        fps=stats["fps"],
        total_requests=stats["total_requests"],
        gpu_name=gpu["gpu_name"],
        gpu_memory_used_mb=gpu["gpu_memory_used_mb"],
        gpu_memory_total_mb=gpu["gpu_memory_total_mb"],
        gpu_utilization=gpu["gpu_utilization"],
    )
