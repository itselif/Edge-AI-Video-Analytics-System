from typing import List, Optional
from pydantic import BaseModel


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls_id: int
    label: Optional[str] = None


class DetectResponse(BaseModel):
    backend: str
    model_path: str
    inference_time_ms: float
    num_detections: int
    detections: List[BBox]


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_path: str
    detail: str


class MetricsResponse(BaseModel):
    backend: str
    model_path: str

    avg_latency_ms: float
    moving_avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    fps: float
    total_requests: int

    gpu_name: Optional[str] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
