export type BBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  cls_id: number;
  label?: string | null;
};

export type DetectResponse = {
  backend: string;
  model_path: string;
  inference_time_ms: number;
  num_detections: number;
  detections: BBox[];
};

export type MetricsResponse = {
  backend: string;
  model_path: string;
  avg_latency_ms: number;
  moving_avg_latency_ms: number;
  p50_latency_ms: number;
  p90_latency_ms: number;
  p95_latency_ms: number;
  fps: number;
  total_requests: number;
  gpu_name?: string | null;
  gpu_memory_used_mb?: number | null;
  gpu_memory_total_mb?: number | null;
  gpu_utilization?: number | null;
};

export type DetectionMode = "image" | "video" | "live";
