import argparse
import json
import time
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pynvml
import torch
from ultralytics import YOLO

# TensorRT / PyCUDA opsiyonel: yoksa sadece PyTorch + ONNX benchmark çalışsın
try:
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda

    HAS_TRT = True
except ImportError:
    trt = None
    cuda = None
    HAS_TRT = False
    print("[WARN] TensorRT or PyCUDA not available, TensorRT benchmarks will be skipped.")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_IMAGES_DIR = ROOT / "datasets" / "coco_5cls" / "images" / "val"
RESULTS_PATH = ROOT / "optimization" / "benchmark_results.json"


def load_images(image_dir: Path, limit: int = 100):
    paths = sorted(
        glob(str(image_dir / "*.jpg"))
        + glob(str(image_dir / "*.jpeg"))
        + glob(str(image_dir / "*.png"))
    )
    paths = paths[:limit]
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        imgs.append((p, img))
    if not imgs:
        raise RuntimeError(f"No images found in {image_dir}")
    return imgs


def preprocess_image(img, imgsz: int):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img


class TrtRunner:
    """Simple wrapper around a TensorRT engine for benchmarking."""

    def __init__(self, engine_path: Path):
        if not HAS_TRT:
            raise RuntimeError("TensorRT is not available in this environment.")

        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_idx = None
        self.output_idx = None
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_idx = i
            else:
                self.output_idx = i

        if self.input_idx is None or self.output_idx is None:
            raise RuntimeError("Failed to find input/output bindings in TensorRT engine.")

    def run(self, img_tensor: np.ndarray):
        # img_tensor: (1, 3, H, W)
        batch, c, h, w = img_tensor.shape
        self.context.set_binding_shape(self.input_idx, (batch, c, h, w))

        inp_shape = self.context.get_binding_shape(self.input_idx)
        out_shape = self.context.get_binding_shape(self.output_idx)

        inp_size = int(np.prod(inp_shape))
        out_size = int(np.prod(out_shape))

        inp_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_idx))
        out_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_idx))

        d_input = cuda.mem_alloc(inp_size * np.dtype(inp_dtype).itemsize)
        d_output = cuda.mem_alloc(out_size * np.dtype(out_dtype).itemsize)

        cuda.memcpy_htod_async(d_input, img_tensor.astype(inp_dtype).ravel(), self.stream)

        bindings = [None] * self.engine.num_bindings
        bindings[self.input_idx] = int(d_input)
        bindings[self.output_idx] = int(d_output)

        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        out_host = np.empty(out_size, dtype=out_dtype)
        cuda.memcpy_dtoh_async(out_host, d_output, self.stream)
        self.stream.synchronize()

        d_input.free()
        d_output.free()

        return out_host.reshape(out_shape)


def init_gpu_monitor():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return handle


def get_gpu_utilization(handle):
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return util.gpu  # percentage


def summarize(name: str, times_ms: list):
    arr = np.array(times_ms)
    return {
        "name": name,
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def benchmark_pytorch(pt_path: Path, imgs, imgsz: int, iters: int, warmup: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(pt_path))
    model.to(device)
    model.eval()

    gpu_handle = init_gpu_monitor()

    pre_times, infer_times, post_times, gpu_utils = [], [], [], []

    # warmup
    for _ in range(warmup):
        _, img = imgs[_ % len(imgs)]
        x = preprocess_image(img, imgsz)
        x = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(x)

    # benchmark
    for i in range(iters):
        _, img = imgs[i % len(imgs)]

        t0 = time.perf_counter()
        x = preprocess_image(img, imgsz)
        pre_t = (time.perf_counter() - t0) * 1000

        x_t = torch.from_numpy(x).unsqueeze(0).to(device)

        t1 = time.perf_counter()
        with torch.no_grad():
            _ = model(x_t)
        inf_t = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        _ = x_t.detach().cpu()
        post_t = (time.perf_counter() - t2) * 1000

        pre_times.append(pre_t)
        infer_times.append(inf_t)
        post_times.append(post_t)
        gpu_utils.append(get_gpu_utilization(gpu_handle))

    total_images = iters
    total_time_s = (sum(pre_times) + sum(infer_times) + sum(post_times)) / 1000.0
    fps = total_images / total_time_s

    return {
        "backend": "pytorch",
        "latency_ms": {
            "pre": summarize("pre", pre_times),
            "infer": summarize("infer", infer_times),
            "post": summarize("post", post_times),
        },
        "throughput_fps": fps,
        "gpu_util_avg": float(np.mean(gpu_utils)),
    }


def benchmark_onnx(onnx_path: Path, imgs, imgsz: int, iters: int, warmup: int):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name

    gpu_handle = init_gpu_monitor()

    pre_times, infer_times, post_times, gpu_utils = [], [], [], []

    # warmup
    for _ in range(warmup):
        _, img = imgs[_ % len(imgs)]
        x = preprocess_image(img, imgsz)
        x = x[None, ...]
        _ = sess.run(None, {input_name: x})

    for i in range(iters):
        _, img = imgs[i % len(imgs)]

        t0 = time.perf_counter()
        x = preprocess_image(img, imgsz)
        x = x[None, ...]
        pre_t = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        out = sess.run(None, {input_name: x})
        inf_t = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        _ = [o for o in out]
        post_t = (time.perf_counter() - t2) * 1000

        pre_times.append(pre_t)
        infer_times.append(inf_t)
        post_times.append(post_t)
        gpu_utils.append(get_gpu_utilization(gpu_handle))

    total_images = iters
    total_time_s = (sum(pre_times) + sum(infer_times) + sum(post_times)) / 1000.0
    fps = total_images / total_time_s

    return {
        "backend": "onnxruntime",
        "latency_ms": {
            "pre": summarize("pre", pre_times),
            "infer": summarize("infer", infer_times),
            "post": summarize("post", post_times),
        },
        "throughput_fps": fps,
        "gpu_util_avg": float(np.mean(gpu_utils)),
    }


def benchmark_trt(engine_path: Path, imgs, imgsz: int, iters: int, warmup: int, name: str):
    if not HAS_TRT:
        raise RuntimeError("TensorRT is not available, cannot benchmark TensorRT backend.")

    runner = TrtRunner(engine_path)
    gpu_handle = init_gpu_monitor()

    pre_times, infer_times, post_times, gpu_utils = [], [], [], []

    # warmup
    for _ in range(warmup):
        _, img = imgs[_ % len(imgs)]
        x = preprocess_image(img, imgsz)
        x = x[None, ...]
        _ = runner.run(x)

    for i in range(iters):
        _, img = imgs[i % len(imgs)]

        t0 = time.perf_counter()
        x = preprocess_image(img, imgsz)
        x = x[None, ...]
        pre_t = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        out = runner.run(x)
        inf_t = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        _ = out  # dummy postprocess
        post_t = (time.perf_counter() - t2) * 1000

        pre_times.append(pre_t)
        infer_times.append(inf_t)
        post_times.append(post_t)
        gpu_utils.append(get_gpu_utilization(gpu_handle))

    total_images = iters
    total_time_s = (sum(pre_times) + sum(infer_times) + sum(post_times)) / 1000.0
    fps = total_images / total_time_s

    return {
        "backend": name,
        "latency_ms": {
            "pre": summarize("pre", pre_times),
            "infer": summarize("infer", infer_times),
            "post": summarize("post", post_times),
        },
        "throughput_fps": fps,
        "gpu_util_avg": float(np.mean(gpu_utils)),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--images_dir", type=str, default=str(DATA_IMAGES_DIR))
    ap.add_argument("--pt", type=str, default=str(MODELS_DIR / "latest.pt"))
    ap.add_argument("--onnx", type=str, default=str(MODELS_DIR / "model.onnx"))
    ap.add_argument("--fp16_engine", type=str, default=str(MODELS_DIR / "model_fp16.engine"))
    ap.add_argument("--int8_engine", type=str, default=str(MODELS_DIR / "model_int8.engine"))
    ap.add_argument("--out", type=str, default=str(RESULTS_PATH))
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    imgs = load_images(Path(args.images_dir), limit=200)

    results = []

    # PyTorch backend
    results.append(
        benchmark_pytorch(
            Path(args.pt), imgs, imgsz=args.imgsz, iters=args.iters, warmup=args.warmup
        )
    )

    # ONNX Runtime backend
    results.append(
        benchmark_onnx(
            Path(args.onnx), imgs, imgsz=args.imgsz, iters=args.iters, warmup=args.warmup
        )
    )

    # TensorRT FP16 backend (opsiyonel)
    fp16_path = Path(args.fp16_engine)
    if fp16_path.exists() and HAS_TRT:
        try:
            results.append(
                benchmark_trt(
                    fp16_path, imgs, imgsz=args.imgsz, iters=args.iters, warmup=args.warmup, name="tensorrt_fp16"
                )
            )
        except Exception as e:
            print(f"[WARN] Skipping TensorRT FP16 benchmark due to error: {e}")
    else:
        print(f"[WARN] FP16 engine not found at {fp16_path} or TensorRT unavailable, skipping TensorRT FP16 benchmark")

    # TensorRT INT8 backend (opsiyonel)
    int8_path = Path(args.int8_engine)
    if int8_path.exists() and HAS_TRT:
        try:
            results.append(
                benchmark_trt(
                    int8_path, imgs, imgsz=args.imgsz, iters=args.iters, warmup=args.warmup, name="tensorrt_int8"
                )
            )
        except Exception as e:
            print(f"[WARN] Skipping TensorRT INT8 benchmark due to error: {e}")
    else:
        print(f"[WARN] INT8 engine not found at {int8_path} or TensorRT unavailable, skipping TensorRT INT8 benchmark")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Benchmark results saved to {args.out}")
