import argparse
import json
import os
import time
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
)

# Optional TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa

    TRT_AVAILABLE = True
except Exception:
    TRT_AVAILABLE = False


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Model benchmarking script.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["onnx", "trt-fp16", "trt-int8"],
        default="onnx",
        help="Which backend to benchmark.",
    )
    parser.add_argument("--onnx-path", type=str, default=str(root / "models" / "model.onnx"))
    parser.add_argument("--trt-fp16-engine", type=str, default=str(root / "models" / "model_fp16.engine"))
    parser.add_argument("--trt-int8-engine", type=str, default=str(root / "models" / "model_int8.engine"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dataset-dir", type=str, default=str(root / "datasets" / "coco_5cls" / "images" / "test"))
    parser.add_argument("--output-json", type=str, default=str(root / "models" / "benchmark_results.json"))
    return parser.parse_args()


def list_images(dataset_dir, max_images=100):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(dataset_dir, e)))
    files = sorted(files)
    if not files:
        raise RuntimeError(f"No images found in {dataset_dir}")
    return files[:max_images]


def preprocess_image(path, imgsz):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img


def start_nvml():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    return handle


def sample_gpu_metrics(handle):
    util = nvmlDeviceGetUtilizationRates(handle)
    mem = nvmlDeviceGetMemoryInfo(handle)
    return util.gpu, mem.used / (1024**2)


def benchmark_onnx(args, handle):
    print("[INFO] Benchmarking ONNXRuntime backend...")
    session = ort.InferenceSession(
        args.onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    image_paths = list_images(args.dataset_dir, max_images=args.iterations + args.warmup)
    imgs = [preprocess_image(p, args.imgsz) for p in image_paths]
    imgs = np.stack(imgs, axis=0)  # [N, C, H, W]

    lat_infer = []
    lat_pre = []
    lat_post = []
    gpu_utils = []
    gpu_mems = []

    N = min(args.iterations + args.warmup, imgs.shape[0])
    batch_size = args.batch_size
    num_batches = N // batch_size

    for b in range(num_batches):
        batch = imgs[b * batch_size : (b + 1) * batch_size]

        # Pre
        t0 = time.time()
        batch_input = batch  # pre-process already done above
        t1 = time.time()

        # Inference
        t2 = time.time()
        _outputs = session.run(None, {input_name: batch_input})
        t3 = time.time()

        # Post (dummy post-processing: max probs)
        t4 = time.time()
        out = _outputs[0]
        _ = out.max(axis=-1)  # fake "NMS" work to simulate post cost
        t5 = time.time()

        pre_ms = (t1 - t0) * 1000
        infer_ms = (t3 - t2) * 1000
        post_ms = (t5 - t4) * 1000

        gpu_u, gpu_mem = sample_gpu_metrics(handle)

        if b >= args.warmup:
            lat_pre.append(pre_ms)
            lat_infer.append(infer_ms)
            lat_post.append(post_ms)
            gpu_utils.append(gpu_u)
            gpu_mems.append(gpu_mem)

    return summarize_results(
        backend="onnx",
        precision="fp32",
        batch_size=batch_size,
        imgsz=args.imgsz,
        lat_pre=lat_pre,
        lat_infer=lat_infer,
        lat_post=lat_post,
        gpu_utils=gpu_utils,
        gpu_mems=gpu_mems,
    )


class TrtEngineRunner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_binding_idx = None
        self.output_binding_idx = None
        for idx, name in enumerate(self.engine):
            if self.engine.get_binding_dtype(name) == trt.float32:
                if self.engine.binding_is_input(name):
                    self.input_binding_idx = idx
                else:
                    self.output_binding_idx = idx

        if self.input_binding_idx is None or self.output_binding_idx is None:
            raise RuntimeError("Could not infer input/output bindings.")

        self.stream = cuda.Stream()

    def infer(self, inputs: np.ndarray):
        # inputs: [N, C, H, W]
        n, c, h, w = inputs.shape
        self.context.set_binding_shape(self.input_binding_idx, (n, c, h, w))

        d_input = cuda.mem_alloc(inputs.nbytes)
        out_shape = self.context.get_binding_shape(self.output_binding_idx)
        out_size = int(np.prod(out_shape))
        d_output = cuda.mem_alloc(out_size * np.float32().nbytes)

        bindings = [None] * self.engine.num_bindings
        bindings[self.input_binding_idx] = int(d_input)
        bindings[self.output_binding_idx] = int(d_output)

        cuda.memcpy_htod_async(d_input, inputs, self.stream)
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
        output = np.empty(out_size, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, d_output, self.stream)
        self.stream.synchronize()
        output = output.reshape(out_shape)
        d_input.free()
        d_output.free()
        return output


def benchmark_trt(args, handle, engine_path, precision_label):
    print(f"[INFO] Benchmarking TensorRT backend: {precision_label} ...")
    runner = TrtEngineRunner(engine_path)

    image_paths = list_images(args.dataset_dir, max_images=args.iterations + args.warmup)
    imgs = [preprocess_image(p, args.imgsz) for p in image_paths]
    imgs = np.stack(imgs, axis=0)

    lat_infer = []
    lat_pre = []
    lat_post = []
    gpu_utils = []
    gpu_mems = []

    N = min(args.iterations + args.warmup, imgs.shape[0])
    batch_size = args.batch_size
    num_batches = N // batch_size

    for b in range(num_batches):
        batch = imgs[b * batch_size : (b + 1) * batch_size]

        # Pre
        t0 = time.time()
        batch_input = batch  # pre-done
        t1 = time.time()

        # Inference
        t2 = time.time()
        outputs = runner.infer(batch_input)
        t3 = time.time()

        # Post (dummy reduction)
        t4 = time.time()
        _ = outputs.max(axis=-1)
        t5 = time.time()

        pre_ms = (t1 - t0) * 1000
        infer_ms = (t3 - t2) * 1000
        post_ms = (t5 - t4) * 1000

        gpu_u, gpu_mem = sample_gpu_metrics(handle)

        if b >= args.warmup:
            lat_pre.append(pre_ms)
            lat_infer.append(infer_ms)
            lat_post.append(post_ms)
            gpu_utils.append(gpu_u)
            gpu_mems.append(gpu_mem)

    return summarize_results(
        backend="tensorrt",
        precision=precision_label,
        batch_size=batch_size,
        imgsz=args.imgsz,
        lat_pre=lat_pre,
        lat_infer=lat_infer,
        lat_post=lat_post,
        gpu_utils=gpu_utils,
        gpu_mems=gpu_mems,
    )


def summarize_results(
    backend,
    precision,
    batch_size,
    imgsz,
    lat_pre,
    lat_infer,
    lat_post,
    gpu_utils,
    gpu_mems,
):
    def stats(xs):
        xs = np.array(xs, dtype=np.float32)
        return {
            "avg": float(xs.mean()),
            "p50": float(np.percentile(xs, 50)),
            "p95": float(np.percentile(xs, 95)),
        }

    lat_total = np.array(lat_pre) + np.array(lat_infer) + np.array(lat_post)
    num_samples = len(lat_total)
    throughput_fps = (num_samples * batch_size) / (lat_total.sum() / 1000.0)

    gpu_utils = np.array(gpu_utils)
    gpu_mems = np.array(gpu_mems)

    return {
        "backend": backend,
        "precision": precision,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "num_samples": int(num_samples * batch_size),
        "latency_ms": {
            "pre": stats(lat_pre),
            "infer": stats(lat_infer),
            "post": stats(lat_post),
            "total": stats(lat_total),
        },
        "throughput_fps": float(throughput_fps),
        "gpu": {
            "utilization_percent": {
                "avg": float(gpu_utils.mean()) if len(gpu_utils) else 0.0,
                "max": float(gpu_utils.max()) if len(gpu_utils) else 0.0,
            },
            "memory_used_mb": {
                "avg": float(gpu_mems.mean()) if len(gpu_mems) else 0.0,
                "max": float(gpu_mems.max()) if len(gpu_mems) else 0.0,
            },
        },
        "cpu": {
            "pre_ms_avg": float(np.mean(lat_pre)) if len(lat_pre) else 0.0,
            "post_ms_avg": float(np.mean(lat_post)) if len(lat_post) else 0.0,
        },
    }


def main():
    args = parse_args()
    handle = start_nvml()

    results = []

    if args.backend == "onnx":
        results.append(benchmark_onnx(args, handle))
    elif args.backend == "trt-fp16":
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available in this environment.")
        results.append(benchmark_trt(args, handle, args.trt_fp16_engine, precision_label="fp16"))
    elif args.backend == "trt-int8":
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available in this environment.")
        results.append(benchmark_trt(args, handle, args.trt_int8_engine, precision_label="int8"))

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[INFO] Benchmark results saved to {out_path}")


if __name__ == "__main__":
    main()
