# optimization/build_trt_engine.py

import argparse
from pathlib import Path
import os

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

from calibration_int8_common import EntropyCalibrator  # we'll inline instead of separate module


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_logger():
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    return logger


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = False,
    int8: bool = False,
    input_shape=(1, 3, 640, 640),
    calib_cache: Path = MODELS_DIR / "calibration.cache",
):
    logger = get_logger()
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    input_name = network.get_input(0).name
    _, c, h, w = input_shape

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (1, c, 320, 320),
        (1, c, h, w),
        (4, c, 1280, 1280),
    )
    config.add_optimization_profile(profile)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        if not builder.platform_has_fast_int8:
            raise RuntimeError("INT8 not supported on this platform")
        config.set_flag(trt.BuilderFlag.INT8)

        # Use existing calibration cache (created by calibrate_int8.py)
        class CacheCalibrator(EntropyCalibrator):
            def __init__(self, cache_file: Path):
                super().__init__(image_paths=[], cache_file=cache_file, input_shape=input_shape, use_cache_only=True)

        calibrator = CacheCalibrator(calib_cache)
        config.int8_calibrator = calibrator

    print(f"[INFO] Building TensorRT engine -> {engine_path.name} (fp16={fp16}, int8={int8})")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    print(f"[OK] Saved TensorRT engine to {engine_path}")


# To avoid importing from another file, re-use EntropyCalibrator definition here (same as in calibrate_int8.py)
class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        image_paths,
        input_shape=(1, 3, 640, 640),
        cache_file: Path = MODELS_DIR / "calibration.cache",
        use_cache_only: bool = False,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.batch_size = input_shape[0]
        self.c, self.h, self.w = input_shape[1:]
        self.cache_file = cache_file
        self.use_cache_only = use_cache_only

        self.current_index = 0
        self.n_images = len(self.image_paths)

        self.device_input = cuda.mem_alloc(self.batch_size * self.c * self.h * self.w * float(np.float32(0)).nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.use_cache_only:
            return None
        return None  # we don't calibrate here, only reuse cache

    def read_calibration_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                print(f"[INFO] Using calibration cache from {self.cache_file}")
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default=str(MODELS_DIR / "model.onnx"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fp16_engine", type=str, default=str(MODELS_DIR / "model_fp16.engine"))
    ap.add_argument("--int8_engine", type=str, default=str(MODELS_DIR / "model_int8.engine"))
    ap.add_argument("--calib_cache", type=str, default=str(MODELS_DIR / "calibration.cache"))
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    onnx_path = Path(args.onnx)
    fp16_engine = Path(args.fp16_engine)
    int8_engine = Path(args.int8_engine)
    calib_cache = Path(args.calib_cache)

    build_engine(
        onnx_path=onnx_path,
        engine_path=fp16_engine,
        fp16=True,
        int8=False,
        input_shape=(1, 3, args.imgsz, args.imgsz),
    )

    if not calib_cache.exists():
        raise RuntimeError(f"Calibration cache not found at {calib_cache}. Run calibrate_int8.py first.")

    build_engine(
        onnx_path=onnx_path,
        engine_path=int8_engine,
        fp16=False,
        int8=True,
        input_shape=(1, 3, args.imgsz, args.imgsz),
        calib_cache=calib_cache,
    )
