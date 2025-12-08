from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_logger() -> trt.Logger:
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    return logger


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Minimal INT8 calibrator that only reads an existing calibration cache.

    The actual calibration (collecting batches and writing the cache) is expected
    to be done in a separate script (e.g. calibrate_int8.py).
    """

    def __init__(
        self,
        cache_file: Path = MODELS_DIR / "calibration.cache",
    ):
        super().__init__()
        self.cache_file = cache_file

    def get_batch_size(self) -> int:
        # Not used when we rely solely on an existing cache.
        return 1

    def get_batch(self, names):
        # Returning None tells TensorRT to use the calibration cache only.
        return None

    def read_calibration_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                print(f"[INFO] Using calibration cache from {self.cache_file}")
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


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
            raise RuntimeError("Failed to parse ONNX model.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    input_name = network.get_input(0).name
    _, c, h, w = input_shape

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (1, c, 320, 320),       # min
        (1, c, h, w),           # opt
        (4, c, 1280, 1280),     # max
    )
    config.add_optimization_profile(profile)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        if not builder.platform_has_fast_int8:
            raise RuntimeError("INT8 not supported on this platform.")
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = EntropyCalibrator(cache_file=calib_cache)
        config.int8_calibrator = calibrator

    print(f"[INFO] Building TensorRT engine -> {engine_path.name} (fp16={fp16}, int8={int8})")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    print(f"[OK] Saved TensorRT engine to {engine_path}")


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
        raise RuntimeError(
            f"Calibration cache not found at {calib_cache}. "
            f"Run calibrate_int8.py first to generate it."
        )

    build_engine(
        onnx_path=onnx_path,
        engine_path=int8_engine,
        fp16=False,
        int8=True,
        input_shape=(1, 3, args.imgsz, args.imgsz),
        calib_cache=calib_cache,
    )
