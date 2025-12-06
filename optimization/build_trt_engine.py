import argparse
from pathlib import Path

import tensorrt as trt


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build TensorRT engines (FP16 & INT8) from ONNX.")
    parser.add_argument("--onnx-path", type=str, default=str(root / "models" / "model.onnx"))
    parser.add_argument("--fp16-engine", type=str, default=str(root / "models" / "model_fp16.engine"))
    parser.add_argument("--int8-engine", type=str, default=str(root / "models" / "model_int8.engine"))
    parser.add_argument("--calibration-cache", type=str, default=str(root / "models" / "calibration.cache"))
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--min-shape", type=int, nargs=4, default=[1, 3, 320, 320], help="min shape NCHW")
    parser.add_argument("--opt-shape", type=int, nargs=4, default=[4, 3, 640, 640], help="opt shape NCHW")
    parser.add_argument("--max-shape", type=int, nargs=4, default=[8, 3, 960, 960], help="max shape NCHW")
    parser.add_argument("--workspace-size", type=int, default=2, help="Workspace size in GB")
    return parser.parse_args()


class EntropyCalibratorFromCache(trt.IInt8EntropyCalibrator2):
    """
    Calibrator that only serves an existing calibration cache.
    """

    def __init__(self, cache_file: str):
        super().__init__()
        self.cache_file = cache_file

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        # No new calibration, rely on existing cache
        return None

    def read_calibration_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                print(f"[INFO] Using calibration cache from {self.cache_file}")
                return f.read()
        except FileNotFoundError:
            print(f"[WARN] Calibration cache not found at {self.cache_file}")
            return None

    def write_calibration_cache(self, cache):
        # We don't write in this class, cache is assumed to be precomputed
        pass


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = False,
    int8: bool = False,
    calibration_cache: Path | None = None,
    max_batch_size: int = 8,
    min_shape=(1, 3, 320, 320),
    opt_shape=(4, 3, 640, 640),
    max_shape=(8, 3, 960, 960),
    workspace_size_gb: int = 2,
):
    logger = trt.Logger(trt.Logger.INFO)
    print(f"[INFO] Building TensorRT engine from {onnx_path}")
    print(f"[INFO] Saving engine to {engine_path}")

    with trt.Builder(logger) as builder, \
         builder.create_network(flags=int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, logger) as parser:

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX.")

        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size_gb * (1 << 30)

        input_tensor = network.get_input(0)
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)

        if fp16:
            if builder.platform_has_fast_fp16:
                print("[INFO] Enabling FP16 mode")
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                print("[WARN] Platform does not support fast FP16, building FP32 engine instead")

        if int8:
            print("[INFO] Enabling INT8 mode")
            config.set_flag(trt.BuilderFlag.INT8)
            if calibration_cache is not None:
                calibrator = EntropyCalibratorFromCache(str(calibration_cache))
                config.int8_calibrator = calibrator
            else:
                print("[WARN] INT8 requested but no calibration cache provided")

        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Engine build failed.")

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"[INFO] Engine saved to {engine_path}")


def main():
    args = parse_args()
    onnx_path = Path(args.onnx_path)

    # FP16 engine
    build_engine(
        onnx_path=onnx_path,
        engine_path=Path(args.fp16_engine),
        fp16=True,
        int8=False,
        calibration_cache=None,
        max_batch_size=args.max_batch_size,
        min_shape=tuple(args.min_shape),
        opt_shape=tuple(args.opt_shape),
        max_shape=tuple(args.max_shape),
        workspace_size_gb=args.workspace_size,
    )

    # INT8 engine (using existing calibration cache)
    build_engine(
        onnx_path=onnx_path,
        engine_path=Path(args.int8_engine),
        fp16=False,
        int8=True,
        calibration_cache=Path(args.calibration_cache),
        max_batch_size=args.max_batch_size,
        min_shape=tuple(args.min_shape),
        opt_shape=tuple(args.opt_shape),
        max_shape=tuple(args.max_shape),
        workspace_size_gb=args.workspace_size,
    )


if __name__ == "__main__":
    main()
