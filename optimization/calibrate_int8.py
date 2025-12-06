
import argparse
import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="INT8 entropy calibration for TensorRT.")
    parser.add_argument("--onnx-path", type=str, default=str(root / "models" / "model.onnx"))
    parser.add_argument("--calib-images-dir", type=str, default=str(root / "datasets" / "coco_5cls" / "images" / "train"))
    parser.add_argument("--cache-path", type=str, default=str(root / "models" / "calibration.cache"))
    parser.add_argument("--num-images", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace-size", type=int, default=2, help="Workspace size in GB")
    return parser.parse_args()


def list_images(calib_dir, max_images):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(calib_dir, e)))
    files = sorted(files)
    if not files:
        raise RuntimeError(f"No images found in {calib_dir}")
    random.shuffle(files)
    return files[:max_images]


def preprocess_image(path, imgsz=640):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img


class ImageBatchStream:
    def __init__(self, batch_size, input_shape, image_paths):
        self.batch_size = batch_size
        self.input_shape = input_shape  # (C, H, W)
        self.image_paths = image_paths
        self.max_batches = len(image_paths) // batch_size
        self.current_batch = 0

    def reset(self):
        self.current_batch = 0

    def next_batch(self):
        if self.current_batch >= self.max_batches:
            return None

        batch_paths = self.image_paths[
            self.current_batch * self.batch_size : (self.current_batch + 1) * self.batch_size
        ]
        c, h, w = self.input_shape
        batch = np.zeros((self.batch_size, c, h, w), dtype=np.float32)
        for i, p in enumerate(batch_paths):
            batch[i] = preprocess_image(p, imgsz=w)
        self.current_batch += 1
        return batch


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batchstream: ImageBatchStream, cache_file: str):
        super().__init__()
        self.batchstream = batchstream
        self.cache_file = cache_file
        self.current_batch = None
        c, h, w = self.batchstream.input_shape
        self.d_input = cuda.mem_alloc(self.batchstream.batch_size * c * h * w * np.float32().nbytes)

    def get_batch_size(self):
        return self.batchstream.batch_size

    def get_batch(self, names):
        batch = self.batchstream.next_batch()
        if batch is None:
            return None
        cuda.memcpy_htod(self.d_input, batch.ravel())
        return [int(self.d_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[INFO] Using existing calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"[INFO] Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def run_calibration(args):
    logger = trt.Logger(trt.Logger.INFO)
    onnx_path = args.onnx_path
    cache_path = args.cache_path

    print(f"[INFO] Using ONNX: {onnx_path}")
    print(f"[INFO] Calibration images dir: {args.calib_images_dir}")
    print(f"[INFO] Cache path: {cache_path}")

    image_paths = list_images(args.calib_images_dir, max_images=args.num_images)
    print(f"[INFO] Selected {len(image_paths)} images for calibration.")

    # Build network
    with trt.Builder(logger) as builder, \
         builder.create_network(flags=int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, logger) as parser:

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX file.")

        input_tensor = network.get_input(0)
        _, c, h, w = input_tensor.shape  # dynamic batch dim is -1

        batchstream = ImageBatchStream(
            batch_size=args.batch_size,
            input_shape=(c, h, w),
            image_paths=image_paths,
        )
        calibrator = EntropyCalibrator(batchstream, cache_file=cache_path)

        config = builder.create_builder_config()
        config.max_workspace_size = args.workspace_size * (1 << 30)  # GB -> bytes
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

        profile = builder.create_optimization_profile()
        min_shape = (1, c, args.imgsz, args.imgsz)
        opt_shape = (args.batch_size, c, args.imgsz, args.imgsz)
        max_shape = (args.batch_size * 2, c, args.imgsz, args.imgsz)
        profile.set_shape(input_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)

        print("[INFO] Starting INT8 calibration (engine build will be discarded)...")
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build engine during calibration.")
        print("[INFO] Calibration completed successfully.")


def main():
    args = parse_args()
    run_calibration(args)


if __name__ == "__main__":
    main()
