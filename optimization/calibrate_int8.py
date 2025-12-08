from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit  # CUDA context
import pycuda.driver as cuda
import tensorrt as trt


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    TensorRT için entropi tabanlı INT8 kalibratör.
    Basit bir görüntü klasörünü kalibrasyon datası olarak kullanır.
    """

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

        self.buf_size = (
            self.batch_size * self.c * self.h * self.w * np.dtype(np.float32).itemsize
        )
        self.device_input = cuda.mem_alloc(self.buf_size)

    def preprocess_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Kalibrasyon görseli okunamadı: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        if self.use_cache_only:
            return None

        if self.current_index >= self.n_images:
            return None

        batch_imgs = []
        for _ in range(self.batch_size):
            if self.current_index >= self.n_images:
                break
            img_path = self.image_paths[self.current_index]
            img = self.preprocess_image(img_path)
            batch_imgs.append(img)
            self.current_index += 1

        if not batch_imgs:
            return None

        batch = np.stack(batch_imgs, axis=0)
        cuda.memcpy_htod(self.device_input, batch.ravel())
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            print(f"[INFO] Var olan kalibrasyon cache kullanılacak: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"[INFO] Kalibrasyon cache yazılıyor: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_int8_calibration_engine(
    onnx_path: Path,
    input_shape=(1, 3, 640, 640),
    calib_images_dir: Path = ROOT / "datasets" / "coco_5cls" / "images" / "train",
    n_calib: int = 256,
):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse işlemi başarısız oldu.")

    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    except AttributeError:
        config.max_workspace_size = 1 << 30

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

    if not builder.platform_has_fast_int8:
        raise RuntimeError("Bu platform INT8 desteğine sahip değil.")

    config.set_flag(trt.BuilderFlag.INT8)

    calib_paths = sorted(
        glob(str(calib_images_dir / "*.jpg"))
        + glob(str(calib_images_dir / "*.jpeg"))
        + glob(str(calib_images_dir / "*.png"))
    )
    if len(calib_paths) == 0:
        raise RuntimeError(f"Kalibrasyon için görüntü bulunamadı: {calib_images_dir}")

    calib_paths = calib_paths[:n_calib]
    print(f"[INFO] INT8 kalibrasyon için {len(calib_paths)} görüntü kullanılacak.")

    calibrator = EntropyCalibrator(
        calib_paths,
        input_shape=input_shape,
        cache_file=MODELS_DIR / "calibration.cache",
        use_cache_only=False,
    )
    config.int8_calibrator = calibrator

    print("[INFO] Kalibrasyon için geçici INT8 engine oluşturuluyor (biraz sürebilir)...")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Kalibrasyon için INT8 engine oluşturulamadı.")

    print("[OK] Kalibrasyon tamamlandı, cache models/calibration.cache dosyasına yazıldı.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default=str(MODELS_DIR / "model.onnx"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--calib_dir",
        type=str,
        default=str(ROOT / "datasets" / "coco_5cls" / "images" / "train"),
    )
    ap.add_argument("--n_calib", type=int, default=256)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_int8_calibration_engine(
        onnx_path=Path(args.onnx),
        input_shape=(1, 3, args.imgsz, args.imgsz),
        calib_images_dir=Path(args.calib_dir),
        n_calib=args.n_calib,
    )
