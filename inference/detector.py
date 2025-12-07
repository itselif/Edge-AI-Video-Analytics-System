# inference/detector.py

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, List, Optional, Tuple, Union, Sequence

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import onnxruntime as ort

# -----------------------------
# Lazy TensorRT / PyCUDA import
# -----------------------------
trt = None
cuda = None
HAS_TRT = False


def _lazy_init_trt():
    """
    Import tensorrt/pycuda only when needed (backend='tensorrt').
    This avoids crashing kernels in environments where TensorRT
    is misconfigured or unavailable.
    """
    global trt, cuda, HAS_TRT
    if HAS_TRT:
        return

    try:
        import tensorrt as _trt  # type: ignore
        import pycuda.driver as _cuda  # type: ignore
        import pycuda.autoinit  # noqa: F401  # creates CUDA context

        trt = _trt
        cuda = _cuda
        HAS_TRT = True
    except Exception:
        trt = None
        cuda = None
        HAS_TRT = False


BackendType = Literal["torch", "onnx", "tensorrt"]


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int


@dataclass
class DetectorStats:
    last_pre_ms: float = 0.0
    last_infer_ms: float = 0.0
    last_post_ms: float = 0.0

    avg_pre_ms: float = 0.0
    avg_infer_ms: float = 0.0
    avg_post_ms: float = 0.0

    num_runs: int = 0

    def update(self, pre: float, infer: float, post: float):
        self.num_runs += 1
        alpha = 1.0 / self.num_runs

        self.avg_pre_ms = (1 - alpha) * self.avg_pre_ms + alpha * pre
        self.avg_infer_ms = (1 - alpha) * self.avg_infer_ms + alpha * infer
        self.avg_post_ms = (1 - alpha) * self.avg_post_ms + alpha * post

        self.last_pre_ms = pre
        self.last_infer_ms = infer
        self.last_post_ms = post


class Detector:

    def __init__(
        self,
        backend: BackendType,
        model_path: Union[str, Path],
        imgsz: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str = "cuda",
        onnx_providers: Optional[List[str]] = None,
    ):
        self.backend: BackendType = backend
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.stats = DetectorStats()

        if onnx_providers is None:
            # TensorRT EP'i en öne koy, ortamda yoksa ONNX otomatik fallback yapar
            self.onnx_providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            self.onnx_providers = onnx_providers

        # backend-specific handles
        self.yolo_model: Optional[YOLO] = None
        self.onnx_session: Optional[ort.InferenceSession] = None
        self.onnx_input_name: Optional[str] = None

        self.trt_logger = None
        self.trt_engine = None
        self.trt_context = None
        self.trt_input_idx = None
        self.trt_output_idx = None

        self._load_backend()

    # ----------------------------
    # Backend initialization
    # ----------------------------
    def _load_backend(self):
        if self.backend == "torch":
            self._init_torch()
        elif self.backend == "onnx":
            self._init_onnx()
        elif self.backend == "tensorrt":
            self._init_tensorrt()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _init_torch(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"PyTorch model not found: {self.model_path}")
        model = YOLO(str(self.model_path))
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA not available, falling back to CPU for torch backend.")
            self.device = "cpu"
        model.to(self.device)
        model.eval()
        self.yolo_model = model
        print(f"[Detector] Loaded PyTorch YOLO model from {self.model_path}")

    def _init_onnx(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        sess = ort.InferenceSession(
            str(self.model_path),
            providers=self.onnx_providers,
        )
        self.onnx_session = sess
        self.onnx_input_name = sess.get_inputs()[0].name
        print(f"[Detector] Loaded ONNX model from {self.model_path}")
        print(f"[Detector] ONNX providers in use: {sess.get_providers()}")

    def _init_tensorrt(self):
        # Lazy import: sadece gerçekten tensorrt backend kullanırsak
        _lazy_init_trt()

        if not HAS_TRT or trt is None or cuda is None:
            raise RuntimeError("TensorRT backend requested but tensorrt/pycuda is not available.")

        if not self.model_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.model_path}")

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, "")

        with open(self.model_path, "rb") as f:
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")

        self.trt_engine = engine
        self.trt_context = engine.create_execution_context()

        # find bindings
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                self.trt_input_idx = i
            else:
                self.trt_output_idx = i

        if self.trt_input_idx is None or self.trt_output_idx is None:
            raise RuntimeError("Could not find input/output bindings in TensorRT engine.")

        print(f"[Detector] Loaded TensorRT engine from {self.model_path}")

    # ----------------------------
    # Public API
    # ----------------------------
    def warmup(self, num_iters: int = 10):
        """
        Run a few dummy inferences to let backend JIT / allocate / optimize.
        Uses batch_size=1 dummy frame.
        """
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(num_iters):
            _ = self(dummy)

    def __call__(
        self,
        frames: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[List[Detection], List[List[Detection]]]:
        """
        Run detection on a single frame or a batch of frames (BGR, HxWx3).

        - Input: np.ndarray (H, W, 3)  → Output: List[Detection]
        - Input: list/sequence of frames → Output: List[List[Detection]]
        """
        # normalize input to list[frame]
        if isinstance(frames, np.ndarray) and frames.ndim == 3:
            frame_list = [frames]
            single_input = True
        elif isinstance(frames, Sequence):
            frame_list = list(frames)
            if len(frame_list) == 0:
                return []  # boş batch
            single_input = False
        else:
            raise ValueError("frames must be a single HxWx3 ndarray or a sequence of such arrays")

        # shapes list
        orig_hws = [f.shape[:2] for f in frame_list]

        # preprocess (batch)
        t0 = time.perf_counter()
        imgs_batched, ratios, pads = self._preprocess_batch(frame_list)
        pre_ms = (time.perf_counter() - t0) * 1000.0

        # inference
        t1 = time.perf_counter()
        if self.backend == "torch":
            raw = self._infer_torch(imgs_batched)
        elif self.backend == "onnx":
            raw = self._infer_onnx(imgs_batched)
        else:
            raw = self._infer_tensorrt(imgs_batched)
        infer_ms = (time.perf_counter() - t1) * 1000.0

        # postprocess per image
        t2 = time.perf_counter()
        all_dets: List[List[Detection]] = []
        for i in range(len(frame_list)):
            raw_i = raw[i]
            hw_i = orig_hws[i]
            ratio_i = ratios[i]
            pad_i = pads[i]
            dets_i = self._postprocess(raw_i, hw_i, ratio_i, pad_i)
            all_dets.append(dets_i)
        post_ms = (time.perf_counter() - t2) * 1000.0

        self.stats.update(pre_ms, infer_ms, post_ms)

        if single_input:
            return all_dets[0]
        return all_dets

    # ----------------------------
    # Pre / post-processing
    # ----------------------------
    def _preprocess_batch(
        self, frames_bgr: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, List[float], List[Tuple[int, int]]]:
        """
        Letterbox + normalize + CHW + batch for multiple frames.
        Returns:
            imgs_batched: (B, 3, H, W)
            ratios: list of scale factors
            pads: list of (dx, dy)
        """
        batch_imgs = []
        ratios: List[float] = []
        pads: List[Tuple[int, int]] = []

        for frame_bgr in frames_bgr:
            h0, w0 = frame_bgr.shape[:2]
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # letterbox to square imgsz
            scale = min(self.imgsz / h0, self.imgsz / w0)
            nh, nw = int(h0 * scale), int(w0 * scale)
            img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            dy = (self.imgsz - nh) // 2
            dx = (self.imgsz - nw) // 2
            canvas[dy : dy + nh, dx : dx + nw, :] = img_resized

            img_norm = canvas.astype(np.float32) / 255.0
            img_chw = np.transpose(img_norm, (2, 0, 1))  # HWC -> CHW
            batch_imgs.append(img_chw)
            ratios.append(scale)
            pads.append((dx, dy))

        imgs_batched = np.stack(batch_imgs, axis=0)  # (B, 3, H, W)
        return imgs_batched, ratios, pads

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
        """
        Simple NMS implementation on CPU.
        boxes: (N, 4) in xyxy
        scores: (N,)
        Returns indices of selected boxes.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]

        return keep

    def _postprocess(
        self,
        raw_single: np.ndarray,
        orig_hw: Tuple[int, int],
        scale: float,
        pad: Tuple[int, int],
    ) -> List[Detection]:
        """
        Post-process single image raw model output to Detection list.
        Assumes YOLO-like output: (num_boxes, 5 + num_classes)
        """
        orig_h, orig_w = orig_hw
        dx, dy = pad

        # shape normalizasyonu
        if raw_single.ndim == 3:
            # (1, N, 5+nc) -> (N, 5+nc)
            raw_single = raw_single[0]
        if raw_single.ndim != 2 or raw_single.shape[1] < 5:
            raise ValueError(f"Unexpected prediction shape for YOLO-style output: {raw_single.shape}")

        boxes = raw_single[:, 0:4]
        obj_conf = raw_single[:, 4:5]
        cls_conf = raw_single[:, 5:]

        cls_idx = cls_conf.argmax(axis=1)
        cls_scores = cls_conf.max(axis=1)
        scores = (obj_conf[:, 0] * cls_scores).astype(np.float32)

        mask = scores > self.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        cls_idx = cls_idx[mask]

        if boxes.shape[0] == 0:
            return []

        # xywh -> xyxy
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # undo letterbox (scale + pad)
        boxes_xyxy[:, 0] -= dx
        boxes_xyxy[:, 1] -= dy
        boxes_xyxy[:, 2] -= dx
        boxes_xyxy[:, 3] -= dy
        boxes_xyxy /= scale

        # clip
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w - 1)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h - 1)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w - 1)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h - 1)

        # custom NMS
        keep = self._nms(boxes_xyxy, scores, self.iou_thres)

        dets: List[Detection] = []
        for i in keep:
            dets.append(
                Detection(
                    x1=float(boxes_xyxy[i, 0]),
                    y1=float(boxes_xyxy[i, 1]),
                    x2=float(boxes_xyxy[i, 2]),
                    y2=float(boxes_xyxy[i, 3]),
                    score=float(scores[i]),
                    cls=int(cls_idx[i]),
                )
            )
        return dets

    # ----------------------------
    # Backend-specific inference
    # ----------------------------
    def _infer_torch(self, imgs_batched: np.ndarray) -> np.ndarray:
        """
        imgs_batched: (B, 3, H, W) float32
        Returns: (B, N, 5+nc) numpy
        """
        assert self.yolo_model is not None
        x = torch.from_numpy(imgs_batched).to(self.device)
        with torch.no_grad():
            # Ultralytics low-level model call: returns list, first element is pred tensor
            out = self.yolo_model.model(x)[0]  # (B, N, 5+nc)
        return out.detach().cpu().numpy()

    def _infer_onnx(self, imgs_batched: np.ndarray) -> np.ndarray:
        """
        imgs_batched: (B, 3, H, W)
        Returns: (B, N, 5+nc) numpy
        """
        assert self.onnx_session is not None
        assert self.onnx_input_name is not None
        out = self.onnx_session.run(None, {self.onnx_input_name: imgs_batched})
        if isinstance(out, list):
            out = out[0]
        return out

    def _infer_tensorrt(self, imgs_batched: np.ndarray) -> np.ndarray:
        """
        imgs_batched: (B, 3, H, W)
        Returns: (B, N, 5+nc) numpy
        """
        _lazy_init_trt()
        if not HAS_TRT or trt is None or cuda is None:
            raise RuntimeError("TensorRT is not available in this environment.")
        if self.trt_engine is None or self.trt_context is None:
            raise RuntimeError("TensorRT engine/context not initialized.")

        batch, c, h, w = imgs_batched.shape
        self.trt_context.set_binding_shape(self.trt_input_idx, (batch, c, h, w))

        inp_shape = self.trt_context.get_binding_shape(self.trt_input_idx)
        out_shape = self.trt_context.get_binding_shape(self.trt_output_idx)

        inp_size = int(np.prod(inp_shape))
        out_size = int(np.prod(out_shape))

        inp_dtype = trt.nptype(self.trt_engine.get_binding_dtype(self.trt_input_idx))
        out_dtype = trt.nptype(self.trt_engine.get_binding_dtype(self.trt_output_idx))

        d_input = cuda.mem_alloc(inp_size * np.dtype(inp_dtype).itemsize)
        d_output = cuda.mem_alloc(out_size * np.dtype(out_dtype).itemsize)

        stream = cuda.Stream()

        cuda.memcpy_htod_async(d_input, imgs_batched.astype(inp_dtype).ravel(), stream)

        bindings = [None] * self.trt_engine.num_bindings
        bindings[self.trt_input_idx] = int(d_input)
        bindings[self.trt_output_idx] = int(d_output)

        self.trt_context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        out_host = np.empty(out_size, dtype=out_dtype)
        cuda.memcpy_dtoh_async(out_host, d_output, stream)
        stream.synchronize()

        d_input.free()
        d_output.free()

        return out_host.reshape(out_shape)
