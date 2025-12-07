# inference/detector.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Union, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int


class Detector:
    def __init__(
        self,
        backend: str,
        model_path: Union[str, Path],
        imgsz: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str = "cpu",
        onnx_providers: Optional[Sequence[str]] = None,
    ):
        self.backend = backend.lower()
        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.onnx_providers = list(onnx_providers) if onnx_providers is not None else None

        if self.backend not in ("torch", "onnx"):
            raise ValueError(f"Unsupported backend: {backend}")

        if self.backend == "torch":
            # PyTorch YOLO
            self.model = YOLO(self.model_path)
            try:
                self.model.to(self.device)
            except Exception:
                pass
            self.ort_session = None
            print(f"[Detector] PyTorch backend, model={self.model_path}, device={self.device}")

        else:
            # ONNX Runtime
            import onnxruntime as ort

            if self.onnx_providers is None:
                self.onnx_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 2

            try:
                self.ort_session = ort.InferenceSession(
                    self.model_path, sess_opts, providers=self.onnx_providers
                )
            except Exception as e:
                print(f"[Detector][WARN] ORT session with {self.onnx_providers} failed: {e}")
                self.onnx_providers = ["CPUExecutionProvider"]
                self.ort_session = ort.InferenceSession(
                    self.model_path, sess_opts, providers=self.onnx_providers
                )

            self.model = None
            print(f"[Detector] ONNX backend, model={self.model_path}, providers={self.onnx_providers}")

    # ------------------------------------------------------------------
    # Ana API
    # ------------------------------------------------------------------
    def __call__(
        self,
        frames: Union[np.ndarray, Sequence[np.ndarray]],
    ) -> Union[List[Detection], List[List[Detection]]]:
        # frames None ise boş dön
        if frames is None:
            return []

        if isinstance(frames, np.ndarray):
            single = True
            batch = [frames]
        elif isinstance(frames, Sequence):
            single = False
            batch = list(frames)
        else:
            raise ValueError(
                f"frames must be a numpy array or a sequence of arrays, got {type(frames)}"
            )

        if self.backend == "torch":
            all_dets = [self._infer_torch_single(img) for img in batch]
        else:
            all_dets = [self._infer_onnx_single(img) for img in batch]

        return all_dets[0] if single else all_dets

    # ------------------------------------------------------------------
    # PyTorch path (Ultralytics kendi post-process)
    # ------------------------------------------------------------------
    def _infer_torch_single(self, img_bgr: np.ndarray) -> List[Detection]:
        if img_bgr is None:
            return []

        res_list = self.model.predict(
            source=img_bgr,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )
        if not res_list:
            return []

        res = res_list[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy
        conf = boxes.conf
        cls_ = boxes.cls

        if isinstance(xyxy, torch.Tensor):
            xyxy = xyxy.cpu().numpy()
        if isinstance(conf, torch.Tensor):
            conf = conf.cpu().numpy()
        if isinstance(cls_, torch.Tensor):
            cls_ = cls_.cpu().numpy()

        dets: List[Detection] = []
        for i in range(len(conf)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            dets.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=float(conf[i]),
                    cls=int(cls_[i]),
                )
            )
        return dets

    # ------------------------------------------------------------------
    # ONNX Runtime path + manuel post-process
    # ------------------------------------------------------------------
    def _infer_onnx_single(self, img_bgr: np.ndarray) -> List[Detection]:
        if img_bgr is None:
            return []

        sess = self.ort_session
        if sess is None:
            return []

        orig_h, orig_w = img_bgr.shape[:2]

        # 1) Preprocess
        img_resized = cv2.resize(img_bgr, (self.imgsz, self.imgsz))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))  # (3, H, W)
        inp = img_chw[None, :, :, :]  # (1, 3, H, W)

        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: inp})

        out = outputs[0]
        # Debug: şekli görelim
        # Örn: (1, 9, 8400)
        out = np.squeeze(out, axis=0)  # (C, N) veya (N, C)
        print("[ONNX] raw output shape after squeeze:", out.shape)

        if out.ndim != 2:
            return []

        if out.shape[0] == 9:
            out = out.transpose(1, 0)  # (8400, 9)

        num_preds, num_ch = out.shape
        if num_ch < 5:
            return []

        # Varsayım: 4 bbox (cx, cy, w, h) + 5 class skor = 9
        xywh = out[:, :4]          # (N, 4)
        class_scores = out[:, 4:]  # (N, 5)

        # class_scores içinden max skor + sınıf id
        best_scores = class_scores.max(axis=1)
        best_cls = class_scores.argmax(axis=1)

        # conf threshold
        keep = best_scores >= self.conf_thres
        xywh = xywh[keep]
        best_scores = best_scores[keep]
        best_cls = best_cls[keep]

        if xywh.shape[0] == 0:
            return []

        # 2) xywh -> xyxy
        boxes_xyxy = self._xywh_to_xyxy(xywh)

        # 3) 640x640 -> orijinal boyuta scale
        scale_x = orig_w / float(self.imgsz)
        scale_y = orig_h / float(self.imgsz)
        boxes_xyxy[:, 0] *= scale_x
        boxes_xyxy[:, 2] *= scale_x
        boxes_xyxy[:, 1] *= scale_y
        boxes_xyxy[:, 3] *= scale_y

        # 4) NMS
        keep_idx = self._nms(boxes_xyxy, best_scores, self.iou_thres)
        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = best_scores[keep_idx]
        cls_ids = best_cls[keep_idx]

        dets: List[Detection] = []
        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            dets.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=float(scores[i]),
                    cls=int(cls_ids[i]),
                )
            )

        return dets

    # ----------------- yardımcılar -----------------

    @staticmethod
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        """
        xywh: (N, 4), (cx, cy, w, h)
        -> (N, 4), (x1, y1, x2, y2)
        """
        cx = xywh[:, 0]
        cy = xywh[:, 1]
        w = xywh[:, 2]
        h = xywh[:, 3]

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
        """
        boxes: (N, 4)  xyxy
        scores: (N,)
        """
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]

        return keep
