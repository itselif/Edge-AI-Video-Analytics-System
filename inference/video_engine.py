from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .detector import Detector, Detection
from .tracker import MultiObjectTracker, Track
from .fusion import FusedObject, fuse_detections_and_tracks


@dataclass
class VideoEngineConfig:
    detect_every_n: int = 5
    draw_thickness: int = 2
    font_scale: float = 0.5
    score_thres: float = 0.25


class VideoEngine:
    """
    Real-time video processing engine:
        frame -> detector (periodic) -> tracker -> fusion -> visualization
    """

    def __init__(
        self,
        detector: Detector,
        tracker: MultiObjectTracker,
        class_names: Optional[List[str]] = None,
        detect_every_n: int = 5,
        display: bool = False,
        save_path: Optional[str] = None,
    ):
        self.detector = detector
        self.tracker = tracker
        self.class_names = class_names or []
        self.cfg = VideoEngineConfig(detect_every_n=detect_every_n)
        self.display = display
        self.save_path = save_path

        self._writer = None
        self._last_detections: List[Detection] = []

    # ----------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int):
        """
        Process a single BGR frame and return:
            - annotated frame
            - fused detection+track objects
        """
        if frame_bgr is None:
            raise ValueError("Frame cannot be None.")

        run_detector = (frame_idx % self.cfg.detect_every_n == 0)

        if run_detector or not self._last_detections:
            detections = self.detector(frame_bgr)
            self._last_detections = detections
            tracks: List[Track] = self.tracker.update_with_detections(frame_bgr, detections)
        else:
            tracks = self.tracker.track_only(frame_bgr)
            detections = self._last_detections

        drift_iou = getattr(self.tracker, "drift_iou_thres", 0.5)
        fused: List[FusedObject] = fuse_detections_and_tracks(
            detections, tracks, iou_thres=drift_iou
        )

        annotated = self._draw_detections(frame_bgr, detections)
        return annotated, fused

    # ----------------------------------------------------------------------
    def run(self, video_path: str, max_frames: Optional[int] = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.save_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.save_path, fourcc, fps, (w, h))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            annotated, _ = self.process_frame(frame, frame_idx)

            if self._writer is not None:
                self._writer.write(annotated)

            if self.display:
                cv2.imshow("VideoEngine", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

        cap.release()
        if self._writer is not None:
            self._writer.release()
        if self.display:
            cv2.destroyAllWindows()

    # ----------------------------------------------------------------------
    def _draw_detections(self, frame_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
        out = frame_bgr.copy()
        h, w = out.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = float(det.x1), float(det.y1), float(det.x2), float(det.y2)

            # Normalize coordinates if in [0,1]
            if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0:
                x1 *= w
                x2 *= w
                y1 *= h
                y2 *= h

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            cls_name = self._get_class_name(det.cls)
            score = float(det.score)
            color = self._color_from_id(int(det.cls))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

            label = f"{cls_name} {score:.2f}"
            ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                            self.cfg.font_scale, 1)
            cy1 = max(0, y1 - th - 4)

            cv2.rectangle(out, (x1, cy1), (x1 + tw + 4, cy1 + th + 4), color, -1)
            cv2.putText(out, label, (x1 + 2, cy1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.cfg.font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        return out

    # ----------------------------------------------------------------------
    def _get_class_name(self, cls_id: int) -> str:
        if 0 <= cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return str(cls_id)

    @staticmethod
    def _color_from_id(idx: int):
        palette = [
            (255, 56, 56),
            (56, 255, 56),
            (56, 56, 255),
            (255, 157, 151),
            (72, 249, 10),
            (0, 255, 255),
            (255, 255, 0),
            (255, 0, 255),
            (180, 130, 70),
            (0, 149, 255),
        ]
        bgr = palette[idx % len(palette)]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])
