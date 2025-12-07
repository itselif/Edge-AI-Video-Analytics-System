# inference/utils.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

from inference.detector import Detection


def compute_iou(box1: Tuple[float, float, float, float],
                box2: Tuple[float, float, float, float]) -> float:
    """
    IoU between two boxes in xyxy format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))

    union = area1 + area2 - inter + 1e-6
    return float(inter / union)


@dataclass
class TrackViz:
    track_id: int
    bbox: Tuple[float, float, float, float]
    score: float
    cls: int


def _get_color(track_id: int) -> Tuple[int, int, int]:
    """
    Deterministic pseudo-random color for a given track id.
    """
    np.random.seed(track_id + 42)
    color = np.random.randint(0, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    class_names: Optional[List[str]] = None,
    base_color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw plain detections (no track ids).
    """
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, [det.x1, det.y1, det.x2, det.y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), base_color, 2)
        label = f"{det.cls}:{det.score:.2f}"
        if class_names is not None and 0 <= det.cls < len(class_names):
            label = f"{class_names[det.cls]} {det.score:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, base_color, 1, cv2.LINE_AA)
    return out


def draw_tracks(
    frame: np.ndarray,
    tracks: List[TrackViz],
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Draw tracked objects with unique colors per track id.
    """
    out = frame.copy()
    for trk in tracks:
        x1, y1, x2, y2 = map(int, trk.bbox)
        color = _get_color(trk.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        cls_name = str(trk.cls)
        if class_names is not None and 0 <= trk.cls < len(class_names):
            cls_name = class_names[trk.cls]

        label = f"ID {trk.track_id} | {cls_name} {trk.score:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out
