# tests/test_tracker.py

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from inference.detector import Detection
from inference.fusion import fuse_detections_and_tracks, FusedObject
from inference.tracker import Track


def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Yardımcı IoU fonksiyonu sadece test için.
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    denom = area1 + area2 - inter + 1e-7
    return float(inter / denom) if denom > 0 else 0.0


def test_fusion_high_iou_keeps_track_id() -> None:
    """
    Drift tespit IoU eşiğinin ÜSTÜNDE olduğunda,
    fused objede track_id korunmalı (det+trk).
    """
    det = Detection(
        x1=10.0,
        y1=10.0,
        x2=50.0,
        y2=50.0,
        score=0.9,
        cls=0,
    )
    trk_box = np.array([12.0, 12.0, 48.0, 48.0], dtype=float)
    trk = Track(
        track_id=42,
        bbox=trk_box,
        score=0.8,
        cls=0,
    )

    fused: List[FusedObject] = fuse_detections_and_tracks(
        [det], [trk], iou_thres=0.5
    )

    # En az bir fused obje track_id=42 olsun ve IoU gerçekten büyük olsun
    assert any(
        f.track_id == 42 and _iou(
            np.array([f.x1, f.y1, f.x2, f.y2], dtype=float),
            trk_box,
        ) > 0.5
        for f in fused
    )


def test_fusion_low_iou_triggers_drift() -> None:
    """
    Drift tespit IoU eşiğinin ALTINDA olduğunda,
    detection'ın track ile eşleşmemesi ve yeni objeye yol açması beklenir.
    (Yani track_id ya None olur ya da farklı davranış, ama en azından
    eski bbox ile yüksek IoU'lu fused + aynı track_id olmamalı.)
    """
    det_far = Detection(
        x1=200.0,
        y1=200.0,
        x2=260.0,
        y2=260.0,
        score=0.9,
        cls=0,
    )
    trk_box = np.array([10.0, 10.0, 60.0, 60.0], dtype=float)
    trk = Track(
        track_id=7,
        bbox=trk_box,
        score=0.8,
        cls=0,
    )

    fused: List[FusedObject] = fuse_detections_and_tracks(
        [det_far], [trk], iou_thres=0.5
    )

    # Eski track_id=7 ile yüksek IoU'lu fused obje olmamalı
    assert not any(
        f.track_id == 7 and _iou(
            np.array([f.x1, f.y1, f.x2, f.y2], dtype=float),
            trk_box,
        ) > 0.5
        for f in fused
    )
