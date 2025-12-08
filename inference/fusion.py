from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional

import numpy as np

from inference.detector import Detection
from inference.tracker import Track


@dataclass
class FusedObject:
    """
    Unified object representation that merges detection and tracker information.

    track_id:
        The associated tracker ID. None if the object is newly detected or reinitialized.

    source:
        "det+trk"  - Detection and tracker are matched (high IoU).
        "det_only" - Detection exists but no valid tracker match (drift / new object).
        "trk_only" - No detections in this frame; relying on tracker only.
    """

    track_id: Optional[int]
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int
    source: str


def _bbox_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU between two sets of bounding boxes.

    boxes1: (N, 4) in [x1, y1, x2, y2]
    boxes2: (M, 4)
    Returns:
        IoU matrix of shape (N, M)
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=float)

    x11, y11, x12, y12 = (
        boxes1[:, 0][:, None],
        boxes1[:, 1][:, None],
        boxes1[:, 2][:, None],
        boxes1[:, 3][:, None],
    )
    x21, y21, x22, y22 = (
        boxes2[:, 0][None, :],
        boxes2[:, 1][None, :],
        boxes2[:, 2][None, :],
        boxes2[:, 3][None, :],
    )

    inter_x1 = np.maximum(x11, x21)
    inter_y1 = np.maximum(y11, y21)
    inter_x2 = np.minimum(x12, x22)
    inter_y2 = np.minimum(y12, y22)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    union = area1 + area2 - inter_area + 1e-7
    return inter_area / union


def fuse_detections_and_tracks(
    detections: Sequence[Detection],
    tracks: Sequence[Track],
    iou_thres: float = 0.5,
) -> List[FusedObject]:
    """
    Fuses detector outputs and tracker predictions for a single frame.

    Drift Handling Logic:
        - If IoU(det, track) >= threshold:
              → trusted match → "det+trk"
        - If IoU(det, track) < threshold:
              → tracker is not trusted for this object → detection becomes "det_only"
        - If no detections exist:
              → return tracker predictions as "trk_only"

    This function does not return unmatched trackers in frames where detections exist,
    ensuring proper drift reset behavior.
    """

    dets = list(detections)
    trks = list(tracks)
    fused: List[FusedObject] = []

    # Case 0: No detections → rely on tracker only
    if len(dets) == 0:
        for trk in trks:
            x1, y1, x2, y2 = trk.bbox.tolist()
            fused.append(
                FusedObject(
                    track_id=trk.track_id,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=float(trk.score),
                    cls=int(trk.cls),
                    source="trk_only",
                )
            )
        return fused

    Nd, Nt = len(dets), len(trks)

    if Nt > 0:
        det_boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=float)
        trk_boxes = np.array([trk.bbox for trk in trks], dtype=float)
        ious = _bbox_iou(det_boxes, trk_boxes)
    else:
        ious = np.zeros((Nd, 0), dtype=float)

    matched_dets = set()
    matched_trks = set()

    # Greedy IoU-based matching
    if Nd > 0 and Nt > 0:
        ious_copy = ious.copy()
        while True:
            max_idx = np.unravel_index(np.argmax(ious_copy), ious_copy.shape)
            max_iou = ious_copy[max_idx]

            if max_iou < iou_thres:
                break

            d_idx, t_idx = max_idx

            if d_idx in matched_dets or t_idx in matched_trks:
                ious_copy[d_idx, t_idx] = -1.0
                continue

            det = dets[d_idx]
            trk = trks[t_idx]

            fused.append(
                FusedObject(
                    track_id=trk.track_id,
                    x1=float(det.x1),
                    y1=float(det.y1),
                    x2=float(det.x2),
                    y2=float(det.y2),
                    score=float(det.score),
                    cls=int(det.cls),
                    source="det+trk",
                )
            )

            matched_dets.add(d_idx)
            matched_trks.add(t_idx)

            ious_copy[d_idx, :] = -1.0
            ious_copy[:, t_idx] = -1.0

    # Unmatched detections → new objects (drift)
    for d_idx, det in enumerate(dets):
        if d_idx in matched_dets:
            continue

        fused.append(
            FusedObject(
                track_id=None,
                x1=float(det.x1),
                y1=float(det.y1),
                x2=float(det.x2),
                y2=float(det.y2),
                score=float(det.score),
                cls=int(det.cls),
                source="det_only",
            )
        )

    # Unmatched trackers are intentionally NOT included when detections exist
    # to enforce drift reset behavior.

    return fused
