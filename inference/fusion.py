# inference/fusion.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from inference.detector import Detection
from inference.tracker import Track
from inference.utils import compute_iou


@dataclass
class FusedObject:
    track_id: Optional[int]
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int
    source: str  # "detector", "tracker", "det+trk"


def fuse_detections_and_tracks(
    detections: List[Detection],
    tracks: List[Track],
    iou_thres: float = 0.5,
) -> List[FusedObject]:

    fused: List[FusedObject] = []

    used_det = set()

    # 1) For each track, match to detection
    for trk in tracks:
        best_iou = 0.0
        best_det_idx = None
        for i, det in enumerate(detections):
            if i in used_det:
                continue
            if det.cls != trk.cls:
                continue
            iou = compute_iou(
                (trk.bbox[0], trk.bbox[1], trk.bbox[2], trk.bbox[3]),
                (det.x1, det.y1, det.x2, det.y2),
            )
            if iou > best_iou:
                best_iou = iou
                best_det_idx = i

        if best_det_idx is not None and best_iou >= iou_thres:
            det = detections[best_det_idx]
            used_det.add(best_det_idx)
            fused.append(
                FusedObject(
                    track_id=trk.track_id,
                    x1=det.x1,
                    y1=det.y1,
                    x2=det.x2,
                    y2=det.y2,
                    score=det.score,
                    cls=det.cls,
                    source="det+trk",
                )
            )
        else:
            # no matching detection → trust track
            fused.append(
                FusedObject(
                    track_id=trk.track_id,
                    x1=trk.bbox[0],
                    y1=trk.bbox[1],
                    x2=trk.bbox[2],
                    y2=trk.bbox[3],
                    score=trk.score,
                    cls=trk.cls,
                    source="tracker",
                )
            )

    # 2) Remaining detections → new fused objects (no track attached)
    for i, det in enumerate(detections):
        if i in used_det:
            continue
        fused.append(
            FusedObject(
                track_id=None,
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
                score=det.score,
                cls=det.cls,
                source="detector",
            )
        )

    return fused
