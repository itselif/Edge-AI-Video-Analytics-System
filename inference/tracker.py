from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .detector import Detection


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # (4,) xyxy
    score: float
    cls: int
    age: int = 0
    lost: int = 0
    is_active: bool = True
    cv2_tracker: Optional[object] = None  # OpenCV CSRT tracker


class MultiObjectTracker:
    """
    IoU-based multi-object tracking for association on detector frames,
    combined with OpenCV CSRT trackers to propagate boxes on frames
    where the detector is not run.
    """

    def __init__(
        self,
        tracker_type: str = "IOU_CSRT",
        max_lost: int = 30,
        iou_thres: float = 0.5,
    ):
        self.tracker_type = tracker_type
        self.max_lost = max_lost

        self.iou_thres = iou_thres
        self.drift_iou_thres = iou_thres

        self.tracks: List[Track] = []
        self.next_id: int = 1

    # ---------------------------------------------------------------------
    # Public API: detector + tracker
    # ---------------------------------------------------------------------
    def update_with_detections(
        self,
        frame_bgr: np.ndarray,
        detections: List[Detection],
    ) -> List[Track]:
        if len(detections) == 0:
            self._increment_lost()
            self._remove_dead_tracks()
            return self.tracks

        det_bboxes = np.array(
            [[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32
        )
        det_scores = np.array([d.score for d in detections], dtype=np.float32)
        det_classes = np.array([d.cls for d in detections], dtype=np.int32)

        if len(self.tracks) == 0:
            # First frame: create tracks for all detections
            for i in range(len(detections)):
                self._init_track(
                    frame_bgr=frame_bgr,
                    bbox=det_bboxes[i],
                    score=float(det_scores[i]),
                    cls=int(det_classes[i]),
                )
            return self.tracks

        track_bboxes = np.array([t.bbox for t in self.tracks], dtype=np.float32)
        iou_mat = self._iou_matrix(track_bboxes, det_bboxes)

        num_tracks, num_dets = iou_mat.shape
        matched_tracks = set()
        matched_dets = set()

        # Greedy IoU-based matching
        flat_indices = [(t, d) for t in range(num_tracks) for d in range(num_dets)]
        flat_indices.sort(key=lambda x: iou_mat[x[0], x[1]], reverse=True)

        for t_idx, d_idx in flat_indices:
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            iou = iou_mat[t_idx, d_idx]
            if iou < self.iou_thres:
                continue

            self._update_track(
                frame_bgr=frame_bgr,
                track=self.tracks[t_idx],
                bbox=det_bboxes[d_idx],
                score=float(det_scores[d_idx]),
                cls=int(det_classes[d_idx]),
            )
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

        # Unmatched detections → new tracks
        for d_idx in range(num_dets):
            if d_idx not in matched_dets:
                self._init_track(
                    frame_bgr=frame_bgr,
                    bbox=det_bboxes[d_idx],
                    score=float(det_scores[d_idx]),
                    cls=int(det_classes[d_idx]),
                )

        # Unmatched tracks → lost++
        for t_idx in range(num_tracks):
            if t_idx not in matched_tracks:
                trk = self.tracks[t_idx]
                trk.age += 1
                trk.lost += 1
                if trk.lost > self.max_lost:
                    trk.is_active = False

        self._remove_dead_tracks()
        return self.tracks

    def track_only(self, frame_bgr: np.ndarray) -> List[Track]:
        """
        Update tracks using OpenCV CSRT trackers when no detections are available.
        """
        for trk in self.tracks:
            if trk.cv2_tracker is not None:
                ok, box = trk.cv2_tracker.update(frame_bgr)
                if ok:
                    x, y, w, h = box
                    x1 = float(x)
                    y1 = float(y)
                    x2 = float(x + w)
                    y2 = float(y + h)
                    trk.bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                    trk.age += 1
                    trk.lost = 0
                    trk.is_active = True
                    continue

            # If tracker is missing or update failed
            trk.age += 1
            trk.lost += 1
            if trk.lost > self.max_lost:
                trk.is_active = False

        self._remove_dead_tracks()
        return self.tracks

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _create_csrt_tracker(self) -> object:
        try:
            return cv2.legacy.TrackerCSRT_create()  # OpenCV >= 4.5 (legacy module)
        except AttributeError:
            return cv2.TrackerCSRT_create()  # Older OpenCV builds

    def _init_track(self, frame_bgr: np.ndarray, bbox: np.ndarray, score: float, cls: int):
        tracker = self._create_csrt_tracker()
        x1, y1, x2, y2 = bbox.tolist()
        w = float(x2 - x1)
        h = float(y2 - y1)
        tracker.init(frame_bgr, (float(x1), float(y1), w, h))

        new_track = Track(
            track_id=self.next_id,
            bbox=bbox.copy(),
            score=score,
            cls=cls,
            age=1,
            lost=0,
            is_active=True,
            cv2_tracker=tracker,
        )
        self.tracks.append(new_track)
        self.next_id += 1

    def _update_track(
        self,
        frame_bgr: np.ndarray,
        track: Track,
        bbox: np.ndarray,
        score: float,
        cls: int,
    ):
        track.bbox = bbox.copy()
        track.score = score
        track.cls = cls
        track.age += 1
        track.lost = 0
        track.is_active = True

        tracker = self._create_csrt_tracker()
        x1, y1, x2, y2 = bbox.tolist()
        w = float(x2 - x1)
        h = float(y2 - y1)
        tracker.init(frame_bgr, (float(x1), float(y1), w, h))
        track.cv2_tracker = tracker

    def _increment_lost(self):
        for trk in self.tracks:
            trk.age += 1
            trk.lost += 1
            if trk.lost > self.max_lost:
                trk.is_active = False

    def _remove_dead_tracks(self):
        self.tracks = [t for t in self.tracks if t.is_active]

    @staticmethod
    def _iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        if boxes1.size == 0 or boxes2.size == 0:
            return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

        b1 = boxes1[:, None, :]  # (N, 1, 4)
        b2 = boxes2[None, :, :]  # (1, M, 4)

        x1 = np.maximum(b1[..., 0], b2[..., 0])
        y1 = np.maximum(b1[..., 1], b2[..., 1])
        x2 = np.minimum(b1[..., 2], b2[..., 2])
        y2 = np.minimum(b1[..., 3], b2[..., 3])

        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter = inter_w * inter_h

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        union = area1[:, None] + area2[None, :] - inter
        iou = inter / (union + 1e-6)
        return iou.astype(np.float32)
