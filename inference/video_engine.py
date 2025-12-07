# inference/video_engine.py

from __future__ import annotations

import threading
import time
from queue import Queue, Empty
from typing import Optional, List, Tuple

import cv2
import numpy as np

from inference.detector import Detector, Detection
from inference.tracker import MultiObjectTracker, Track
from inference.fusion import fuse_detections_and_tracks, FusedObject
from inference.utils import TrackViz, draw_tracks


class VideoEngine:

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
        self.detect_every_n = max(1, detect_every_n)
        self.display = display
        self.save_path = save_path

        self._frame_queue: "Queue[Tuple[int, np.ndarray]]" = Queue(maxsize=10)
        self._result_queue: "Queue[Tuple[int, np.ndarray]]" = Queue(maxsize=10)

        self._stop_flag = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    # ------------- public API -------------

    def run(self, source=0):
        """
        Run full pipeline on webcam (source=0) or video file path.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        writer = None
        if self.save_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(self.save_path, fourcc, fps, (w, h))

        # start worker thread (inference pipeline)
        self._worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._worker_thread.start()

        frame_idx = 0
        prev_time = time.perf_counter()
        try:
            while not self._stop_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                # push frame into queue (non-blocking)
                try:
                    self._frame_queue.put_nowait((frame_idx, frame.copy()))
                except Exception:
                    # queue full, drop frame
                    pass

                # try to get processed result
                try:
                    idx_out, frame_out = self._result_queue.get_nowait()
                except Empty:
                    frame_out = frame

                # FPS measure (rough)
                now = time.perf_counter()
                dt = now - prev_time
                prev_time = now
                fps = 1.0 / dt if dt > 0 else 0.0
                cv2.putText(
                    frame_out,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                if writer is not None:
                    writer.write(frame_out)

                if self.display:
                    cv2.imshow("Edge AI Video Engine", frame_out)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break

                frame_idx += 1

        finally:
            self._stop_flag.set()
            cap.release()
            if writer is not None:
                writer.release()
            if self.display:
                cv2.destroyAllWindows()
            if self._worker_thread is not None:
                self._worker_thread.join()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
    ) -> Tuple[np.ndarray, List[FusedObject]]:
        """
        Single-threaded, deterministic processing for unit tests / offline use.
        Pipeline:
            - If frame_idx % detect_every_n == 0 → run detector + tracker.update_with_detections
            - Else → tracker.track_only
            - Then fuse detections and tracks.
        Returns annotated frame and fused objects.
        """
        run_detector = (frame_idx % self.detect_every_n == 0)

        detections: List[Detection] = []
        if run_detector:
            detections = self.detector(frame_bgr)

        if run_detector:
            tracks: List[Track] = self.tracker.update_with_detections(frame_bgr, detections)
        else:
            tracks = self.tracker.track_only(frame_bgr)

        fused = fuse_detections_and_tracks(detections, tracks, iou_thres=self.tracker.drift_iou_thres)

        # convert fused to TrackViz for drawing
        track_viz_list: List[TrackViz] = []
        for fo in fused:
            if fo.track_id is None:
                # still draw but ID=-1
                track_viz_list.append(
                    TrackViz(
                        track_id=-1,
                        bbox=(fo.x1, fo.y1, fo.x2, fo.y2),
                        score=fo.score,
                        cls=fo.cls,
                    )
                )
            else:
                track_viz_list.append(
                    TrackViz(
                        track_id=fo.track_id,
                        bbox=(fo.x1, fo.y1, fo.x2, fo.y2),
                        score=fo.score,
                        cls=fo.cls,
                    )
                )

        annotated = draw_tracks(frame_bgr, track_viz_list, class_names=self.class_names)
        return annotated, fused

    # ------------- worker thread -------------

    def _inference_worker(self):
        """
        Thread A: consumes frames, runs full pipeline, produces annotated frames.
        """
        while not self._stop_flag.is_set():
            try:
                frame_idx, frame_bgr = self._frame_queue.get(timeout=0.1)
            except Empty:
                continue

            annotated, fused = self.process_frame(frame_bgr, frame_idx)
            # fused currently sadece log/analiz için kullanılabilir (burada kuyruğa koymuyoruz)
            try:
                self._result_queue.put_nowait((frame_idx, annotated))
            except Exception:
                # result queue full, drop
                pass
