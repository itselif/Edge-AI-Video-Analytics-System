# inference/fusion.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional

import numpy as np

from inference.detector import Detection
from inference.tracker import Track


@dataclass
class FusedObject:
    """
    Detection + Tracker bilgilerini birleştiren çıktı.

    track_id :
        - Eşleşen bir tracker varsa o ID
        - Yeni obje ise None
    source :
        - "det+trk"  : IoU yüksek, detection + tracker birleşmiş
        - "det_only": Sadece detection var, track yok / drift
        - "trk_only": Sadece tracker (hiç detection yokken kullanılır)
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
    boxes1: (N, 4)  [x1, y1, x2, y2]
    boxes2: (M, 4)
    return: (N, M) IoU matrisi
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=float)

    x11 = boxes1[:, 0][:, None]
    y11 = boxes1[:, 1][:, None]
    x12 = boxes1[:, 2][:, None]
    y12 = boxes1[:, 3][:, None]

    x21 = boxes2[:, 0][None, :]
    y21 = boxes2[:, 1][None, :]
    x22 = boxes2[:, 2][None, :]
    y22 = boxes2[:, 3][None, :]

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
    iou = inter_area / union
    return iou


def fuse_detections_and_tracks(
    detections: Sequence[Detection],
    tracks: Sequence[Track],
    iou_thres: float = 0.5,
) -> List[FusedObject]:
    """
    Drift mantığı:

    - Eğer detection ve track arasında IoU >= iou_thres ise:
        → tek bir FusedObject döner, track_id = track.track_id, source="det+trk"
    - Eğer detection var ama IoU < iou_thres ise:
        → track DRIFT kabul edilir, fused çıktısında o track yer almaz,
          detection ise yeni obje (track_id=None, source="det_only") olur.
        (Bu, test_fusion_low_iou_triggers_drift testinin istediği davranış.)
    - Eğer hiç detection yoksa:
        → sadece tracker sonuçları "trk_only" olarak dönebilir.
    """

    dets = list(detections)
    trks = list(tracks)

    fused: List[FusedObject] = []

    # Hiç detection yoksa → sadece tracker sonuçlarını döndür
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

    # Detection varsa, drift kuralını uygulayacağız.
    # 1) IoU matrisi
    if len(trks) > 0:
        det_boxes = np.array(
            [[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=float
        )  # (Nd, 4)
        trk_boxes = np.array(
            [trk.bbox for trk in trks], dtype=float
        )  # (Nt, 4)
        ious = _bbox_iou(det_boxes, trk_boxes)  # (Nd, Nt)
    else:
        ious = np.zeros((len(dets), 0), dtype=float)

    Nd = len(dets)
    Nt = len(trks)

    matched_dets = set()
    matched_trks = set()

    # 2) Greedy matching (yüksek IoU'lar için)
    if Nd > 0 and Nt > 0:
        # ious[i, j] = det i vs trk j
        # Greedy: her seferinde en büyük IoU'yu al
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

            # Eşleşme: detection + track birleşsin
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

            # Bu satır ve sütunu tekrar seçilmesin diye kapat
            ious_copy[d_idx, :] = -1.0
            ious_copy[:, t_idx] = -1.0

    # 3) Eşleşmeyen detection'lar → yeni obje (track_id=None)
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

    # 4) Eşleşmeyen track'ler:
    #    Burada DRIFT varsayımını uyguluyoruz:
    #    - Eğer detection vardı (bu fonksiyonda var), IoU eşik altı demektir,
    #      drift olmuş track'leri fused çıktılarına KOYMUYORUZ.
    #
    #    Eğer "drift olsa da track'i gösterelim" denseydi, aşağıdaki blokta
    #    "trk_only" eklerdik. Ama test_fusion_low_iou_triggers_drift tam tersini istiyor.
    #
    # ÖNEMLİ: Eğer hiç detection yoksa, fonksiyon başında zaten "trk_only" ile döndük.

    return fused
