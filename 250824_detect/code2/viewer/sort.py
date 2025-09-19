# sort.py
# 간단 SORT (IoU 기반 그리디 매칭 버전) - numpy만 필요
# update(dets: Nx5 [x1,y1,x2,y2,score]) -> Mx5 [x1,y1,x2,y2,track_id]
# 파라미터: max_age(유지 프레임), min_hits(안정화 프레임), iou_threshold(매칭 임계)

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, (a[2]-a[0]) * (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0]) * (b[3]-b[1]))
    union = max(1e-6, area_a + area_b - inter)
    return inter / union

@dataclass
class Track:
    tid: int
    bbox: np.ndarray  # [x1,y1,x2,y2] float
    age: int = 0              # 총 프레임 경과
    time_since_update: int = 0
    hits: int = 0
    hit_streak: int = 0

    def update_with_det(self, det_box: np.ndarray):
        self.bbox = det_box.astype(np.float32)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def mark_missed(self):
        self.time_since_update += 1
        self.hit_streak = 0

class Sort:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self):
        self.tracks.clear()
        self._next_id = 1

    def _greedy_match(self, tracks: list[Track], dets: np.ndarray) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
        """
        그리디 IoU 매칭: 항상 가장 높은 IoU를 우선 매칭 (간단·빠름)
        반환: matches[(ti, di, iou)], unmatched_track_idxs, unmatched_det_idxs
        """
        if len(tracks) == 0 or dets.shape[0] == 0:
            return [], set(range(len(tracks))), set(range(dets.shape[0]))

        # IoU 행렬
        iou_mat = np.zeros((len(tracks), dets.shape[0]), dtype=np.float32)
        for ti, trk in enumerate(tracks):
            for di, db in enumerate(dets[:, :4]):
                iou_mat[ti, di] = iou_xyxy(trk.bbox, db)

        used_t, used_d = set(), set()
        matches = []

        # 가능한 모든 (ti,di)를 IoU 내림차순으로 훑으면서 그리디 매칭
        cand = [(iou_mat[ti, di], ti, di) for ti in range(iou_mat.shape[0]) for di in range(iou_mat.shape[1])]
        cand.sort(key=lambda x: x[0], reverse=True)
        for iou, ti, di in cand:
            if iou < self.iou_threshold:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            matches.append((ti, di, float(iou)))

        unmatched_t = set(range(len(tracks))) - used_t
        unmatched_d = set(range(dets.shape[0])) - used_d
        return matches, unmatched_t, unmatched_d

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        dets: Nx5 (x1,y1,x2,y2,score) 또는 Nx4 (score 생략 가능)
        반환: Kx5 (x1,y1,x2,y2,tid) - 이번 프레임에서 매칭된 트랙만 반환
        """
        # 정규화
        if dets is None or len(dets) == 0:
            dets = np.empty((0, 5), dtype=np.float32)
        else:
            dets = np.asarray(dets, dtype=np.float32)
            if dets.shape[1] == 4:
                dets = np.hstack([dets, np.ones((dets.shape[0], 1), dtype=np.float32)])

        # 1) 기존 트랙들을 "미스"로 표시(나중에 매칭되면 update가 덮어씀)
        for trk in self.tracks:
            trk.age += 1
            trk.mark_missed()

        # 2) 매칭
        matches, unmatched_t, unmatched_d = self._greedy_match(self.tracks, dets)

        # 3) 매칭된 트랙 갱신
        for ti, di, _ in matches:
            self.tracks[ti].update_with_det(dets[di, :4])

        # 4) 매칭 안 된 detection → 새 트랙 생성
        for di in unmatched_d:
            t = Track(tid=self._next_id, bbox=dets[di, :4].astype(np.float32))
            t.hits = 1; t.hit_streak = 1; t.time_since_update = 0
            self.tracks.append(t)
            self._next_id += 1

        # 5) 오래된 트랙 제거
        alive = []
        for t in self.tracks:
            if t.time_since_update <= self.max_age:
                alive.append(t)
        self.tracks = alive

        # 6) 이번 프레임에서 "업데이트된" 트랙만 반환
        outs = []
        for t in self.tracks:
            if t.time_since_update == 0:
                x1, y1, x2, y2 = t.bbox.tolist()
                outs.append([x1, y1, x2, y2, float(t.tid)])
        return np.array(outs, dtype=np.float32)
