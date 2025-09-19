# utils.py
"""
공통 유틸 모음
- 텍스트/박스 그리기
- 안전 패딩/크롭
- 업스케일
- IoU/NMS
- aHash/dHash 및 해밍거리
- 비디오 라이터 경로 보장
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
import os
import datetime
import numpy as np
import cv2


# ========== 시각화 ==========
def put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.6,
    thickness: int = 2,
    color_fg: Tuple[int, int, int] = (255, 255, 255),
    color_bg: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """가독성 좋은 이중 레이어 텍스트"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color_bg, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color_fg, thickness, cv2.LINE_AA)


def draw_box(
    img: np.ndarray,
    xyxy: Iterable[float],
    color: Tuple[int, int, int] = (230, 70, 70),
    label: Optional[str] = None,
) -> None:
    """테두리 + 선택 라벨"""
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    if label:
        put_text(img, label, (x1 + 2, max(18, y1 - 6)), scale=0.55, thickness=1)


# ========== 박스/크롭 ==========
def safe_pad_box(
    box: Iterable[float],
    img_w: int,
    img_h: int,
    margin_ratio: float = 0.8,
) -> Optional[List[int]]:
    """박스 주변 여유(margin_ratio 비율) 주고 이미지 경계로 클램프"""
    x1, y1, x2, y2 = map(float, box)
    if margin_ratio and margin_ratio > 0:
        w = x2 - x1
        h = y2 - y1
        x1 -= w * margin_ratio
        x2 += w * margin_ratio
        y1 -= h * margin_ratio
        y2 += h * margin_ratio
    x1 = int(max(0, min(x1, img_w - 1)))
    x2 = int(max(0, min(x2, img_w)))
    y1 = int(max(0, min(y1, img_h - 1)))
    y2 = int(max(0, min(y2, img_h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def safe_crop(
    frame: np.ndarray,
    box: Iterable[float],
    margin_ratio: float = 0.0,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """여유를 적용한 안전 크롭과 최종 글로벌 박스 반환"""
    h, w = frame.shape[:2]
    b = safe_pad_box(box, w, h, margin_ratio)
    if b is None:
        return None, None
    x1, y1, x2, y2 = b
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def maybe_upscale(img: np.ndarray, scale: int) -> np.ndarray:
    """scale>1이면 보간 업스케일"""
    if not scale or scale <= 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


# ========== NMS/IoU ==========
def iou(box: Iterable[float], boxes: np.ndarray) -> np.ndarray:
    """단일 box vs 여러 boxes IoU (vectorized)"""
    x1, y1, x2, y2 = box
    xx1 = np.maximum(x1, boxes[:, 0])
    yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2])
    yy2 = np.minimum(y2, boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = np.maximum(area1 + area2 - inter, 1e-6)
    return inter / union


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """간단 NMS (인덱스 keep 리스트 반환)"""
    if len(boxes) == 0:
        return []
    idxs = np.argsort(-scores)
    keep: List[int] = []
    while len(idxs) > 0:
        i = int(idxs[0])
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep


# ========== 해시/해밍 ==========
def ahash64(img: np.ndarray) -> int:
    """8x8 average hash → 64-bit int"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = (small >= mean).astype(np.uint8).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return int(val)


def dhash64(img: np.ndarray) -> int:
    """8x8 dHash(gradient) → 64-bit int"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)  # 9x8 → 좌우 비교 8x8
    diff = (small[:, 1:] > small[:, :-1]).astype(np.uint8).flatten()
    val = 0
    for b in diff:
        val = (val << 1) | int(b)
    return int(val)


def hamming64(a: int, b: int) -> int:
    """64-bit 해밍 거리"""
    return int(bin(a ^ b).count("1"))


# ========== 비디오 저장 경로 ==========
def ensure_video_writer_path(
    path_like: str,
    width: int,
    height: int,
    fps: float = 30.0,
) -> Tuple[Optional[cv2.VideoWriter], Optional[str]]:
    """
    - path_like가 폴더이거나 확장자가 .mp4가 아니면 폴더로 간주하고 타임스탬프 파일명 생성
    - 반환: (VideoWriter, 실제 저장 경로)
    """
    if not path_like:
        return None, None

    path_like = os.path.normpath(path_like)
    if os.path.isdir(path_like) or os.path.splitext(path_like)[1].lower() != ".mp4":
        os.makedirs(path_like if os.path.isdir(path_like) else os.path.dirname(path_like) or ".", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = path_like if os.path.isdir(path_like) else (os.path.dirname(path_like) or ".")
        path = os.path.join(out_dir, f"result_{ts}.mp4")
    else:
        os.makedirs(os.path.dirname(path_like) or ".", exist_ok=True)
        path = path_like

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (int(width), int(height)))
    return writer, path
