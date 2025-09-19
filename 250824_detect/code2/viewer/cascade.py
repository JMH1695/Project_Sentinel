# cascade.py
"""
캐스케이드(차량 ROI 기반 번호판 탐지) 유틸
- 차량 박스에서 ROI 추출(+마진) → 복제패딩으로 줌아웃 → (선택)RGB 변환
- ROI 좌표계에서 나온 번호판 박스를 글로벌 좌표로 되돌리는 매핑 제공
"""

from __future__ import annotations
from typing import Iterable, Iterator, List, Tuple, Optional
import numpy as np
import cv2
from utils import safe_crop, safe_pad_box  # 기존 utils.py 사용


# ========== ROI 생성 ==========

def build_cascade_roi(
    frame: np.ndarray,
    car_box_xyxy: Iterable[float],
    *,
    margin_ratio: float = 0.20,
    zoom_out: float = 1.5,
    rgb: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], int, int]:
    """
    차량 박스 → ROI 추출(+마진) → 복제패딩으로 줌아웃 → (선택) RGB 변환

    반환:
      roi_canvas            : 패딩/줌아웃이 적용된 ROI 이미지 (모델 입력용)
      roi_rect_global_xyxy  : 원본 프레임 기준 ROI 영역(마진 적용 후)의 글로벌 좌표 (x1g,y1g,x2g,y2g)
      pad_x, pad_y          : 줌아웃 시 추가된 좌우/상하 패딩 픽셀(글로벌 매핑에 필요)
    """
    roi, roi_rect = safe_crop(frame, car_box_xyxy, margin_ratio=margin_ratio)
    if roi is None or roi.size == 0 or roi_rect is None:
        # 빈 ROI라도 일관된 타입 반환
        return np.empty((0, 0, 3), dtype=frame.dtype), (0, 0, 0, 0), 0, 0

    # 색공간 변환
    roi_in = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) if rgb else roi

    # 복제패딩으로 컨텍스트 보강(줌아웃)
    h, w = roi_in.shape[:2]
    if zoom_out and zoom_out > 1.0:
        long_edge = int(max(h, w) * float(zoom_out))
        pad_y = max(0, (long_edge - h) // 2)
        pad_x = max(0, (long_edge - w) // 2)
        roi_canvas = cv2.copyMakeBorder(roi_in, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
    else:
        pad_x = pad_y = 0
        roi_canvas = roi_in

    x1g, y1g, x2g, y2g = roi_rect
    return roi_canvas, (int(x1g), int(y1g), int(x2g), int(y2g)), int(pad_x), int(pad_y)


def make_rois_for_tracks(
    frame: np.ndarray,
    tracked_cars: Iterable[dict],
    *,
    margin_ratio: float = 0.20,
    zoom_out: float = 1.5,
    rgb: bool = True,
) -> Iterator[Tuple[int, np.ndarray, Tuple[int, int, int, int], int, int]]:
    """
    여러 차량 트랙에 대해 ROI 캔버스를 생성하는 제너레이터.

    입력:
      tracked_cars: [{'tid': int, 'box': [x1,y1,x2,y2], ...}, ...] 형태

    yield:
      (tid, roi_canvas, roi_rect_global_xyxy, pad_x, pad_y)
    """
    for tc in tracked_cars:
        tid = int(tc["tid"])
        car_box = tc["box"]
        roi_canvas, roi_rect_g, pad_x, pad_y = build_cascade_roi(
            frame, car_box, margin_ratio=margin_ratio, zoom_out=zoom_out, rgb=rgb
        )
        if roi_canvas.size == 0:
            continue
        yield tid, roi_canvas, roi_rect_g, pad_x, pad_y


# ========== 글로벌 좌표 매핑 ==========

def map_roi_box_to_global(
    roi_box_xyxy: Iterable[float],
    roi_rect_global_xyxy: Tuple[int, int, int, int],
    pad_x: int,
    pad_y: int,
    frame_shape_hw: Tuple[int, int],
) -> Optional[List[int]]:
    """
    ROI 좌표(줌아웃 캔버스 기준)에서 나온 xyxy 박스를 원본 프레임 글로벌 좌표로 변환.

    roi_box_xyxy            : ROI 캔버스(패딩 포함) 기준 번호판 박스
    roi_rect_global_xyxy    : 원본 프레임 기준 ROI(마진 적용)의 글로벌 사각형
    pad_x, pad_y            : ROI 캔버스에 추가된 좌우/상하 패딩
    frame_shape_hw          : 원본 프레임 (H, W)

    반환:
      글로벌 xyxy (이미지 경계로 안전 클램프) 또는 None(무효)
    """
    bx1, by1, bx2, by2 = map(int, roi_box_xyxy)
    x1g, y1g, _, _ = roi_rect_global_xyxy

    # 패딩을 제거하여 ROI 원좌표로 되돌린 뒤, 글로벌로 오프셋
    gx1 = x1g + (bx1 - pad_x)
    gy1 = y1g + (by1 - pad_y)
    gx2 = x1g + (bx2 - pad_x)
    gy2 = y1g + (by2 - pad_y)

    H, W = frame_shape_hw
    g = safe_pad_box([gx1, gy1, gx2, gy2], W, H, margin_ratio=0.0)
    return g
