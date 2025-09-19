# plate_infer.py
"""
번호판 검출 래퍼 (Ultralytics YOLO)
- viewer.py와 호환되도록 .model 속성 그대로 노출
- __init__(model_path, conf, imgsz, device, use_half, max_det) 시그니처 유지
- plate_det.model.predict(...) 또는 plate_det.predict(...) 둘 다 가능
"""

from typing import Union, Optional
import numpy as np
import torch
from ultralytics import YOLO


def resolve_device(pref: Union[str, int] = "auto") -> str:
    """
    device 문자열 결정:
      - "auto"  : GPU 있으면 cuda:0, 없으면 cpu
      - "cpu"   : cpu
      - int     : 해당 인덱스의 cuda (예: 0 -> "cuda:0")
      - "cuda:N": 그대로 사용
    """
    if isinstance(pref, str):
        if pref.lower() == "cpu":
            return "cpu"
        if pref.startswith("cuda:"):
            return pref if torch.cuda.is_available() else "cpu"
        if pref.lower() == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(pref, int):
        return f"cuda:{pref}" if torch.cuda.is_available() else "cpu"
    return "cpu"


class PlateDetector:
    """
    번호판 검출기 래퍼
    - self.model: Ultralytics YOLO 객체 (viewer에서 직접 predict 호출 가능)
    - self.device, self.half: 뷰어에서 참조할 수 있도록 노출
    - predict(frame_or_roi, **overrides): self 설정값 기반 간편 호출
    """

    def __init__(self,
                 model_path: str,
                 conf: float = 0.7,
                 imgsz: int = 640,
                 device: Union[str, int] = "auto",
                 use_half: bool = True,
                 max_det: int = 100):
        self.model_path = model_path
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)

        self.device = resolve_device(device)
        self.half = bool(use_half and self.device.startswith("cuda"))

        # 모델 로드 및 디바이스 배치
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # half precision은 predict 호출에서 half=True로 전달 (가중치 자체 캐스팅은 생략)
        # 필요 시 아래 주석 해제 가능:
        # if self.half:
        #     for m in self.model.model.modules():
        #         if hasattr(m, "half"):
        #             m.half()

    def predict(self,
                frame_or_roi: np.ndarray,
                *,
                conf: Optional[float] = None,
                imgsz: Optional[int] = None,
                iou: Optional[float] = None,
                max_det: Optional[int] = None,
                agnostic_nms: Optional[bool] = None,
                verbose: bool = False):
        """
        간편 예측 호출
        - frame_or_roi: BGR 또는 RGB numpy 배열(OpenCV/ROI)
          (색공간은 호출부 정책에 따름; 여기선 그대로 전달)
        - 반환: ultralytics 결과의 첫 번째 요소(single-image)
        """
        return self.model.predict(
            frame_or_roi,
            conf=self._val(conf, self.conf),
            imgsz=self._val(imgsz, self.imgsz),
            device=self.device,
            half=self.half,
            iou=iou,
            max_det=self._val(max_det, self.max_det),
            agnostic_nms=agnostic_nms,
            verbose=verbose
        )[0]

    def warmup(self, shape=(640, 640, 3)):
        """
        초기 한두 프레임 지연 줄이기 위한 워밍업.
        - shape: (H, W, 3) dummy 배열
        """
        dummy = np.zeros(shape, dtype=np.uint8)
        _ = self.predict(dummy, verbose=False)

    @staticmethod
    def _val(value, default):
        return default if value is None else value
