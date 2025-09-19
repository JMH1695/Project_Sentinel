# car_infer.py
"""
차량 검출 래퍼 (Ultralytics YOLO)
- viewer.py와 바로 호환되도록 .model 속성 그대로 노출
- __init__(model_path, conf, imgsz, device, use_half, max_det) 시그니처 유지
- car_det.model.predict(...) 또는 car_det.predict(...) 둘 다 가능
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
            # 사용자가 직접 지정한 경우 (예: "cuda:1")
            return pref if torch.cuda.is_available() else "cpu"
        if pref.lower() == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(pref, int):
        return f"cuda:{pref}" if torch.cuda.is_available() else "cpu"
    # 기타 입력은 CPU로 폴백
    return "cpu"


class CarDetector:
    """
    차량 검출기 래퍼
    - self.model: Ultralytics YOLO 객체 (viewer에서 직접 predict 호출 가능)
    - self.device, self.half: 뷰어에서 참조할 수 있도록 노출
    - predict(frame, **overrides): self 설정값 기반 간편 호출
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

        # half precision (GPU일 때만)
        # 주: ultralytics는 predict 호출에 half=True 전달해도 동작하므로,
        #     여기선 플래그만 저장하고 predict에서 전달.
        #     (가중치 자체를 half로 cast하지는 않음)
        # 필요시 아래 주석 해제:
        # if self.half:
        #     for m in self.model.model.modules():
        #         if hasattr(m, "half"):
        #             m.half()

    def predict(self,
                frame: np.ndarray,
                *,
                conf: Optional[float] = None,
                imgsz: Optional[int] = None,
                iou: Optional[float] = None,
                max_det: Optional[int] = None,
                agnostic_nms: Optional[bool] = None,
                verbose: bool = False):
        """
        간편 예측 호출 (viewer에서 self.model.predict 대신 이걸 써도 됨)
        - frame: BGR numpy 배열(OpenCV 프레임)
        - 나머지 파라미터는 없으면 __init__ 값/기본값 사용
        - 반환: ultralytics 결과의 첫 번째 요소 (single-image)
        """
        return self.model.predict(
            frame,
            conf=self._val(conf, self.conf),
            imgsz=self._val(imgsz, self.imgsz),
            device=self.device,
            half=self.half,
            iou=iou,                     # None이면 ultralytics 기본 사용
            max_det=self._val(max_det, self.max_det),
            agnostic_nms=agnostic_nms,   # None이면 ultralytics 기본
            verbose=verbose
        )[0]

    def warmup(self, shape=(640, 640, 3)):
        """
        초기 한두 프레임 지연 줄이기 위한 워밍업.
        - shape: (H, W, 3) BGR dummy
        """
        dummy = np.zeros(shape, dtype=np.uint8)
        _ = self.predict(dummy, verbose=False)

    @staticmethod
    def _val(value, default):
        return default if value is None else value
