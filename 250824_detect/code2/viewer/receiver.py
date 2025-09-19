# receiver.py
import cv2
import time
from typing import Generator, Optional, Tuple

class RTMPReceiver:
    """
    RTMP/RTSP/파일/웹캠 입력을 통합 처리하는 수신기.
    - FFmpeg 백엔드 강제 사용 (RTMP 안정성)
    - 자동 재연결(지수 백오프)
    - 프레임 제너레이터 제공 (for frame in receiver.frames():)
    """
    def __init__(
        self,
        url: str,
        backend: int = cv2.CAP_FFMPEG,
        retry: bool = True,
        max_retries: int = -1,              # -1: 무제한
        backoff: Tuple[float, float] = (1.0, 30.0),  # (초기, 최대)
        reconnect_on_empty: bool = True,    # 빈 프레임 시 재연결
        verbose: bool = True
    ):
        self.url = url
        self.backend = backend
        self.retry = retry
        self.max_retries = max_retries
        self.backoff = backoff
        self.reconnect_on_empty = reconnect_on_empty
        self.verbose = verbose

        self.cap: Optional[cv2.VideoCapture] = None
        self._opened = False

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _connect(self) -> bool:
        # 기존 cap 정리
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        # 새 연결
        self.cap = cv2.VideoCapture(self.url, self.backend)
        self._opened = bool(self.cap.isOpened())
        if self._opened:
            self._log("✅ RTMP 스트림 연결 성공")
        else:
            self._log("❌ RTMP 스트림 열기 실패")
        return self._opened

    def open(self) -> "RTMPReceiver":
        """
        최초 연결 시도. 실패하면 retry 정책에 따라 재시도.
        """
        attempt = 0
        delay = self.backoff[0]
        while True:
            if self._connect():
                return self
            attempt += 1
            if not self.retry or (self.max_retries >= 0 and attempt > self.max_retries):
                raise RuntimeError("RTMP 스트림 연결에 실패했습니다.")
            self._log(f"⏳ 재연결 대기 {delay:.1f}s (시도 {attempt})")
            time.sleep(delay)
            delay = min(delay * 2, self.backoff[1])

    def is_open(self) -> bool:
        return self._opened and self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        """
        단발 프레임 읽기. 필요하면 재연결.
        """
        if not self.is_open():
            # 열린 상태가 아니면 재연결 시도
            if not self.retry:
                return False, None
            self._log("🔁 연결이 닫혀 재연결 시도")
            self.open()

        ret, frame = self.cap.read() if self.cap is not None else (False, None)

        # 빈 프레임/실패 처리
        if not ret or frame is None or (hasattr(frame, "size") and frame.size == 0):
            self._log("⚠️ 프레임 수신 실패")
            if self.reconnect_on_empty:
                return self._reconnect_and_read()
            return False, None

        return True, frame

    def _reconnect_and_read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        """
        재연결 후 1프레임 재시도
        """
        if not self.retry:
            return False, None

        attempt = 0
        delay = self.backoff[0]
        while True:
            attempt += 1
            self._log(f"🔌 프레임 실패 → 재연결 시도 {attempt}")
            if self._connect():
                ret, frame = self.cap.read() if self.cap is not None else (False, None)
                if ret and frame is not None and frame.size != 0:
                    return True, frame
            if self.max_retries >= 0 and attempt >= self.max_retries:
                return False, None
            self._log(f"⏳ 재연결 대기 {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, self.backoff[1])

    def frames(self) -> Generator["cv2.Mat", None, None]:
        """
        무한 프레임 제너레이터.
        사용 예: for frame in receiver.frames(): ...
        """
        # 최초 오픈 (필요 시)
        if not self.is_open():
            self.open()

        while True:
            ret, frame = self.read()
            if not ret or frame is None:
                # 더 이상 재시도 불가
                self._log("❌ 프레임 수신 종료")
                break
            yield frame

    def close(self) -> None:
        self._opened = False
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None
        self._log("🧹 수신기 자원 정리 완료")

    # 컨텍스트 매니저 지원 (with 문)
    def __enter__(self) -> "RTMPReceiver":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
