# viewer.py  — 불법 주정차 단속 드론 뷰어 (캐스케이드 안정화 + SORT + 5초 조건 + OCR + 중복 캐시)
# - ocr_infer.py 통합 버전 (OcrEngine 내장)

import os
import time
import datetime
import cv2
import numpy as np
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import io
try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None  # MQTT 안 쓰면 무시
from sort import Sort  # SORT 추적기
from receiver import RTMPReceiver  # 드론 RTMP 수신기
from winforms_bridge import TcpJsonSender, TcpConfig, make_violation_event  # 윈폼 전송
from cascade import make_rois_for_tracks, map_roi_box_to_global
from utils import (put_text, draw_box, safe_pad_box, safe_crop, maybe_upscale, nms_numpy, ahash64, dhash64, hamming64, ensure_video_writer_path,)

# ================================
# [A] 자주 바꾸는 설정 (한 곳에 모음)
# ================================

# [A0] RTMP/송신 설정 (드론 영상 데이터 받아오는 용도)
USE_RTMP   = False
RTMP_URL   = "rtmp://54.180.119.91/live/stream2"   # 드론 송출 URL

SEND_TO_WINFORMS = True # TCP 쓸때 True
TCP_HOST   = "127.0.0.1"
TCP_PORT   = 5059

HTTP_HOST = "127.0.0.1"   # WebView2가 접속할 MJPEG 서버
HTTP_PORT = 8099
JPEG_QUALITY = 70         # 60~80 권장

# [A0-1] MQTT 설정 (드론 위치 데이터 등 받아오는 용도)
USE_MQTT = False
MQTT_HOST = "52.79.237.147"
MQTT_PORT = 1883
MQTT_TOPIC_VEHICLE = "drone/vehicle"
MQTT_TOPIC_STATUS  = "drone/status"  # 선택

# [A1] 입출력 경로/모드 -----------------q----------
video_path = r"c:\Users\user\Desktop\detect\data\a.avi"  # 입력 영상 경로
mode = "both"  # "car" | "plate" | "both"   # plate 모드면 cascade 자동 OFF

car_model_path   = r"c:\Users\user\Desktop\detect\model\car_detect_V5.pt"  # 차량 모델(.pt)
plate_model_path = r"c:\Users\user\Desktop\detect\model\plate.pt"          # 번호판 모델(.pt)

resize_display   = (1280, 720)   # 표시 해상도 (None이면 원본)
save_video_path  = r"c:\Users\user\Desktop\detect\code2\viewer\save_video"  # 결과 mp4 경로 또는 폴더

save_crops   = True                                   # 번호판 크롭 저장 여부
crops_dir    = r"c:\Users\user\Desktop\detect\code2\viewer\crop_plate"  # 저장 폴더
jpeg_quality = 95                                     # 크롭 JPEG 품질(1~100)

# [A2] 모델/추론 옵션 -----------------------------
use_device = "auto"    # "auto" | "cpu" | 정수 GPU index(0 등)
use_half   = True      # GPU일 때 half 사용
car_conf, car_imgsz, car_maxdet       = 0.7, 640, 100
plate_conf, PLATE_IMG_SZ, plate_maxdet = 0.7, 640, 100  # 비캐스케이드용
PLATE_CASCADE_IMG_SZ = 512  # 캐스케이드(ROI) 추론 입력크기

car_iou   = 0.60       # car 모델 내부 NMS IoU
plate_iou = 0.60       # plate 모델 내부 NMS IoU
agnostic_nms = True    # 멀티클래스일 때 클래스 무시 NMS
plate_rgb = True       # 번호판 모델 입력을 RGB로 줄지 여부(True=RGB)

# [A3] 캐스케이드/ROI/표시 ------------------------
cascade_plate = True      # 번호판은 차량 ROI 안에서만 탐지
ROI_MARGIN = 0.20         # 차량 ROI 여유(외곽 컨텍스트 포함 비율)
render_plate_boxes = True # 번호판 미리보기 박스/점수 표시(ready 전 단계)
HIDE_PLATE_SCORE_WHEN_READY = True  # ready면 plate 0.xx 텍스트 숨김(겹침 방지)

# [A4] 추적/위반 판정 -----------------------------
VIOLATION_SECONDS = 5.0   # 같은 차량이 '추론된 프레임' 기준 5초 이상 잡혔을 때 ready
SORT_IOU = 0.3
SORT_MIN_HITS = 3
frame_skip_n = 1          # n>1이면 n프레임 중 1회만 추론(체감 FPS↑, 5초는 '추론 프레임' 기준)

# [A5] OCR/크롭 옵션 ------------------------------
# (ocr_infer.py 통합: 아래에 OcrEngine 클래스를 정의하고 사용)
from lpr_ocr_viewer import LPRViewerOCR
LPR_MODEL_PATH = r"c:\Users\user\Desktop\detect\model\model.tflite"
LPR_LABEL_PATH = r"c:\Users\user\Desktop\detect\model\label.name"
OCR_RGB = False                 # 예전과 동일: BGR 그대로(모델이 그걸 기대)
OCR_MIN_CONF = 0.50
ocr_upscale  = 2                # 예전과 동일: 2배 업스케일
min_crop_wh  = (16, 8)   # (w,h) 미만이면 저장/표시는 생략
crop_margin  = 0.1       # 저장 썸네일용 마진(번호판 주변 여유)
ocr_crop_margin = 0.1    # OCR 입력용 마진(글자만 깔끔히)
POST_NMS_IOU_PLATE = 0.5 # plate 후처리 NMS IoU(모델 NMS 뒤에 한 번 더)

# [A6] OCR 중복 캐시(같은 프레임 내) --------------
OCR_CACHE_BY_TID   = True
OCR_SIZE_BUCKET    = 8
OCR_AHASH_THRESH   = 6          # 예전과 동일
OCR_DHASH_THRESH   = 6          # 예전과 동일
OCR_RECALL_IF_LOW  = 0.60       # 예전과 동일

# =========================================
# [B] 덜 자주 바꾸는 설정/유틸 (아래는 건드릴 일 적음)
# =========================================

# plate 무검출 연속 프레임 수가 일정 이상이면 헬스체크/폴백
PLATE_HEALTH_N = 5
plate_nohit_count = 0

# MQTT → TCP 브릿지용 콜백
def _to_float(x, default=0.0):
    try: return float(x)
    except: return default
# 쓰기: _to_float(data.get("Battery"))

def _on_mqtt_connect(client, userdata, flags, rc):
    try:
        # 드론이 보내는 텔레메트리/차량 토픽 구독
        client.subscribe(MQTT_TOPIC_STATUS, qos=0)
        client.subscribe(MQTT_TOPIC_VEHICLE, qos=0)
        print(f"✅ MQTT 구독: {MQTT_TOPIC_STATUS}, {MQTT_TOPIC_VEHICLE}")
    except Exception as e:
        print(f"❌ MQTT subscribe 실패: {e}")

def _on_mqtt_message(sender, msg):
    """
    - drone/status  : 텔레메트리 그대로 또는 키 보정 후 TCP로 전송
    - drone/vehicle : 차량 이벤트 그대로 또는 키 보정 후 TCP로 전송
    """
    try:
        payload = msg.payload.decode("utf-8", "ignore")
        data = json.loads(payload)

        if msg.topic == MQTT_TOPIC_STATUS:
            # 키 보정(드론 쪽 키가 다를 수도 있으니 유연하게)
            status = {
                "Battery":   float(data.get("Battery",   data.get("battery",   data.get("batt", 0)))),
                "Altitude":  float(data.get("Altitude",  data.get("alt",      0))),
                "Latitude":  float(data.get("Latitude",  data.get("lat",      0))),
                "Longitude": float(data.get("Longitude", data.get("lon",      0))),
                "EngineStatus": str(data.get("EngineStatus", data.get("engine", ""))).upper()
            }
            # → WinForms DataReceiver.HandleDroneStatus()가 그대로 먹음
            if sender:
                sender.send(status)  # dict 그대로 보내면 TcpJsonSender가 JSON으로 직렬화

        elif msg.topic == MQTT_TOPIC_VEHICLE:
            # 이미 WinForms가 기대하는 스키마면 그대로,
            # 다르면 최소한 아래 필드로 맞춰서 보냄
            vehicle = {
                "vehicle_number": data.get("vehicle_number", data.get("plate_text", "")),
                "detected_time":  data.get("detected_time",  data.get("ts", "")),
                "is_illegal":     bool(data.get("is_illegal", True)),
                "confidence":     float(data.get("confidence", data.get("plate_conf", 0.0))),
                "latitude":       float(data.get("latitude",  0)),
                "longitude":      float(data.get("longitude", 0)),
                "image_path":     data.get("image_path", "")
            }
            if sender:
                sender.send(vehicle)

        else:
            # 필요 시 기타 토픽도 로깅
            print(f"ℹ️ MQTT 기타 토픽 수신: {msg.topic}")

    except Exception as e:
        print(f"⚠️ MQTT on_message 처리 오류: {e}")

# --- 간단 NMS ---
def _iou(box, boxes):
    x1, y1, x2, y2 = box
    xx1 = np.maximum(x1, boxes[:, 0]); yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2]); yy2 = np.minimum(y2, boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = np.maximum(area1 + area2 - inter, 1e-6)
    return inter / union

def nms_numpy(boxes, scores, iou_thres):
    if len(boxes) == 0: return []
    idxs = np.argsort(-scores); keep = []
    while len(idxs) > 0:
        i = idxs[0]; keep.append(i)
        if len(idxs) == 1: break
        ious = _iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep

# --- 해시 캐시용: aHash/dHash ---
def ahash64(img):
    """8x8 average hash -> 64bit int"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = (small >= mean).astype(np.uint8).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return int(val)

def dhash64(img):
    """8x8 dHash(gradient) -> 64bit int"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)  # 9x8 -> 좌우 비교 8x8
    diff = (small[:, 1:] > small[:, :-1]).astype(np.uint8).flatten()
    val = 0
    for b in diff:
        val = (val << 1) | int(b)
    return int(val)

def hamming64(a, b):
    return int(bin(a ^ b).count("1"))

# =========================================================
# [C] OCR 엔진 (ocr_infer.py 통합: TFLite OcrEngine 내장)
# =========================================================

# --- TFLite Interpreter 로드 (여러 경로 폴백) ---
try:
    # 윈도우/파이썬 3.11에서 안정적
    from tensorflow.lite.python.interpreter import Interpreter
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
    except Exception:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except Exception as e:
            raise ImportError(
                "TFLite Interpreter를 찾을 수 없음. tensorflow(>=2.x) 또는 tflite-runtime 필요."
            ) from e

def _load_labels(path: str):
    labels = []
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    labels.append(s)
    return labels

def _ctc_greedy_decode(logits: np.ndarray, labels, blank_idx: int = 0):
    logits = logits.astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits); probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-6, None)

    idx_seq = probs.argmax(axis=1).tolist()
    conf_seq = probs.max(axis=1)

    out_idx, conf_used = [], []
    last = None
    for i, idx in enumerate(idx_seq):
        if idx == blank_idx:
            last = None
            continue
        if last is not None and idx == last:
            continue
        out_idx.append(idx); conf_used.append(float(conf_seq[i])); last = idx

    out_chars = []
    num_classes = len(labels)
    for idx in out_idx:
        if 0 <= idx < num_classes:
            out_chars.append(labels[idx])
    text = "".join(out_chars)
    conf = float(np.mean(conf_used)) if conf_used else 0.0
    return text, conf

class OcrEngine:
    """
    TFLite 번호판 OCR 엔진
      - rgb: 입력을 RGB로 변환해서 넣을지 여부 (기본 False = BGR 유지)
      - predict_image(img_bgr): (text, conf) 반환
    """

    def __init__(self, model_path: str, label_path: str = None, rgb: bool = False, num_threads: int = 2):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"OCR 모델을 찾을 수 없음: {model_path}")
        self.rgb = bool(rgb)
        self.labels = _load_labels(label_path) if label_path else []
        self.blank_idx = 0
        self.interp = Interpreter(model_path=model_path, num_threads=int(num_threads))
        self.interp.allocate_tensors()
        self.input_details = self.interp.get_input_details()
        self.output_details = self.interp.get_output_details()
        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]

        in_shape = self.input_details[0]["shape"]
        if len(in_shape) == 4:
            self.in_h = int(in_shape[1] if in_shape[1] > 0 else 32)
            self.in_w = int(in_shape[2] if in_shape[2] > 0 else 128)
            self.in_c = int(in_shape[3] if in_shape[3] > 0 else 3)
        elif len(in_shape) == 3:
            self.in_h = int(in_shape[0] if in_shape[0] > 0 else 32)
            self.in_w = int(in_shape[1] if in_shape[1] > 0 else 128)
            self.in_c = int(in_shape[2] if in_shape[2] > 0 else 3)
        else:
            self.in_h, self.in_w, self.in_c = 32, 128, 3
        self.in_dtype = self.input_details[0]["dtype"]

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("빈 이미지가 들어왔어.")
        img = img_bgr
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        if self.in_dtype == np.float32:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(self.in_dtype)
        if img.ndim == 2:
            img = img[..., None]
        if img.shape[-1] != self.in_c:
            if img.shape[-1] == 1 and self.in_c == 3:
                img = np.repeat(img, 3, axis=-1)
            elif img.shape[-1] == 3 and self.in_c == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
            else:
                img = img[..., :self.in_c]
        img = np.expand_dims(img, axis=0)
        return img

    def predict_image(self, img_bgr: np.ndarray):
        inp = self._preprocess(img_bgr)
        try:
            self.interp.resize_tensor_input(self.input_index, inp.shape)
            self.interp.allocate_tensors()
        except Exception:
            pass
        self.interp.set_tensor(self.input_index, inp)
        self.interp.invoke()
        out = self.interp.get_tensor(self.output_index)
        out_np = np.array(out)
        if out_np.ndim == 3 and out_np.shape[0] == 1:
            out_np = out_np[0]

        text, conf = "", 0.0
        if out_np.ndim == 2:
            text, conf = _ctc_greedy_decode(out_np, self.labels, blank_idx=self.blank_idx)
        elif out_np.ndim == 1:
            idxs = out_np.astype(int).tolist()
            chars = []
            for idx in idxs:
                if idx == self.blank_idx:
                    continue
                if 0 <= idx < len(self.labels):
                    if chars and self.labels[idx] == chars[-1]:
                        continue
                    chars.append(self.labels[idx])
            text = "".join(chars)
            conf = 1.0 if text else 0.0
        else:
            text, conf = "", 0.0
        return text, conf

# =========================================================
# [D] 메인 파이프라인
# =========================================================

def _get_car_box(tracked_cars, tid):
    for tc in tracked_cars:
        if tc['tid'] == tid:
            return [int(v) for v in tc['box']]
    return [0, 0, 0, 0]
class FrameBuffer:
    def __init__(self, jpeg_quality=70):
        self.lock = threading.Lock()
        self.jpeg_quality = int(jpeg_quality)
        self.last_jpeg = None

    def update(self, frame_bgr):
        if frame_bgr is None:
            return
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        with self.lock:
            self.last_jpeg = buf.tobytes()

    def get_jpeg(self):
        with self.lock:
            return self.last_jpeg

class _MjpegHandler(BaseHTTPRequestHandler):
    frame_buffer: FrameBuffer = None
    boundary = b"--frame"
    page_html = b"""<!doctype html><html><head><meta charset='utf-8'>
<title>Viewer MJPEG</title>
<style>body{margin:0;background:#111;display:flex;align-items:center;justify-content:center;height:100vh}img{max-width:100%;max-height:100vh}</style>
</head><body><img src="/video" alt="stream"></body></html>"""

    def log_message(self, fmt, *args): return

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.page_html)
            return

        if self.path.startswith("/snapshot"):
            jpg = self.frame_buffer.get_jpeg()
            if jpg is None:
                self.send_error(503, "No Frame")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
            return

        if self.path.startswith("/video"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = self.frame_buffer.get_jpeg()
                    if jpg is None:
                        time.sleep(0.02)
                        continue
                    self.wfile.write(self.boundary + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(1.0/30.0)  # ~30fps
            except (BrokenPipeError, ConnectionResetError):
                return

        else:
            self.send_error(404, "Not Found")

class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class MjpegServer:
    def __init__(self, host, port, frame_buffer: FrameBuffer):
        self.host = host; self.port = int(port)
        self.fb = frame_buffer
        self.httpd = None
        self.thread = None

    def start(self):
        _MjpegHandler.frame_buffer = self.fb
        self.httpd = _ThreadingHTTPServer((self.host, self.port), _MjpegHandler)
        def _serve():
            try:
                self.httpd.serve_forever(poll_interval=0.2)
            except Exception:
                pass
        self.thread = threading.Thread(target=_serve, daemon=True)
        self.thread.start()
        print(f"[MJPEG] http://{self.host}:{self.port}/  (WebView2 여기 열어)")
        return self

    def stop(self):
        try:
            if self.httpd:
                self.httpd.shutdown()
                self.httpd.server_close()
        except Exception:
            pass
        finally:
            self.httpd = None


def main():
    global plate_nohit_count, plate_rgb, ROI_MARGIN

    # mode=plate면 cascade 비활성(차량 박스만 찍히는 혼동 방지)
    _cascade = cascade_plate
    if mode == "plate":
        _cascade = False

    # 동적 import
    car_det = None
    plate_det = None

    need_car = (mode in ("car", "both")) or (mode == "plate" and _cascade)
    if need_car:
        from car_infer import CarDetector
        car_det = CarDetector(
            model_path=car_model_path, conf=car_conf, imgsz=car_imgsz,
            device=use_device, use_half=use_half, max_det=car_maxdet
        )

    if mode in ("plate", "both"):
        from plate_infer import PlateDetector
        plate_det = PlateDetector(
            model_path=plate_model_path, conf=plate_conf, imgsz=PLATE_IMG_SZ,
            device=use_device, use_half=use_half, max_det=plate_maxdet
        )
        lpr = LPRViewerOCR(LPR_MODEL_PATH, LPR_LABEL_PATH, rgb=OCR_RGB)

    # 비디오 준비
    writer = None
    if USE_RTMP:
        receiver = RTMPReceiver(RTMP_URL).open()
        frame_iter = receiver.frames()
        # 첫 프레임으로 출력 크기 결정
        first_frame = next(frame_iter)
        h0, w0 = first_frame.shape[:2]
        w_out, h_out = (resize_display or (w0, h0))
        # VideoWriter 준비는 기존 로직 그대로
        if save_video_path:
            writer, _ = ensure_video_writer_path(save_video_path, w_out, h_out, fps=30.0)
            
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"비디오를 열 수 없음: {video_path}")
        if resize_display:
            w_out, h_out = resize_display
        else:
            ok, fr0 = cap.read()
            if not ok: raise RuntimeError("첫 프레임 읽기 실패")
            h0, w0 = fr0.shape[:2]; w_out, h_out = (w0, h0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if save_video_path:
            writer, _ = ensure_video_writer_path(save_video_path, w_out, h_out, fps=30.0)

    # ── 공통 초기화 (비디오 준비 직후) ──────────────────────────────
    fps_src = 30.0 if USE_RTMP else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    eff_fps = fps_src / max(1, frame_skip_n)
    dwell_frames_th = int(np.ceil(VIOLATION_SECONDS * eff_fps))

    # === MJPEG 서버 시작 ===
    fb = FrameBuffer(jpeg_quality=JPEG_QUALITY)
    mjpeg = MjpegServer(HTTP_HOST, HTTP_PORT, fb).start()

    # SORT 트래커 & 트랙 상태
    tracker = Sort(
        max_age=int(np.ceil(eff_fps * 1.5)),
        min_hits=SORT_MIN_HITS,
        iou_threshold=SORT_IOU
    )
    track_state = {}

    # WinForms(TCP) 송신기 & 중복 방지 플래그
    sender = TcpJsonSender(TcpConfig(host=TCP_HOST, port=TCP_PORT)) if SEND_TO_WINFORMS else None
    sent_final = {}       # track_id -> True
    last_crop_path = None # 최근 썸네일 경로

    # MQTT 클라이언트 (WinForms DataReceiver 호환)
    mqtt_cli = None
    if USE_MQTT:
        if mqtt is None:
            print("⚠️ paho-mqtt 미설치 → MQTT 비활성화")
        else:
            mqtt_cli = mqtt.Client(client_id="drone-processor")
            try:
                # ✅ 콜백 연결
                mqtt_cli.on_connect = _on_mqtt_connect
                # on_message에 sender를 넘기기 위해 래핑
                mqtt_cli.on_message = lambda c, u, m: _on_mqtt_message(sender, m)

                mqtt_cli.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
                mqtt_cli.loop_start()
                print(f"✅ MQTT 연결: {MQTT_HOST}:{MQTT_PORT}")
            except Exception as e:
                print(f"❌ MQTT 연결 실패: {e}")
                mqtt_cli = None

    # HUD/루프 변수
    win = "Viewer (cascade+SORT+OCR+cache)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if resize_display:
        cv2.resizeWindow(win, *resize_display)
    frame_idx = 0
    ema = None

    while True:
        if USE_RTMP:
            try:
                frame = next(frame_iter)
            except StopIteration:
                break
        else:
            ok, frame = cap.read()
            if not ok:
                break

        frame_idx += 1

        # 주기 상태 전송 (선택)
        #if USE_MQTT and mqtt_cli is not None and (frame_idx % 60 == 0):
        #    try:
        #        status = {
        #            "Battery": 100.0,   # 실제 값 없으면 임시값
        #            "Altitude": 0.0,
        #            "Latitude": 0.0,
        #            "Longitude": 0.0,
        #            "EngineStatus": "ON"
        #        }
        #        mqtt_cli.publish(MQTT_TOPIC_STATUS, json.dumps(status, ensure_ascii=False), qos=0, retain=False)
        #    except Exception as e:
        #        print(f"⚠️ MQTT status 실패: {e}")

        clean = frame.copy()

        # 프레임별 OCR 캐시 초기화
        # key: (tid_or_-1, w_bucket, h_bucket) -> { (ahash, dhash): (text, conf, box) }
        ocr_cache_frame = {}

        do_infer = (frame_skip_n <= 1) or (frame_idx % frame_skip_n == 1)

        # ---------------- CAR DETECT ----------------
        car_res = None
        if do_infer and car_det is not None:
            car_res = car_det.model.predict(
                frame, conf=car_conf, imgsz=car_imgsz, device=car_det.device,
                half=car_det.half, max_det=car_maxdet, iou=car_iou,
                agnostic_nms=agnostic_nms, verbose=False
            )[0]

        car_dets = []
        if car_res is not None and getattr(car_res, "boxes", None) is not None:
            names = getattr(car_res, "names", {})
            xyxy = car_res.boxes.xyxy.cpu().numpy()
            clss = car_res.boxes.cls.cpu().numpy().astype(int)
            confs = car_res.boxes.conf.cpu().numpy()
            for box, c, cf in zip(xyxy, clss, confs):
                cname = names[c] if isinstance(names, (list, dict)) else str(c)
                if str(cname).lower() not in ("car", "truck", "bus"):
                    continue
                x1, y1, x2, y2 = map(float, box)
                car_dets.append([x1, y1, x2, y2, float(cf)])

        # ---------------- SORT UPDATE ----------------
        if do_infer:
            dets = np.array(car_dets, dtype=np.float32) if len(car_dets) else np.empty((0,5), np.float32)
            tracks_np = tracker.update(dets)
            tracked_cars = []
            for x1, y1, x2, y2, tid_f in tracks_np:
                tid = int(tid_f)
                st = track_state.get(tid)
                if st is None:
                    st = {'count': 0, 'ready': False, 'last_seen': frame_idx}
                    track_state[tid] = st
                # '추론된 프레임'에서만 카운트 증가
                st['count'] += 1
                st['last_seen'] = frame_idx
                if (not st['ready']) and st['count'] >= dwell_frames_th:
                    st['ready'] = True

                secs = st['count'] / max(eff_fps, 1e-6)
                color = (60,220,60) if st['ready'] else (180,180,60)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                put_text(frame, f"ID {tid}  {secs:.1f}s/{VIOLATION_SECONDS:.0f}s",
                         (int(x1)+2, max(18, int(y1)-6)), 0.55, 1)

                tracked_cars.append({'tid': tid, 'box': [x1,y1,x2,y2], 'ready': st['ready']})
        else:
            tracks_np = np.empty((0,5), dtype=np.float32)
            tracked_cars = []

        # 오래된 트랙 정리(실제 프레임 기준 3초)
        prune_gap = int(np.ceil((fps_src or 30.0) * 3.0))
        for tid in list(track_state.keys()):
            if frame_idx - track_state[tid]['last_seen'] > prune_gap:
                del track_state[tid]

        # ---------------- PLATE (캐스케이드/전체 프레임) ----------------
        plate_boxes_global, plate_scores_global, plate_owner_tid = [], [], []

        if (mode in ("plate", "both")) and plate_det is not None and do_infer:
            if _cascade:
                # 차량 ROI 내부에서 plate
                ZOOM_OUT = 1.5
                for tid, roi_canvas, roi_rect_g, pad_x, pad_y in make_rois_for_tracks(
                        clean, tracked_cars, margin_ratio=ROI_MARGIN, zoom_out=ZOOM_OUT, rgb=plate_rgb):
                    res_roi = plate_det.model.predict(
                        roi_canvas, conf=plate_conf, iou=plate_iou,
                        imgsz=PLATE_CASCADE_IMG_SZ, device=plate_det.device, half=plate_det.half,
                        max_det=plate_maxdet, agnostic_nms=agnostic_nms, verbose=False
                    )[0]
                    if getattr(res_roi, "boxes", None) is None:
                        continue
                    xyxy_p = res_roi.boxes.xyxy.cpu().numpy()
                    confs_p = res_roi.boxes.conf.cpu().numpy()
                    for (bx, by, ex, ey), sc in zip(xyxy_p, confs_p):
                        g = map_roi_box_to_global(
                            (bx, by, ex, ey), roi_rect_g, pad_x, pad_y,
                            frame_shape_hw=clean.shape[:2]
                        )
                        if g is None:
                            continue
                        plate_boxes_global.append(g)
                        plate_scores_global.append(float(sc))
                        plate_owner_tid.append(tid)
            else:
                # 비캐스케이드: 원본 전체 프레임에서 plate 탐지
                frame_in = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB) if plate_rgb else clean
                r_full = plate_det.model.predict(
                    frame_in, conf=plate_conf, imgsz=PLATE_IMG_SZ, device=plate_det.device,
                    half=plate_det.half, max_det=plate_maxdet, iou=plate_iou,
                    agnostic_nms=agnostic_nms, verbose=False
                )[0]
                if getattr(r_full, "boxes", None) is not None:
                    xyxy_p = r_full.boxes.xyxy.cpu().numpy()
                    confs_p = r_full.boxes.conf.cpu().numpy()
                    for (bx, by, ex, ey), sc in zip(xyxy_p, confs_p):
                        plate_boxes_global.append([int(bx), int(by), int(ex), int(ey)])
                        plate_scores_global.append(float(sc))
                        plate_owner_tid.append(-1)

        # ---------------- plate 후처리 NMS ----------------
        if plate_boxes_global:
            pb = np.array(plate_boxes_global, dtype=np.float32)
            ps = np.array(plate_scores_global, dtype=np.float32)
            keep = nms_numpy(pb, ps, iou_thres=POST_NMS_IOU_PLATE)
            pb = pb[keep]; ps = ps[keep]
            plate_owner_tid = [plate_owner_tid[k] for k in keep]
        else:
            pb = np.zeros((0, 4), dtype=np.float32)
            ps = np.zeros((0,), dtype=np.float32)

        # ---------------- plate 미리보기(겹침 방지) ----------------
        if render_plate_boxes and len(pb) > 0:
            for idx, (box, sc) in enumerate(zip(pb, ps)):
                tid = plate_owner_tid[idx] if idx < len(plate_owner_tid) else -1
                is_ready = (mode == "plate") or (tid in track_state and track_state[tid].get('ready', False))
                if HIDE_PLATE_SCORE_WHEN_READY and is_ready:
                    continue
                else:
                    draw_box(frame, box, (230, 70, 70), f"plate {sc:.2f}")

        # ---------------- OCR (5초 충족 트랙만, plate 모드는 즉시) ----------------
        if (mode in ("plate", "both")) and len(pb) > 0:
            for i, (box, sc) in enumerate(zip(pb, ps)):
                tid = plate_owner_tid[i] if i < len(plate_owner_tid) else -1
                is_ready = (mode == "plate") or (tid in track_state and track_state[tid].get('ready', False))
                if not is_ready:
                    continue

                # OCR 입력 크롭(마진 0.0)
                crop_ocr, _ = safe_crop(clean, box, margin_ratio=ocr_crop_margin)
                if crop_ocr is None:
                    continue
                # 너무 작으면 보정
                h_, w_ = crop_ocr.shape[:2]
                if w_ < min_crop_wh[0] or h_ < min_crop_wh[1]:
                    new_w = max(min_crop_wh[0], w_)
                    new_h = max(min_crop_wh[1], h_)
                    crop_ocr = cv2.resize(crop_ocr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    h_, w_ = crop_ocr.shape[:2]

                # 프레임 내 OCR 캐시 키(트랙/사이즈 버킷)
                w_bucket = int(round(w_ / OCR_SIZE_BUCKET) * OCR_SIZE_BUCKET)
                h_bucket = int(round(h_ / OCR_SIZE_BUCKET) * OCR_SIZE_BUCKET)
                tid_key = (tid if OCR_CACHE_BY_TID else -1)
                cache_key = (tid_key, w_bucket, h_bucket)
                if cache_key not in ocr_cache_frame:
                    ocr_cache_frame[cache_key] = {}

                # 해시 계산(aHash + dHash)
                ah = ahash64(crop_ocr)
                dh = dhash64(crop_ocr)

                # 캐시 탐색: 임계 이내면 히트
                hit = None
                for (ah_prev, dh_prev), (t_prev, c_prev, _) in ocr_cache_frame[cache_key].items():
                    if hamming64(ah, ah_prev) <= OCR_AHASH_THRESH and hamming64(dh, dh_prev) <= OCR_DHASH_THRESH:
                        hit = (t_prev, c_prev)
                        break

                if hit is not None and hit[1] >= OCR_RECALL_IF_LOW:
                    text, conf = hit
                else:
                    # 캐시 미스거나 conf 낮으면 OCR 호출
                    crop_ocr = maybe_upscale(crop_ocr, ocr_upscale)
                    text, conf = lpr.predict_image(crop_ocr)
                    # 캐시에 저장
                    ocr_cache_frame[cache_key][(ah, dh)] = (text, conf, box)

                # 표시(ready이므로 이 단계에서만 사각형+텍스트를 그림)
                x1,y1,x2,y2 = map(int, box)
                color = (50, 200, 50) if conf >= OCR_MIN_CONF else (80, 80, 200)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                label = f"ID{tid} {text or '---'} ({conf:.2f})"
                put_text(frame, label, (x1+2, max(18, y1-6)), scale=0.55, thickness=1)

                # 저장 썸네일
                if save_crops:
                    crop_thumb, _ = safe_crop(clean, box, margin_ratio=crop_margin)
                    if crop_thumb is not None:
                        h_t, w_t = crop_thumb.shape[:2]
                        mw, mh = min_crop_wh
                        if w_t >= mw and h_t >= mh:
                            crop_thumb = maybe_upscale(crop_thumb, ocr_upscale)
                            safe_text = "".join(ch for ch in (text or "") if ch.isalnum())
                            fname = f"plate_{frame_idx:06d}_tid{tid}_{i:02d}_{int(sc*100):02d}_{int(conf*100):02d}_{safe_text or 'unk'}.jpg"
                            out_path = os.path.join(crops_dir, fname)
                            cv2.imwrite(out_path, crop_thumb, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                            last_crop_path = out_path  # [추가] MQTT/TCP 전송용으로 최근 크롭 경로 저장

                # --- [추가] WinForms/TCP + MQTT 이벤트 전송 ---
                if SEND_TO_WINFORMS and is_ready and not sent_final.get(tid, False):
                    car_box_xyxy = _get_car_box(tracked_cars, tid)
                    duration_sec = track_state[tid]['count'] / max(eff_fps, 1e-6)

                    # TCP(JSON Lines) 이벤트
                    event = make_violation_event(
                        track_id=tid,
                        plate_text=text or "",
                        plate_conf=float(conf),
                        duration_sec=float(duration_sec),
                        plate_box_xyxy=[int(v) for v in box],
                        car_box_xyxy=car_box_xyxy,
                        image_path=(last_crop_path if save_crops else None),
                        stream_id="stream1",
                        source_url=(RTMP_URL if USE_RTMP else video_path),
                        extra={"frame_idx": frame_idx}
                    )
                    if sender:
                        sender.send(event)

                    # MQTT (WinForms DataReceiver 호환)
                    if USE_MQTT and mqtt_cli is not None:
                        try:
                            payload = {
                                "vehicle_number": text or "",
                                "detected_time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                                "is_illegal": True,
                                "confidence": float(conf),
                                "latitude": 0.0,
                                "longitude": 0.0,
                                "image_path": (last_crop_path if save_crops else None)
                            }
                            mqtt_cli.publish(MQTT_TOPIC_VEHICLE, json.dumps(payload, ensure_ascii=False), qos=0, retain=False)
                        except Exception as e:
                            print(f"⚠️ MQTT publish 실패: {e}")

                    sent_final[tid] = True

        # ---------------- plate 헬스체크/폴백(연속 무검출) ----------------
        if (mode in ("plate", "both")) and plate_det is not None and do_infer:
            total_plates = len(plate_boxes_global)
            if total_plates == 0:
                plate_nohit_count += 1
            else:
                plate_nohit_count = 0

            if total_plates == 0 and plate_nohit_count >= PLATE_HEALTH_N:
                # 1) 캐스케이드 ON이면 ROI에서 RGB/BGR 반전하여 1회 재시도
                def run_plate_on_rois(use_rgb: bool):
                    boxes, scores, owners = [], [], []
                    for tc in tracked_cars:
                        car_box = tc['box']; tid = tc['tid']
                        roi, (x1g, y1g, x2g, y2g) = safe_crop(clean, car_box, margin_ratio=ROI_MARGIN)
                        if roi is None or roi.size == 0:
                            continue
                        roi_in = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) if use_rgb else roi
                        r = plate_det.model.predict(
                            roi_in, conf=plate_conf, imgsz=PLATE_CASCADE_IMG_SZ, device=plate_det.device,
                            half=plate_det.half, max_det=plate_maxdet, iou=plate_iou,
                            agnostic_nms=agnostic_nms, verbose=False
                        )[0]
                        if getattr(r, "boxes", None) is None:
                            continue
                        xyxy_p = r.boxes.xyxy.cpu().numpy()
                        confs_p = r.boxes.conf.cpu().numpy()
                        for (bx, by, ex, ey), sc in zip(xyxy_p, confs_p):
                            gx1 = x1g + int(bx); gy1 = y1g + int(by)
                            gx2 = x1g + int(ex); gy2 = y1g + int(ey)
                            b_global = safe_pad_box([gx1, gy1, gx2, gy2], clean.shape[1], clean.shape[0], 0.0)
                            if b_global is None: continue
                            boxes.append(b_global); scores.append(float(sc)); owners.append(tid)
                    return boxes, scores, owners

                if _cascade:
                    alt_boxes, alt_scores, alt_owners = run_plate_on_rois(use_rgb=not plate_rgb)
                    if len(alt_boxes) > 0:
                        plate_boxes_global.extend(alt_boxes)
                        plate_scores_global.extend(alt_scores)
                        plate_owner_tid.extend(alt_owners)
                        plate_rgb = not plate_rgb
                        plate_nohit_count = 0

                # 2) 그래도 0이면 전체 프레임 헬스체크 1회
                if len(plate_boxes_global) == 0:
                    frame_in = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB) if plate_rgb else clean
                    r_full = plate_det.model.predict(
                        frame_in, conf=plate_conf, imgsz=PLATE_IMG_SZ, device=plate_det.device,
                        half=plate_det.half, max_det=plate_maxdet, iou=plate_iou,
                        agnostic_nms=agnostic_nms, verbose=False
                    )[0]
                    if getattr(r_full, "boxes", None) is not None:
                        xyxy_p = r_full.boxes.xyxy.cpu().numpy()
                        confs_p = r_full.boxes.conf.cpu().numpy()
                        for (bx, by, ex, ey), sc in zip(xyxy_p, confs_p):
                            plate_boxes_global.append([int(bx), int(by), int(ex), int(ey)])
                            plate_scores_global.append(float(sc))
                            plate_owner_tid.append(-1)
                        if len(xyxy_p) > 0:
                            ROI_MARGIN = min(0.40, ROI_MARGIN + 0.04)  # ROI가 너무 타이트했을 수 있음
                            plate_nohit_count = 0

        # ---------------- FPS/표시/저장 ----------------
        now = time.perf_counter()
        fps_inst = 1.0 / max(now - (getattr(main, "_last_t", now)), 1e-6)
        main._last_t = now
        ema = fps_inst if ema is None else 0.2 * fps_inst + 0.8 * ema
        put_text(frame, f"FPS: {ema:.1f}  mode={mode}  cascade={_cascade}", (10, 26), 0.7, 2)

        frame_show = cv2.resize(frame, resize_display, interpolation=cv2.INTER_LINEAR) if resize_display else frame

        fb.update(frame_show)

        cv2.imshow(win, frame_show)
        if save_video_path and writer is not None:
            writer.write(frame_show)

        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
            break

    # --- [종료 정리] ---
    if not USE_RTMP:
        cap.release()
    else:
        receiver.close()

    if writer is not None:
        writer.release()

    if SEND_TO_WINFORMS and sender:
        sender.close()

    if USE_MQTT and mqtt_cli is not None:
        try:
            mqtt_cli.loop_stop()
            mqtt_cli.disconnect()
        except Exception:
            pass
    
    mjpeg.stop()
    cv2.destroyAllWindows()
    # -------------------

if __name__ == "__main__":
    main()
