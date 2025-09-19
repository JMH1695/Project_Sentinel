# winforms_bridge.py
# -*- coding: utf-8 -*-
import json
import socket
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# ======== 공통 이벤트 스키마 ========
def make_violation_event(
    track_id: int,
    plate_text: str,
    plate_conf: float,
    duration_sec: float,
    plate_box_xyxy: list[int],
    car_box_xyxy: list[int],
    image_path: Optional[str],
    stream_id: str = "stream1",
    source_url: Optional[str] = None,
    detected_time_iso: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import datetime
    ts = detected_time_iso or datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")
    ev = {
        "type": "violation_finalized",
        "schema": 1,
        "ts": ts,
        "track_id": track_id,
        "duration_sec": round(float(duration_sec), 3),
        "plate_text": plate_text or "",
        "plate_conf": round(float(plate_conf), 4),
        "plate_box_xyxy": plate_box_xyxy,
        "car_box_xyxy": car_box_xyxy,
        "image_path": image_path,
        "meta": {
            "stream_id": stream_id,
            "source": source_url,
        },
    }
    if extra:
        ev["extra"] = extra
    return ev

# ======== TCP(JSON Lines) 송신기 ========
@dataclass
class TcpConfig:
    host: str = "127.0.0.1"
    port: int = 5055
    reconnect: bool = True
    backoff_initial: float = 1.0
    backoff_max: float = 10.0
    timeout_sec: float = 5.0

class TcpJsonSender:
    """
    WinForms 쪽이 TCP로 개행구분 JSON을 받는 경우.
    WinForms에서는 .NET의 NetworkStream/StreamReader로 라인 단위 수신하면 됨.
    """
    def __init__(self, cfg: TcpConfig):
        self.cfg = cfg
        self.sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def _connect(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.cfg.timeout_sec)
        s.connect((self.cfg.host, self.cfg.port))
        self.sock = s

    def _reconnect_loop(self) -> None:
        delay = self.cfg.backoff_initial
        while True:
            try:
                self._connect()
                return
            except Exception as e:
                if not self.cfg.reconnect:
                    raise
                print(f"⚠️ TCP 연결 실패: {e} → {delay:.1f}s 후 재시도")
                time.sleep(delay)
                delay = min(delay * 2, self.cfg.backoff_max)

    def send(self, obj: Dict[str, Any]) -> bool:
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            if not self.sock:
                self._reconnect_loop()
            try:
                assert self.sock is not None
                self.sock.sendall(data)
                return True
            except Exception as e:
                print(f"⚠️ TCP 전송 실패: {e}")
                if self.cfg.reconnect:
                    self._reconnect_loop()
                    try:
                        assert self.sock is not None
                        self.sock.sendall(data)
                        return True
                    except Exception as e2:
                        print(f"❌ 재시도 실패: {e2}")
                return False

    def close(self) -> None:
        with self._lock:
            if self.sock:
                try:
                    self.sock.close()
                finally:
                    self.sock = None

# ======== MQTT 퍼블리셔(옵션) ========
try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None  # 필요 없으면 설치 안 해도 됨

@dataclass
class MqttConfig:
    host: str = "127.0.0.1"
    port: int = 1883
    client_id: str = "drone-publisher"
    topic: str = "vehicle/detection"
    keepalive: int = 30
    username: Optional[str] = None
    password: Optional[str] = None
    qos: int = 0
    retain: bool = False
    reconnect_delay: float = 2.0

class MqttSender:
    """
    WinForms가 MQTT를 구독하는 구조일 때 사용.
    """
    def __init__(self, cfg: MqttConfig):
        if mqtt is None:
            raise ImportError("paho-mqtt가 설치되어 있지 않음: pip install paho-mqtt")
        self.cfg = cfg
        self.cli = mqtt.Client(client_id=cfg.client_id, clean_session=True)
        if cfg.username:
            self.cli.username_pw_set(cfg.username, cfg.password)
        self.cli.on_disconnect = self._on_disconnect
        self._connected = False

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        print("⚠️ MQTT 연결 끊김")

    def connect(self):
        while not self._connected:
            try:
                self.cli.connect(self.cfg.host, self.cfg.port, self.cfg.keepalive)
                self.cli.loop_start()
                # 연결 완료까지 잠깐 대기
                time.sleep(0.2)
                self._connected = True
                print("✅ MQTT 연결")
            except Exception as e:
                print(f"⚠️ MQTT 연결 실패: {e} → {self.cfg.reconnect_delay}s 후 재시도")
                time.sleep(self.cfg.reconnect_delay)

    def publish(self, obj: Dict[str, Any]) -> bool:
        if not self._connected:
            self.connect()
        try:
            payload = json.dumps(obj, ensure_ascii=False)
            res = self.cli.publish(self.cfg.topic, payload, qos=self.cfg.qos, retain=self.cfg.retain)
            res.wait_for_publish()
            if res.rc != 0:
                print(f"⚠️ MQTT publish rc={res.rc}")
                return False
            return True
        except Exception as e:
            print(f"❌ MQTT publish 실패: {e}")
            self._connected = False
            return False

    def close(self):
        try:
            self.cli.loop_stop()
            self.cli.disconnect()
        except Exception:
            pass
