# lpr_ocr_viewer.py
# Viewer 연동 전용: BGR ndarray → (text, conf)
from typing import List, Tuple
import os, numpy as np, cv2

def _import_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except Exception:
        try:
            from tensorflow.lite import Interpreter
            return Interpreter
        except Exception:
            from tensorflow.lite.python.interpreter import Interpreter
            return Interpreter

Interpreter = _import_interpreter()

class LPRViewerOCR:
    def __init__(self, model_path: str, label_path: str, *, rgb: bool = False, normalize: bool = True, blank_index: int = 0):
        if not os.path.exists(model_path):  raise FileNotFoundError(f"model not found: {model_path}")
        if not os.path.exists(label_path):  raise FileNotFoundError(f"labels not found: {label_path}")
        self.rgb, self.normalize, self.blank_index = rgb, normalize, blank_index

        with open(label_path, "r", encoding="utf-8") as f:
            self.labels = [ln.strip() for ln in f]

        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        ishape = self.inp["shape"]  # (1,H,W,C)
        if len(ishape) != 4: raise ValueError(f"unsupported input shape: {ishape}")
        self.H, self.W, self.C = int(ishape[1]), int(ishape[2]), int(ishape[3])
        self.in_idx, self.out_idx = self.inp["index"], self.out["index"]
        self.in_dtype = self.inp["dtype"]

    # ---- public ----
    def predict_image(self, bgr: np.ndarray) -> Tuple[str, float]:
        arr = self._preprocess(bgr)
        self.interp.set_tensor(self.in_idx, arr)
        self.interp.invoke()
        logits = self.interp.get_tensor(self.out_idx)[0]
        return self._decode_ctc(logits)

    def predict_batch(self, crops_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
        return [self.predict_image(x) for x in crops_bgr]

    # ---- internal ----
    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        if self.C == 1:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(gray, (self.W, self.H))
            if self.in_dtype == np.float32 and self.normalize: img = img.astype(np.float32) / 255.0
            else: img = img.astype(self.in_dtype)
            return img[None, ..., None]
        else:
            x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if self.rgb else bgr
            img = cv2.resize(x, (self.W, self.H))
            if self.in_dtype == np.float32 and self.normalize: img = img.astype(np.float32) / 255.0
            else: img = img.astype(self.in_dtype)
            return img[None, ...]

    def _softmax(self, x, axis=-1):
        m = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - m)
        return e / np.sum(e, axis=axis, keepdims=True)

    def _decode_ctc(self, logits) -> Tuple[str, float]:
        if logits.ndim != 2:
            logits = np.squeeze(logits, 0)
            if logits.ndim != 2: raise ValueError(f"unexpected logits shape: {logits.shape}")
        probs = self._softmax(logits, -1)
        idxs  = np.argmax(probs, -1)
        text_chars, confs = [], []
        prev = -1
        for t, k in enumerate(idxs):
            if k == self.blank_index or k == prev:  # blank or repeat
                prev = k; continue
            ch = self.labels[k] if 0 <= k < len(self.labels) else ""
            if ch:
                text_chars.append(ch)
                confs.append(float(probs[t, k]))
            prev = k
        return "".join(text_chars), (float(np.mean(confs)) if confs else 0.0)
