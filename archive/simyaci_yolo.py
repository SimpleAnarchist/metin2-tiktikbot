from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Optional, Tuple

import numpy as np
import pyautogui


import torch
from ultralytics import YOLO

Region = Tuple[int, int, int, int]
Target = int | str


@dataclass(frozen=True)
class DetectionHit:
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    conf: float
    class_id: int
    class_name: str


_model: Optional[YOLO] = None
_model_path: Optional[str] = None
_cached_target: Optional[Target] = None
_cached_target_id: Optional[int] = None


def _normalize_device(device: int | str) -> str:
    if isinstance(device, int):
        return f"cuda:{device}"
    d = str(device).strip().lower()
    if d == "cpu":
        return "cpu"
    if d in {"cuda", "gpu"}:
        return "cuda:0"
    if d.startswith("cuda:"):
        return d
    if d.isdigit():
        return f"cuda:{int(d)}"
    return str(device)


def _load_model(model_path: str) -> YOLO:
    global _model, _model_path
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"YOLO ağırlık dosyası bulunamadı: {p}")

    if _model is None or _model_path != str(p):
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        _model = YOLO(str(p))
        _model_path = str(p)

        try:
            _model.fuse()
        except Exception:
            pass

    return _model


def _resolve_class_id(model: YOLO, target: Target) -> int:
    global _cached_target, _cached_target_id

    if _cached_target == target and _cached_target_id is not None:
        return _cached_target_id

    names = model.names
    if isinstance(target, int):
        _cached_target, _cached_target_id = target, target
        return target

    t = str(target).strip().lower()
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).strip().lower() == t:
                _cached_target, _cached_target_id = target, int(k)
                return int(k)
    else:
        for i, v in enumerate(names):
            if str(v).strip().lower() == t:
                _cached_target, _cached_target_id = target, int(i)
                return int(i)

    raise ValueError(f"Class bulunamadı. target='{target}'. model.names={names}")


# --------- COLOR CAPTURE (mss -> fallback ImageGrab) ----------
_mss_instance = None

def _grab_color(region: Region) -> np.ndarray:
    """
    return: (H, W, 3) BGR uint8
    Önce mss dener (hızlı), yoksa PIL.ImageGrab (daha yavaş ama stabil).
    """
    x1, y1, x2, y2 = region
    w, h = x2 - x1, y2 - y1

    # 1) MSS (varsa)
    try:
        global _mss_instance
        import mss  # type: ignore

        if _mss_instance is None:
            _mss_instance = mss.mss()

        mon = {"left": x1, "top": y1, "width": w, "height": h}
        raw = _mss_instance.grab(mon)              # BGRA
        img = np.asarray(raw)[:, :, :3]            # BGR
        return img
    except Exception:
        pass

    # 2) PIL.ImageGrab fallback
    from PIL import ImageGrab  # pillow genelde sende var
    img = ImageGrab.grab(bbox=(x1, y1, x2, y2))    # RGB
    arr = np.array(img)                             # RGB
    return arr[:, :, ::-1].copy()                   # BGR


# ---------------- Public API ----------------
def detect_simyaci_once(
    *,
    model_path: str,
    region: Region,
    target: Target = "simyaci",
    conf_accept: float = 0.25,   # <-- kabul eşiği (senin istediğin)
    conf_search: float = 0.05,   # <-- arama eşiği (düşük)
    imgsz: int = 640,
    iou: float = 0.45,
    device: int | str = 0,
    half: bool = True,
    require_cuda: bool = True,
    debug: bool = False,
) -> Optional[DetectionHit]:

    dev = _normalize_device(device)
    if dev != "cpu":
        if not torch.cuda.is_available():
            if require_cuda:
                raise RuntimeError("CUDA görünmüyor. GPU inference için CUDA'lı PyTorch gerekli.")
            dev = "cpu"
            half = False

    frame = _grab_color(region)  # BGR
    if frame is None:
        if debug:
            print("[YOLO] frame None")
        return None

    model = _load_model(model_path)
    target_id = _resolve_class_id(model, target)

    # <-- DİKKAT: predict conf artık conf_search
    results = model.predict(
        source=frame,
        device=dev,
        half=(half and dev != "cpu"),
        imgsz=imgsz,
        conf=conf_search,
        iou=iou,
        classes=[target_id],
        max_det=20,          # 1 yerine biraz bırak: en iyiyi biz seçeceğiz
        verbose=False,
    )

    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        if debug:
            print(f"[YOLO] det=0 (conf_search={conf_search})")
        return None

    # en iyi conf'ı seç
    confs = boxes.conf
    best_i = int(torch.argmax(confs).item())
    pred_conf = float(confs[best_i].item())

    xyxy = boxes.xyxy[best_i]
    cls_id = int(boxes.cls[best_i].item())

    # minimum CPU transfer
    x1, y1, x2, y2 = [int(v) for v in xyxy.round().to(torch.int64).cpu().tolist()]

    rx1, ry1, _, _ = region
    sx1, sy1, sx2, sy2 = rx1 + x1, ry1 + y1, rx1 + x2, ry1 + y2
    cx, cy = (sx1 + sx2) // 2, (sy1 + sy2) // 2

    names = model.names
    class_name = names[cls_id] if not isinstance(names, dict) else names.get(cls_id, str(cls_id))

    if debug:
        print(f"[YOLO] best_conf={pred_conf:.3f} accept={conf_accept:.2f} center=({cx},{cy})")

    if pred_conf < conf_accept:
        return None

    return DetectionHit(
        center=(cx, cy),
        bbox=(sx1, sy1, sx2, sy2),
        conf=pred_conf,
        class_id=cls_id,
        class_name=str(class_name),
    )


def find_simyaci_until_found(
    *,
    model_path: str,
    region: Region,
    target: Target = "simyaci",
    conf_accept: float = 0.25,
    press_key: str = "q",
    press_count: int = 10,
    press_delay: float = 0.03,
    retry_delay: float = 0.12,
    imgsz: int = 640,
    iou: float = 0.45,
    device: int | str = 0,
    half: bool = True,
    require_cuda: bool = True,
    max_cycles: Optional[int] = None,
    debug: bool = False,
) -> DetectionHit:
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False

    cycles = 0
    while True:
        hit = detect_simyaci_once(
            model_path=model_path,
            region=region,
            target=target,
            conf_accept=conf_accept,
            imgsz=imgsz,
            iou=iou,
            device=device,
            half=half,
            require_cuda=require_cuda,
            debug=debug,
        )
        if hit is not None:
            return hit

        if debug:
            print(f"[YOLO] bulunamadı -> {press_count}x '{press_key}' basıyorum...")

        for _ in range(press_count):
            pyautogui.press(press_key)
            sleep(press_delay)

        sleep(retry_delay)

        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            raise TimeoutError(f"Simyacı bulunamadı. max_cycles={max_cycles} doldu.")