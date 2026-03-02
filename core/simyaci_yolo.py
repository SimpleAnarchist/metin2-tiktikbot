from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Optional, Tuple, Union

import pyautogui
import torch
from ultralytics import YOLO

from core.capture_service import grab_gray
import numpy as np

Region = Tuple[int, int, int, int]  # (x1,y1,x2,y2)
Target = Union[int, str]            # class_id veya class_name


@dataclass(frozen=True)
class DetectionHit:
    center: Tuple[int, int]             # (cx_screen, cy_screen)
    bbox: Tuple[int, int, int, int]     # (x1,y1,x2,y2) screen coords
    conf: float
    class_id: int
    class_name: str


_model: Optional[YOLO] = None
_model_path: Optional[str] = None
_cached_target: Optional[Target] = None
_cached_target_id: Optional[int] = None


def _normalize_device(device: Union[int, str]) -> str:
    if isinstance(device, int):
        return f"cuda:{device}"
    d = str(device).strip().lower()
    if d in {"cpu"}:
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
        # Performans flag’leri (özellikle aynı img size sürekli geliyorsa iyi)
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        _model = YOLO(str(p))
        _model_path = str(p)

        # PyTorch modelinde küçük hız kazancı sağlayabilir
        try:
            _model.fuse()
        except Exception:
            pass

    return _model


def _resolve_class_id(model: YOLO, target: Target) -> int:
    global _cached_target, _cached_target_id

    if _cached_target == target and _cached_target_id is not None:
        return _cached_target_id

    names = model.names  # dict veya list
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


def _warmup(model: YOLO, device: str, imgsz: int, half: bool) -> None:
    # İlk çağrıda “compile / allocate” gecikmesini azaltır
    if device == "cpu":
        return
    dummy = torch.zeros((1, 3, imgsz, imgsz), device=device)
    if half:
        dummy = dummy.half()
    with torch.inference_mode():
        try:
            _ = model.predict(source=dummy, device=device, verbose=False)
        except Exception:
            # bazı ultralytics sürümlerinde dummy tensor source ile uyumsuz olabilir
            pass


def detect_simyaci_once(
    *,
    model_path: str,
    region: Region,
    target: Target = "simyaci",
    conf_thres: float = 0.80,
    imgsz: int = 640,
    iou: float = 0.45,
    device: str = "cuda:0",
    half: bool = True,
    require_cuda: bool = True,
    warmup: bool = False,
    debug: bool = False,
) -> Optional[DetectionHit]:

    if device != "cpu":
        import torch
        if not torch.cuda.is_available():
            if require_cuda:
                raise RuntimeError("CUDA görünmüyor. GPU inference için CUDA'lı PyTorch gerekli.")
            device = "cpu"
            half = False

    gray = grab_gray(region)  # <-- capture_service aynı kalsın
    if gray is None:
        return None

    # YOLO 3 kanal sever: GRAY -> (H,W,3)
    frame = np.repeat(gray[:, :, None], 3, axis=2)

    model = _load_model(model_path)
    target_id = _resolve_class_id(model, target)

    results = model.predict(
        source=frame,
        device=device,
        half=(half and device != "cpu"),
        imgsz=imgsz,
        conf=conf_thres,
        iou=iou,
        classes=[target_id],  # sadece simyacı
        max_det=1,            # tek hedef
        verbose=False,
    )

    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return None

    # max_det=1 olduğundan 0. kutu yeter
    xyxy = boxes.xyxy[0]
    conf = float(boxes.conf[0].item())
    cls_id = int(boxes.cls[0].item())

    # 4 koordinatı CPU'ya indir (çok küçük veri)
    import torch
    if isinstance(xyxy, torch.Tensor):
        x1, y1, x2, y2 = [int(v) for v in xyxy.round().to(torch.int64).cpu().tolist()]
    else:
        x1, y1, x2, y2 = [int(v) for v in xyxy]

    rx1, ry1, _, _ = region
    sx1, sy1, sx2, sy2 = rx1 + x1, ry1 + y1, rx1 + x2, ry1 + y2
    cx, cy = (sx1 + sx2) // 2, (sy1 + sy2) // 2

    names = model.names
    class_name = names[cls_id] if not isinstance(names, dict) else names.get(cls_id, str(cls_id))

    if debug:
        print(f"[YOLO] hit: {class_name} conf={conf:.3f} center=({cx},{cy}) dev={device} half={half}")

    return DetectionHit(
        center=(cx, cy),
        bbox=(sx1, sy1, sx2, sy2),
        conf=conf,
        class_id=cls_id,
        class_name=str(class_name),
    )


def find_simyaci_until_found(
    *,
    model_path: str,
    region: Region,
    target: Target = "simyaci",
    conf_thres: float = 0.80,
    press_key: str = "q",
    press_count: int = 10,
    press_delay: float = 0.03,   # biraz daha hızlı
    retry_delay: float = 0.12,   # çok bekleme -> daha responsive
    imgsz: int = 640,
    iou: float = 0.45,
    device: Union[int, str] = "cuda:0",
    half: bool = True,
    require_cuda: bool = True,
    warmup: bool = True,
    max_cycles: Optional[int] = None,
    debug: bool = False,
) -> DetectionHit:
    """
    Bulana kadar döner:
    - Bulursa: DetectionHit döndürür.
    - Bulamazsa: 10x 'q' basar, tekrar dener.
    """
    # pyautogui hız/overhead ayarları
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False

    cycles = 0

    while True:
        hit = detect_simyaci_once(
            model_path=model_path,
            region=region,
            target=target,
            conf_thres=conf_thres,
            imgsz=imgsz,
            iou=iou,
            device=device,
            half=half,
            require_cuda=require_cuda,
            warmup=warmup and cycles == 0,  # sadece ilk tur warmup
            debug=debug,
        )
        if hit is not None and hit.conf >= conf_thres:
            return hit

        if debug:
            print(f"[YOLO] bulunamadı. {press_count}x '{press_key}' basıyorum...")

        for _ in range(press_count):
            pyautogui.press(press_key)
            sleep(press_delay)

        # UI’nin güncellenmesine minicik nefes
        sleep(retry_delay)

        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            raise TimeoutError(f"Simyacı bulunamadı. max_cycles={max_cycles} doldu.")