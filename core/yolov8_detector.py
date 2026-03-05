# core/yolov8_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import threading

import numpy as np
import cv2

try:
    import torch
except Exception:
    torch = None

from ultralytics import YOLO
import mss


# ============================================================
# MSS CAPTURE (BGR)
# ============================================================
_mss_lock = threading.Lock()
_mss_obj: Optional[mss.mss] = None


def _set_dpi_aware_windows():
    """Windows scaling (125%,150%) ROI koordinatlarını şaşırtmasın diye."""
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _get_mss() -> mss.mss:
    global _mss_obj
    if _mss_obj is None:
        _set_dpi_aware_windows()
        _mss_obj = mss.mss()
    return _mss_obj


def grab_bgr_roi(roi: Dict[str, int]) -> Optional[np.ndarray]:
    """
    roi = {"left": x, "top": y, "width": w, "height": h}
    Return: BGR frame (H,W,3) uint8 veya None
    """
    left = int(roi["left"])
    top = int(roi["top"])
    width = int(roi["width"])
    height = int(roi["height"])

    cam = _get_mss()
    with _mss_lock:
        raw = cam.grab({"left": left, "top": top, "width": width, "height": height})

    if raw is None:
        return None

    img_bgra = np.asarray(raw, dtype=np.uint8)  # (H,W,4) BGRA
    img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
    return img_bgr


# ============================================================
# DEBUG VIEWER: cv2.imshow varsa onu, yoksa matplotlib
# ============================================================
class DebugViewer:
    def __init__(self, window_name: str = "YOLOv8 Debug", resize_to: Optional[Tuple[int, int]] = None):
        self.window_name = window_name
        self.resize_to = resize_to
        self._backend = "auto"  # "cv2" / "mpl"
        self._mpl_inited = False
        self._mpl_fig = None
        self._mpl_ax = None
        self._mpl_im = None

    def show(self, bgr: np.ndarray):
        if self.resize_to is not None:
            bgr = cv2.resize(bgr, self.resize_to, interpolation=cv2.INTER_AREA)

        # 1) OpenCV GUI dene
        if self._backend in ("auto", "cv2"):
            try:
                cv2.imshow(self.window_name, bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise KeyboardInterrupt
                self._backend = "cv2"
                return
            except cv2.error:
                self._backend = "mpl"

        # 2) Matplotlib fallback
        import matplotlib.pyplot as plt

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if not self._mpl_inited:
            plt.ion()
            self._mpl_fig, self._mpl_ax = plt.subplots()
            try:
                self._mpl_fig.canvas.manager.set_window_title(self.window_name)
            except Exception:
                pass
            self._mpl_ax.axis("off")
            self._mpl_im = self._mpl_ax.imshow(rgb)
            self._mpl_inited = True
        else:
            self._mpl_im.set_data(rgb)

        self._mpl_fig.canvas.draw_idle()
        plt.pause(0.001)

        # pencere kapatılırsa döngüyü kır
        try:
            if not plt.fignum_exists(self._mpl_fig.number):
                raise KeyboardInterrupt
        except Exception:
            pass


_debug_viewer: Optional[DebugViewer] = None


def close_debug_windows():
    """Main sonunda çağır: GUI yoksa hata vermesin."""
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


# ============================================================
# OUTPUT TYPE (istersen ileride dışarı da döndürürsün)
# ============================================================
@dataclass(frozen=True)
class YoloHit:
    center_screen: Tuple[int, int]
    bbox_screen_xyxy: Tuple[int, int, int, int]
    conf: float
    cls: int


# ============================================================
# YOLOv8 DETECTOR
# ============================================================
class YoloV8Detector:
    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.15,
        iou_thres: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 10,
        prefer_fp16: bool = True,
        fuse: bool = True,
        warmup: bool = True,
    ):
        self.model_path = model_path
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.imgsz = int(imgsz)
        self.classes = classes
        self.max_det = int(max_det)
        self.prefer_fp16 = bool(prefer_fp16)

        if device is None:
            if torch is not None and torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = YOLO(self.model_path)

        try:
            self.model.to(self.device)
        except Exception:
            pass

        if fuse:
            try:
                self.model.fuse()
            except Exception:
                pass

        self.half = bool(self.prefer_fp16 and str(self.device).startswith("cuda"))

        if warmup:
            self._warmup()

    def _warmup(self):
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        try:
            _ = self.model.predict(
                source=dummy,
                imgsz=self.imgsz,
                conf=self.conf_thres,
                iou=self.iou_thres,
                device=self.device,
                half=self.half,
                classes=self.classes,
                max_det=self.max_det,
                verbose=False,
            )
        except Exception:
            pass

    @staticmethod
    def _annotate(
        frame_bgr: np.ndarray,
        found: bool,
        best_xyxy: Optional[Tuple[int, int, int, int]] = None,
        best_conf: Optional[float] = None,
        center_local: Optional[Tuple[int, int]] = None,
        note: str = "",
    ) -> np.ndarray:
        vis = frame_bgr.copy()

        if found and best_xyxy is not None:
            x1, y1, x2, y2 = best_xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if center_local is not None:
                cx, cy = center_local
                cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            txt = f"DETECTED conf={best_conf:.2f} {note}".strip() if best_conf is not None else f"DETECTED {note}".strip()
            cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            txt = f"NO DET {note}".strip()
            cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return vis

    def _predict(self, img: np.ndarray, imgsz_val: int):
        return self.model.predict(
            source=img,
            imgsz=imgsz_val,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            half=self.half,
            classes=self.classes,
            max_det=self.max_det,
            verbose=False,
        )

    def detect_center_from_bgr(
        self,
        bgr: np.ndarray,
        roi_left: int = 0,
        roi_top: int = 0,
        debug_show: bool = False,
        window_name: str = "YOLOv8 Debug",
        resize_to: Optional[Tuple[int, int]] = None,
        try_bigger_imgsz: bool = True,
        bigger_imgsz: int = 960,
        force_rgb: bool = False,
    ) -> Optional[Tuple[int, int]]:
        """
        Return: None veya (cx_screen, cy_screen)
        - 1. tur: imgsz=self.imgsz
        - Bulamazsa 2. tur: imgsz=bigger_imgsz
        - force_rgb: A/B test için (BGR->RGB çevirir)
        """
        global _debug_viewer  # <-- EN BAŞTA tek sefer

        if bgr is None:
            return None

        if bgr.dtype != np.uint8:
            bgr = bgr.astype(np.uint8, copy=False)

        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"Beklenmeyen frame shape: {bgr.shape} (BGR bekleniyor)")

        img_in = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if force_rgb else bgr

        # 1) normal tur
        results = self._predict(img_in, self.imgsz)
        r0 = results[0]
        used_imgsz = self.imgsz

        # 2) büyük imgsz turu
        if (r0.boxes is None or len(r0.boxes) == 0) and try_bigger_imgsz:
            results = self._predict(img_in, int(bigger_imgsz))
            r0 = results[0]
            used_imgsz = int(bigger_imgsz)

        if r0.boxes is None or len(r0.boxes) == 0:
            if debug_show:
                if _debug_viewer is None or _debug_viewer.window_name != window_name:
                    _debug_viewer = DebugViewer(window_name=window_name, resize_to=resize_to)
                vis = self._annotate(bgr, found=False, note=f"(imgsz={used_imgsz})")
                _debug_viewer.show(vis)
            return None

        boxes_xyxy = r0.boxes.xyxy.detach().cpu().numpy()
        confs = r0.boxes.conf.detach().cpu().numpy()

        best_i = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes_xyxy[best_i]
        conf = float(confs[best_i])

        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        cx_local = int((x1 + x2) / 2.0)
        cy_local = int((y1 + y2) / 2.0)

        if debug_show:
            if _debug_viewer is None or _debug_viewer.window_name != window_name:
                _debug_viewer = DebugViewer(window_name=window_name, resize_to=resize_to)
            vis = self._annotate(
                bgr,
                found=True,
                best_xyxy=(x1i, y1i, x2i, y2i),
                best_conf=conf,
                center_local=(cx_local, cy_local),
                note=f"(imgsz={used_imgsz})",
            )
            _debug_viewer.show(vis)

        cx_screen = int(roi_left + cx_local)
        cy_screen = int(roi_top + cy_local)
        return (cx_screen, cy_screen)

    def detect_center_from_roi(
        self,
        roi: Dict[str, int],
        debug_show: bool = False,
        window_name: str = "YOLOv8 Debug",
        resize_to: Optional[Tuple[int, int]] = None,
        try_bigger_imgsz: bool = True,
        bigger_imgsz: int = 960,
        force_rgb: bool = False,
    ) -> Optional[Tuple[int, int]]:
        bgr = grab_bgr_roi(roi)
        if bgr is None:
            return None
        return self.detect_center_from_bgr(
            bgr=bgr,
            roi_left=roi["left"],
            roi_top=roi["top"],
            debug_show=debug_show,
            window_name=window_name,
            resize_to=resize_to,
            try_bigger_imgsz=try_bigger_imgsz,
            bigger_imgsz=bigger_imgsz,
            force_rgb=force_rgb,
        )


# ============================================================
# SINGLETON API (main'den kolay çağrı)
# ============================================================
_detector_singleton: Optional[YoloV8Detector] = None

DEFAULT_ROI_800x600 = {"left": 15, "top": 50, "width": 1193, "height": 892}


def detect_yolov8_center(
    roi: Optional[Dict[str, int]] = None,
    model_path: str = r"C:\Users\boran\Desktop\image detection\metin2\assets\best.pt",
    conf_thres: float = 0.15,
    iou_thres: float = 0.45,
    imgsz: int = 640,
    device: Optional[str] = None,
    classes: Optional[List[int]] = None,
    debug_show: bool = True,
    window_name: str = "YOLOv8 Debug",
    resize_to: Optional[Tuple[int, int]] = None,
    try_bigger_imgsz: bool = True,
    bigger_imgsz: int = 960,
    force_rgb: bool = False,
) -> Optional[Tuple[int, int]]:
    """
    roi verilmezse otomatik 800x600 (0,0) yakalar.
    None veya (cx_screen, cy_screen) döndürür.
    """
    global _detector_singleton

    if roi is None:
        roi = DEFAULT_ROI_800x600

    need_new = (
        _detector_singleton is None
        or _detector_singleton.model_path != model_path
        or _detector_singleton.imgsz != int(imgsz)
        or abs(_detector_singleton.conf_thres - float(conf_thres)) > 1e-9
        or abs(_detector_singleton.iou_thres - float(iou_thres)) > 1e-9
    )

    if need_new:
        _detector_singleton = YoloV8Detector(
            model_path=model_path,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            imgsz=imgsz,
            device=device,
            classes=classes,
            max_det=10,
            prefer_fp16=True,
            fuse=True,
            warmup=True,
        )

    return _detector_singleton.detect_center_from_roi(
        roi=roi,
        debug_show=debug_show,
        window_name=window_name,
        resize_to=resize_to,
        try_bigger_imgsz=try_bigger_imgsz,
        bigger_imgsz=bigger_imgsz,
        force_rgb=force_rgb,
    )

def shutdown_yolov8():
    """
    YOLO'yu 'kapatmak' için:
    - singleton modeli sıfırlar (bir sonraki çağrıda yeniden yükler)
    - MSS objesini kapatır
    - debug pencerelerini kapatır
    - CUDA cache temizler (varsa)
    """
    global _detector_singleton, _mss_obj, _debug_viewer

    _detector_singleton = None
    _debug_viewer = None

    close_debug_windows()

    try:
        if _mss_obj is not None:
            _mss_obj.close()
    except Exception:
        pass
    _mss_obj = None

    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass