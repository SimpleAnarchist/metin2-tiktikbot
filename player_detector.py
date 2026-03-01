import cv2
import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple

# >>> EK: ekran yakalama için
import mss


@dataclass(frozen=True)
class Detection:
    """Represents one template match detection."""

    bbox_xywh: Tuple[int, int, int, int]  # (x, y, w, h) in ROI coordinates
    center_xy: Tuple[int, int]  # (cx, cy) in ROI coordinates
    score: float  # matchTemplate score [0..1]
    scale: float  # scale used for the template


@dataclass(frozen=True)
class DetectorConfig:
    """
    Configuration knobs. Start with these defaults and tune based on your UI scale.
    """

    # Temporal stabilization for blinking icon:
    temporal_window: int = 12  # number of frames to keep for temporal max
    use_temporal_max: bool = True

    # Template matching:
    method: int = cv2.TM_CCOEFF_NORMED
    score_threshold: float = 0.62  # if your icon is hard, you may lower to ~0.55
    scales: Tuple[float, ...] = (0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25)

    # Edge extraction:
    canny1: int = 50
    canny2: int = 140

    # Optional speed-up:
    enable_local_search: bool = False
    local_search_radius: int = 120  # pixels


class PlayerIconDetector:
    """
    Detects a small player icon in a larger map ROI using:
    - (optional) temporal max to handle blinking
    - edge-based multi-scale template matching for robustness
    """

    def __init__(
        self, template_bgr: np.ndarray, config: DetectorConfig = DetectorConfig()
    ):
        if template_bgr is None or template_bgr.size == 0:
            raise ValueError("template_bgr is empty.")
        self.cfg = config

        self.template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        self.template_edges = self._edges(self.template_gray)

        self._gray_buffer: deque[np.ndarray] = deque(maxlen=self.cfg.temporal_window)
        self._last_det: Optional[Detection] = None

    def reset(self) -> None:
        self._gray_buffer.clear()
        self._last_det = None

    def detect(self, roi_bgr: np.ndarray) -> Optional[Detection]:
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        search_gray = (
            self._apply_temporal_max(roi_gray)
            if self.cfg.use_temporal_max
            else roi_gray
        )

        if self.cfg.enable_local_search and self._last_det is not None:
            search_gray, offset_xy = self._crop_around_last(
                search_gray, self._last_det.center_xy
            )
        else:
            offset_xy = (0, 0)

        search_edges = self._edges(search_gray)

        best = self._multi_scale_match(search_edges, offset_xy)
        if best is None or best.score < self.cfg.score_threshold:
            return None

        self._last_det = best
        return best

    def _apply_temporal_max(self, gray: np.ndarray) -> np.ndarray:
        self._gray_buffer.append(gray)
        if len(self._gray_buffer) < 2:
            return gray
        stacked = np.stack(list(self._gray_buffer), axis=0)
        return stacked.max(axis=0).astype(np.uint8)

    def _edges(self, gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.Canny(blur, self.cfg.canny1, self.cfg.canny2)

    def _multi_scale_match(
        self, roi_edges: np.ndarray, offset_xy: Tuple[int, int]
    ) -> Optional[Detection]:
        best_score = -1.0
        best_bbox = None
        best_scale = 1.0

        th0, tw0 = self.template_edges.shape[:2]
        rh, rw = roi_edges.shape[:2]

        for s in self.cfg.scales:
            tw = int(round(tw0 * s))
            th = int(round(th0 * s))
            if tw < 8 or th < 8:
                continue
            if tw >= rw or th >= rh:
                continue

            tmpl = cv2.resize(
                self.template_edges, (tw, th), interpolation=cv2.INTER_NEAREST
            )

            res = cv2.matchTemplate(roi_edges, tmpl, self.cfg.method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            score = float(max_val)
            loc = max_loc

            if score > best_score:
                best_score = score
                best_bbox = (loc[0], loc[1], tw, th)
                best_scale = s

        if best_bbox is None:
            return None

        x, y, w, h = best_bbox
        ox, oy = offset_xy
        x_full, y_full = x + ox, y + oy
        cx, cy = x_full + w // 2, y_full + h // 2

        return Detection(
            bbox_xywh=(x_full, y_full, w, h),
            center_xy=(cx, cy),
            score=best_score,
            scale=best_scale,
        )

    def _crop_around_last(
        self, gray: np.ndarray, last_center_xy: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        cx, cy = last_center_xy
        r = self.cfg.local_search_radius
        h, w = gray.shape[:2]

        x1 = max(cx - r, 0)
        y1 = max(cy - r, 0)
        x2 = min(cx + r, w)
        y2 = min(cy + r, h)

        cropped = gray[y1:y2, x1:x2]
        return cropped, (x1, y1)


def draw_detection(roi_bgr: np.ndarray, det: Detection) -> np.ndarray:
    vis = roi_bgr.copy()
    x, y, w, h = det.bbox_xywh
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cx, cy = det.center_xy
    cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)
    cv2.putText(
        vis,
        f"{det.score:.2f} s={det.scale:.2f}",
        (x, max(0, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


template = cv2.imread("template.png")
if template is None:
    raise RuntimeError("template.png okunamadı. Dosya yolu dogru mu?")

detector = PlayerIconDetector(template)


# ==========================
# >>> EK: Sadece belirlenen ROI'yi tarayan canlı döngü
# ==========================

MAP_X = 372
MAP_Y = 175
MAP_W = 387
MAP_H = 421

WIN_NAME = "ROI Detection (ESC=quit)"

with mss.mss() as sct:
    monitor = sct.monitors[1]  # primary monitor

    while True:
        shot = np.array(sct.grab(monitor))  # BGRA
        frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

        # sadece harita kısmını kırp
        roi = frame[MAP_Y : MAP_Y + MAP_H, MAP_X : MAP_X + MAP_W]

        det = detector.detect(roi)

        if det is not None:
            vis = draw_detection(roi, det)

            # ekran koordinatını da göster (isteğe bağlı)
            sx = MAP_X + det.center_xy[0]
            sy = MAP_Y + det.center_xy[1]
            cv2.putText(
                vis,
                f"screen=({sx},{sy})",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            vis = roi

        cv2.imshow(WIN_NAME, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

cv2.destroyAllWindows()
