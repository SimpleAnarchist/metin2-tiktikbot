from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

from core.capture_service import grab_gray  # shared dxcam gray service

Region = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

# (template_path, scale) -> (tmpl_full, tmpl_small, th, tw)
_TEMPLATE_CACHE: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, int, int]] = {}


def _resolve_template_path(template_name: str, templates_dir: str | None) -> str:
    """
    template_name: "x.png" gibi
    templates_dir: None ise proje kökü /assets altı varsayılır
    """
    p = Path(template_name)
    if p.is_absolute():
        return str(p)

    if templates_dir is None:
        # archive/ altındayız -> 1 üst: proje kökü
        base = Path(__file__).resolve().parents[1] / "assets"
    else:
        base = Path(templates_dir)

    return str((base / template_name).resolve())


def _load_template(template_path: str, scale: float) -> tuple[np.ndarray, np.ndarray, int, int]:
    key = (template_path, float(scale))
    if key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[key]

    tmpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        raise RuntimeError(f"Template okunamadı: {template_path}")

    th, tw = tmpl.shape[:2]
    tmpl_small = cv2.resize(tmpl, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    _TEMPLATE_CACHE[key] = (tmpl, tmpl_small, th, tw)
    return tmpl, tmpl_small, th, tw


def find_template_center_once(
    region: Region,
    template_name: str,
    templates_dir: str | None = None,   # assets altını otomatik kullanır
    scale: float = 0.5,
    pad: int = 60,
    thr_coarse: float = 0.65,
    thr_fine: float = 0.80,
) -> Optional[Tuple[int, int]]:
    """
    1 kere dener:
      - Bulursa: (cx_screen, cy_screen)
      - Bulamazsa: None

    template_name: "x.png" gibi (assets altında aranır) veya tam path
    """
    template_path = _resolve_template_path(template_name, templates_dir)
    tmpl, tmpl_small, th, tw = _load_template(template_path, scale)

    frame = grab_gray(region=region)
    if frame is None:
        return None

    # dxcam bazen (H,W,1) dönebilir
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]

    frame = np.ascontiguousarray(frame)
    H, W = frame.shape[:2]
    if H < th or W < tw:
        return None

    method = cv2.TM_CCOEFF_NORMED

    # 1) coarse (small)
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    res_s = cv2.matchTemplate(small, tmpl_small, method)
    _, maxVal_s, _, maxLoc_s = cv2.minMaxLoc(res_s)
    if maxVal_s < thr_coarse:
        return None

    # 2) approx -> refine window
    approx_x = int(maxLoc_s[0] / scale)
    approx_y = int(maxLoc_s[1] / scale)

    x0 = max(approx_x - pad, 0)
    y0 = max(approx_y - pad, 0)
    x1 = min(approx_x + pad + tw, W)
    y1 = min(approx_y + pad + th, H)

    search = frame[y0:y1, x0:x1]
    if search.shape[0] < th or search.shape[1] < tw:
        return None

    res = cv2.matchTemplate(search, tmpl, method)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
    if maxVal < thr_fine:
        return None

    # ROI içi sol-üst
    x_roi = x0 + maxLoc[0]
    y_roi = y0 + maxLoc[1]

    # ROI -> screen
    rx1, ry1, _, _ = region
    x_screen = rx1 + x_roi
    y_screen = ry1 + y_roi

    cx_screen = x_screen + tw // 2
    cy_screen = y_screen + th // 2
    return (cx_screen, cy_screen)


if __name__ == "__main__":
    # hızlı test (kendi region ve template ile)
    # örnek:
    REGION = (15, 50, 15 + 1193, 50 + 892)
    print(find_template_center_once(REGION, "x.png"))