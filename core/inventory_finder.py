from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

import cv2
import numpy as np

from core.capture_service import grab_gray  # shared dxcam gray service :contentReference[oaicite:3]{index=3}


Region = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class _TemplateCacheEntry:
    tmpl: np.ndarray
    tmpl_small: np.ndarray
    th: int
    tw: int
    scale: float
    mtime: float


# (abs_path, scale) -> cache entry
_TEMPLATE_CACHE: Dict[Tuple[str, float], _TemplateCacheEntry] = {}


def _default_assets_dir() -> Path:
    # core/ altındayız -> proje kökü = parents[1]
    return Path(__file__).resolve().parents[1] / "assets"


def _resolve_template_path(template_name_or_path: str, templates_dir: Optional[Union[str, Path]] = None) -> str:
    """
    template_name_or_path:
      - "x.png" gibi (assets altında aranır)
      - veya tam path
    """
    p = Path(template_name_or_path)
    if p.is_absolute():
        return str(p)

    base = _default_assets_dir() if templates_dir is None else Path(templates_dir)
    return str((base / template_name_or_path).resolve())


def _load_template_cached(template_path: str, scale: float) -> _TemplateCacheEntry:
    """
    Template'i (gray) yükler + scale edilmiş halini hazırlar.
    Dosya değiştiyse otomatik reload eder.
    """
    key = (template_path, float(scale))
    mtime = 0.0
    try:
        mtime = Path(template_path).stat().st_mtime
    except Exception:
        pass

    if key in _TEMPLATE_CACHE:
        entry = _TEMPLATE_CACHE[key]
        # dosya değişmediyse cache kullan
        if abs(entry.mtime - mtime) < 1e-6:
            return entry

    tmpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        raise RuntimeError(f"Template okunamadı: {template_path}")

    th, tw = tmpl.shape[:2]
    tmpl_small = cv2.resize(tmpl, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    entry = _TemplateCacheEntry(tmpl=tmpl, tmpl_small=tmpl_small, th=th, tw=tw, scale=float(scale), mtime=mtime)
    _TEMPLATE_CACHE[key] = entry
    return entry


def find_template_center_once(
    region: Region,
    template: str,
    templates_dir: Optional[Union[str, Path]] = None,
    scale: float = 0.5,
    pad: int = 60,
    thr_coarse: float = 0.65,
    thr_fine: float = 0.80,
    return_score: bool = False,
) -> Optional[Tuple[int, int]] | Optional[Tuple[int, int, float]]:
    """
    1 kere dene:
      - Bulursa ekran koordinatında center döndürür
      - Bulamazsa None döndürür

    region: (x1,y1,x2,y2)
    template: "x.png" (assets altında) veya tam path
    """
    template_path = _resolve_template_path(template, templates_dir)
    entry = _load_template_cached(template_path, scale)

    frame = grab_gray(region=region)
    if frame is None:
        return None

    # dxcam griyse (H,W,1) döndürebilir
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]

    frame = np.ascontiguousarray(frame)
    H, W = frame.shape[:2]
    if H < entry.th or W < entry.tw:
        return None

    METHOD = cv2.TM_CCOEFF_NORMED

    # 1) SMALL tarama (hızlı)
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    res_s = cv2.matchTemplate(small, entry.tmpl_small, METHOD)
    _, maxVal_s, _, maxLoc_s = cv2.minMaxLoc(res_s)

    if maxVal_s < thr_coarse:
        return None

    # 2) SMALL -> FULL yaklaşık konum
    approx_x = int(maxLoc_s[0] / scale)
    approx_y = int(maxLoc_s[1] / scale)

    # 3) FULL’da sadece yakın çevrede refine
    x0 = max(approx_x - pad, 0)
    y0 = max(approx_y - pad, 0)
    x1 = min(approx_x + pad + entry.tw, W)
    y1 = min(approx_y + pad + entry.th, H)

    search = frame[y0:y1, x0:x1]
    if search.shape[0] < entry.th or search.shape[1] < entry.tw:
        return None

    res = cv2.matchTemplate(search, entry.tmpl, METHOD)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(res)

    if maxVal < thr_fine:
        return None

    # ROI içi top-left
    x_roi = x0 + maxLoc[0]
    y_roi = y0 + maxLoc[1]

    # EKRAN koordinatı: region offset'iyle çevir (eski dosyada MAP_X/MAP_Y sabitti) :contentReference[oaicite:4]{index=4}
    rx1, ry1, _, _ = region
    x_screen = rx1 + x_roi
    y_screen = ry1 + y_roi

    cx_screen = x_screen + entry.tw // 2
    cy_screen = y_screen + entry.th // 2

    if return_score:
        return (cx_screen, cy_screen, float(maxVal))
    return (cx_screen, cy_screen)


# Backward-friendly isim (istersen kullan)
def find_inventory_center_once(
    region: Region,
    template: str,
    templates_dir: Optional[Union[str, Path]] = None,
    **kwargs,
):
    return find_template_center_once(region=region, template=template, templates_dir=templates_dir, **kwargs)


if __name__ == "__main__":
    # hızlı test:
    # REGION = (15, 50, 15 + 1193, 50 + 892)
    # print(find_template_center_once(REGION, "x.png", return_score=True))
    pass