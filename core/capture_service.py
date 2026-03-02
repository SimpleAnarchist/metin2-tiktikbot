import threading
from typing import Tuple

import dxcam
import numpy as np

Region = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

_cam = None
_lock = threading.Lock()

def get_cam():
    global _cam
    if _cam is None:
        _cam = dxcam.create(output_color="GRAY")  # ✅ senin çalışan mod
    return _cam

def grab_gray(region: Region):
    cam = get_cam()
    with _lock:
        frame = cam.grab(region=region)
    if frame is None:
        return None
    return np.asarray(frame)

"""
dxcam.create(...) sadece bir kez çalışır.

İstediğin dosyada from capture_service import grab_gray deyip farklı region’lar için aynı cam’i kullanırsın.

“Manuel açık bırakma” yerine program çalıştığı sürece yaşayan tek instance modeli.

"""
