import dxcam
import numpy as np
import threading

_cam = None
_lock = threading.Lock()

def get_cam():
    global _cam
    if _cam is None:
        _cam = dxcam.create(output_color="GRAY")
    return _cam

def grab_gray(region):
    """
    region: (x1, y1, x2, y2)  # left, top, right, bottom
    return: frame (H,W) numpy grayscale veya None
    """
    cam = get_cam()
    with _lock:  # aynı anda iki yer grab çağırırsa çakışmasın diye
        #Not: _lock önemli. Main bir yandan başka iş için grab yaparken OCR dosyası da grab yaparsa dxcam bazen None/verimsizleşebiliyor. Lock bunu toparlar.

        frame = cam.grab(region=region)

    if frame is None:
        return None

    frame = np.asarray(frame)

    # (H, W, 1) -> (H, W)
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]

    return frame


"""
dxcam.create(...) sadece bir kez çalışır.

İstediğin dosyada from capture_service import grab_gray deyip farklı region’lar için aynı cam’i kullanırsın.

“Manuel açık bırakma” yerine program çalıştığı sürece yaşayan tek instance modeli.

"""
