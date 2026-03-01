import cv2
import numpy as np
import dxcam  # DirectX ekran yakalama, OpenCV ile birlikte kullanmak için) kütüphaneyi yaznan kişi baya sağlam şekilde yazmış obs den fazlasıyla esinlenilmiş
import pyautogui  # fare haraketi için

MAP_X = 959
MAP_Y = 409
MAP_W = 242
MAP_H = 434
REGION = (MAP_X, MAP_Y, MAP_X + MAP_W, MAP_Y + MAP_H)  # left, top, right, bottom
# konumlar sabit, bu yüzden ekranın neresinde olursa olsun aynı bölgeyi taranacak

TEMPLATE_PATH = r"item_template.png"

tmpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
if tmpl is None:
    raise RuntimeError("Template okunamadı: item_template.png")

th, tw = tmpl.shape[:2]

# --- Coarse-to-fine ayarları (hız için) ---
SCALE = 0.5  # 0.5 -> 4x daha az piksel taranır
PAD = 60  # refine arama payı (piksel)
THR_COARSE = 0.65  # küçükte eşik (daha düşük tut)
THR_FINE = 0.80  # büyüğünde eşik (daha yüksek tut)
METHOD = cv2.TM_CCOEFF_NORMED

tmpl_small = cv2.resize(tmpl, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

cam = dxcam.create(output_color="GRAY")

while True:
    frame = cam.grab(region=REGION)
    if frame is None:
        continue

    frame = np.asarray(frame)
    # dxcam griyse (H,W,1) döndürebilir, bu durumda tek kanala indirgemeliyiz
    if frame.ndim == 3 and frame.shape[2] == 1:  # (H, W, 1) -> (H, W)
        frame = frame[:, :, 0]
    frame = np.ascontiguousarray(frame)

    H, W = frame.shape[:2]
    if H < th or W < tw:
        continue

    # 1) SMALL tarama (hızlı)
    small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    res_s = cv2.matchTemplate(small, tmpl_small, METHOD)
    _, maxVal_s, _, maxLoc_s = cv2.minMaxLoc(res_s)

    if maxVal_s < THR_COARSE:
        continue

    # 2) SMALL -> FULL yaklaşık konum
    approx_x = int(maxLoc_s[0] / SCALE)
    approx_y = int(maxLoc_s[1] / SCALE)

    # 3) FULL’da sadece yakın çevrede refine
    x0 = max(approx_x - PAD, 0)
    y0 = max(approx_y - PAD, 0)
    x1 = min(approx_x + PAD + tw, W)
    y1 = min(approx_y + PAD + th, H)

    search = frame[y0:y1, x0:x1]
    if search.shape[0] < th or search.shape[1] < tw:
        continue

    res = cv2.matchTemplate(search, tmpl, METHOD)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(res)

    if maxVal < THR_FINE:
        continue

    # BULUNAN bbox (ROI içi)
    x_roi = x0 + maxLoc[0]
    y_roi = y0 + maxLoc[1]

    # EKRAN koordinatı (senin ROI offset’i)
    x_screen = MAP_X + x_roi
    y_screen = MAP_Y + y_roi

    cx_screen = x_screen + tw // 2
    cy_screen = y_screen + th // 2

    print(
        f"FOUND score={maxVal:.3f}  top-left=({x_screen},{y_screen})  center=({cx_screen},{cy_screen})"
    )
    pyautogui.moveTo(cx_screen, cy_screen)
    # ilk bulduğunda durmak istersen:
    break
