from core.capture_service import grab_gray
from core.quest_ocr import get_location_from_points
from time import sleep
# OCR lazım olunca:
loc = get_location_from_points()

# başka bir iş için frame lazım olunca:
frame = grab_gray((100, 100, 400, 200))
if frame is not None:
    # burada kendi işlemini yap (template match, hsv mask, vs)
    
    while True:
        loc = get_location_from_points()  # ✅ her seferinde yeniden hesapla

        if loc is not None:
            cx, cy = loc
            print("bulundu:", cx, cy)
        else:
            print("bulunamadı")

        sleep(0.3)  # ✅ CPU/GPU’yu yakmamak için (ör: 300ms)

        pass

