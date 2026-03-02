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

from core.simyaci_yolo import find_simyaci_until_found

import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

MODEL_PATH = r"C:\path\to\simyaci.pt"   # kendi modelin
GAME_REGION = (0, 0, 1920, 1080)       # oyunun ekran bölgesi (ROI)

hit = find_simyaci_until_found(
    model_path=MODEL_PATH,
    region=GAME_REGION,
    target="simyaci",
    conf_thres=0.80,
    device="cuda:0",
    half=True,
    debug=True
)

print("Simyacı bulundu:", hit)
# hit.center -> main'de tıklama/aksiyon vs.