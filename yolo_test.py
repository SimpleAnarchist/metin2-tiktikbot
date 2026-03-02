from time import sleep, perf_counter
import torch

from core.quest_ocr import get_location_from_points
from core.simyaci_yolo import detect_simyaci_once, find_simyaci_until_found

# ===================== AYARLAR =====================
MODE = "yolo_loop"  # "yolo_once" | "yolo_loop" | "yolo_find" | "ocr_loop"

MODEL_PATH = r"assets\best.pt"     # kendi modelin
GAME_REGION = (0, 0, 1920, 1080)          # mümkünse küçült (performans!)

CONF_THRES = 0.80
DEVICE = "cuda:0"
HALF = True

# yolo_loop test hızı
LOOP_SLEEP = 0.20   # 200ms (GPU/CPU'yu yakmaz, yeterince hızlıdır)

# ===================================================


def print_gpu_info():
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))


def test_yolo_once():
    hit = detect_simyaci_once(
        model_path=MODEL_PATH,
        region=GAME_REGION,
        target="simyaci",
        conf_thres=CONF_THRES,
        device=DEVICE,
        half=HALF,
        require_cuda=True,
        warmup=True,     # ilk seferde iyi
        debug=True
    )
    if hit is None:
        print("YOLO: bulunamadı")
    else:
        print("YOLO: bulundu ->", hit)


def test_yolo_loop():
    """
    Görüyor mu görmüyor mu testinin en iyi hali:
    - sürekli dener
    - bulundu/bulunamadı basar
    - yaklaşık FPS gibi de ölçer
    """
    print("YOLO loop başladı. Çıkmak için Ctrl+C.")
    warmup_done = False

    try:
        while True:
            t0 = perf_counter()

            hit = detect_simyaci_once(
                model_path=MODEL_PATH,
                region=GAME_REGION,
                target="simyaci",
                conf_thres=CONF_THRES,
                device=DEVICE,
                half=HALF,
                require_cuda=True,
                warmup=(not warmup_done),
                debug=False
            )
            warmup_done = True

            dt = (perf_counter() - t0) * 1000.0

            if hit is None:
                print(f"[{dt:6.1f} ms] YOLO: bulunamadı")
            else:
                cx, cy = hit.center
                print(f"[{dt:6.1f} ms] YOLO: bulundu conf={hit.conf:.3f} center=({cx},{cy})")

            sleep(LOOP_SLEEP)

    except KeyboardInterrupt:
        print("YOLO loop durduruldu.")


def test_yolo_find_until_found():
    """
    Gerçek davranış:
    - bulamazsa 10x q basar
    - bulana kadar döner
    """
    print("YOLO find başladı. (Oyun penceresi odakta olmalı) Çıkmak için Ctrl+C.")
    hit = find_simyaci_until_found(
        model_path=MODEL_PATH,
        region=GAME_REGION,
        target="simyaci",
        conf_thres=CONF_THRES,
        device=DEVICE,
        half=HALF,
        require_cuda=True,
        debug=True
    )
    print("Simyacı bulundu:", hit)


def test_ocr_loop():
    print("OCR loop başladı. Çıkmak için Ctrl+C.")
    try:
        while True:
            loc = get_location_from_points()
            if loc is not None:
                cx, cy = loc
                print("OCR bulundu:", cx, cy)
            else:
                print("OCR bulunamadı")
            sleep(0.3)
    except KeyboardInterrupt:
        print("OCR loop durduruldu.")


if __name__ == "__main__":
    print_gpu_info()

    if MODE == "yolo_once":
        test_yolo_once()
    elif MODE == "yolo_loop":
        test_yolo_loop()
    elif MODE == "yolo_find":
        test_yolo_find_until_found()
    elif MODE == "ocr_loop":
        test_ocr_loop()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")