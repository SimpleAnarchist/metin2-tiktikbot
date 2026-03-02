from core.simyaci_yolo import find_simyaci_until_found

MODEL_PATH = r"C:\Users\boran\Desktop\image detection\metin2\assets\best.pt"
REGION = (0, 0, 1920, 1080)  # mümkünse küçült

hit = detect_simyaci_once(
    model_path=MODEL_PATH,
    region=REGION,
    conf_accept=0.25,
    conf_search=0.05,
    device=0,
    half=True,
    debug=True
)
cx, cy = hit.center
print("Simyacı bulundu:", (cx, cy), "conf:", hit.conf)