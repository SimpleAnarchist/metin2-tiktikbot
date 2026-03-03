from core.yolov8_detector import detect_yolov8_center, close_debug_windows

try:
    while True:
        # conf_thres'ı 0.30 yerine 0.15 civarı tut (sen zaten “eskiden 0.8 conf görüyordum” diyorsun;
        # threshold yüksek olunca “hiç bulmuyor” hissi olur)
        pt = detect_yolov8_center(conf_thres=0.65, imgsz=640, bigger_imgsz=960,debug_show=True)

        if pt is not None:
            print("Buldum:", pt)

except KeyboardInterrupt:
    pass
finally:
    close_debug_windows()