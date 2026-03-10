# PROJE TAMAMEN WINDOWS ÜZERİNDE ÇALIŞACAK ŞEKİLDE AYARLANDI

from __future__ import annotations

from time import sleep
from typing import Optional, Tuple

import click
import cv2
import numpy as np
import pyautogui
from rsa import cli

from core.capture_service import grab_gray
from core.quest_ocr import get_location_from_points

# YOLO (MSS + renkli) — capture_service kullanmaz
from core.yolov8_detector import detect_yolov8_center, close_debug_windows, shutdown_yolov8

a = input("Kaç tane id ile giriş yapacaksınız? (1-9999): ")

try:
    a = int(a)
    print(f"Giriş yapmak istediğiniz id sayısı: {a}")
except ValueError:
    print("Geçersiz giriş! Lütfen bir sayı girin.")
    exit(1)


a = max(1, min(a, 9999))  # 1 ile 9999 arasında sınırla github önerdi ama hiçbir mantığı yok ama hoşuma gitti  böyle bir fonksiyonun olması

# ========= CONFIG =========
MODEL_PATH = r"assets\best.pt"

# roi_picker çıktın (oyun içeriği)
MAP_X, MAP_Y, MAP_W, MAP_H = 15, 50, 1193, 892
GAME_ROI = {"left": MAP_X, "top": MAP_Y, "width": MAP_W, "height": MAP_H}

# inventory template
INV_REGION = (MAP_X, MAP_Y, MAP_X + MAP_W, MAP_Y + MAP_H)  # dxcam region=(x1,y1,x2,y2)

# YOLO retry
YOLO_TRIES = 6
YOLO_DELAY = 0.12


# ========= HELPERS =========
def click_center(pt: Tuple[int, int], move: bool = True):
    x, y = pt
    if move:
        pyautogui.moveTo(x, y, duration=0.1)
        
    pyautogui.click(duration=0.1)


def find_inventory_center_once(
    region: Tuple[int, int, int, int],
    template_path: str,
    scale: float = 0.5,
    pad: int = 60,
    thr_coarse: float = 0.65,
    thr_fine: float = 0.80,
) -> Optional[Tuple[int, int]]:
    """capture_service(grab_gray) ile 1 kere template match dene -> center veya None."""
    tmpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        raise RuntimeError(f"Template okunamadı: {template_path}")

    th, tw = tmpl.shape[:2]
    method = cv2.TM_CCOEFF_NORMED
    tmpl_small = cv2.resize(tmpl, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    frame = grab_gray(region=region)
    if frame is None:
        return None

    # (H,W,1) -> (H,W)
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]
    frame = np.ascontiguousarray(frame)

    H, W = frame.shape[:2]
    if H < th or W < tw:
        return None

    # coarse
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    res_s = cv2.matchTemplate(small, tmpl_small, method)
    _, maxVal_s, _, maxLoc_s = cv2.minMaxLoc(res_s)
    if maxVal_s < thr_coarse:
        return None

    # approx -> refine window
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

    x_roi = x0 + maxLoc[0]
    y_roi = y0 + maxLoc[1]

    rx1, ry1, _, _ = region
    x_screen = rx1 + x_roi
    y_screen = ry1 + y_roi

    cx_screen = x_screen + tw // 2
    cy_screen = y_screen + th // 2
    return (cx_screen, cy_screen)


def get_simyaci_center_retry_mss(
    tries: int = YOLO_TRIES,
    delay_s: float = YOLO_DELAY,
    debug_show: bool = False,
) -> Optional[Tuple[int, int]]:
    """
    YOLO: MSS+renkli capture ile çalışır (capture_service kullanmaz).
    Bulursa center, bulamazsa None.
    """
    for _ in range(tries):
        pt = detect_yolov8_center(
            roi=GAME_ROI,
            model_path=MODEL_PATH,
            conf_thres=0.15,
            imgsz=640,
            bigger_imgsz=960,
            debug_show=debug_show,
        )
        if pt is not None:
            return pt
        sleep(delay_s)
    return None

def ask_int(msg: str) -> int:
    while True:
        s = input(msg).strip()
        try:
            return int(s)
        except ValueError:
            print("Sayı girmen lazım (örn: 0, 1, 2 ...)")

def karakterdegis(sayac2,toplamsayi2):
    click_center((1186, 918)) # ESC
    sleep(0.1)
    click_center((625, 532)) # karakter değiştirme butonu
    if toplamsayi2 == sayac2:
        print("Tüm karakterler denendi, program sonlandırılıyor.")
        exit(0)
    else:
        return sayac2 + 1
    


# ========= MAIN LOOP =========
def main():
    toplamsayi = ask_int("Kaç id ile giriş yapacaksınız? (1-9999): ")
    sayac = 0
    pyautogui.PAUSE = 0.1
    while True:
        sayac =+ 1
        INV_TEMPLATE_PATH = r"assets\giris.png"
        while True:
            inv_pt = find_inventory_center_once(region=INV_REGION, template_path=INV_TEMPLATE_PATH)
            if inv_pt is not None:
                print("Giriş ekranı bulundu, tıklanıyor...")
                click_center(inv_pt)
                break
            else:
                print("Giriş ekranı bulunamadı, tekrar denenecek... (10 saniye sonra)")
                sleep(10)

        INV_TEMPLATE_PATH = r"assets\girisyapilmis.png"
        while True:
            inv_pt = find_inventory_center_once(region=INV_REGION, template_path=INV_TEMPLATE_PATH)
            if inv_pt is not None:
                print("Karakter oyuna giriş yaptı.")
                click_center(inv_pt)
                sleep(0.5)
                break
            else:
                print("Giriş ekranı bulunamadı, tekrar denenecek... (1 saniye sonra)")
                sleep(1)
        
        # MOUSE: 382 783 GÖREVLER
        click_center((382, 783)) # görev sekmesi açılır

        sleep(0.5)

        click_center((159, 498)) # simyacı görev sekmesi açılır
        sleep(0.1)

        click_center((391, 713)) # görevlerin en aşağısı açılır


        # 1) QUEST OCR -> bulursa tıkla, bulamazsa else boş
        loc = get_location_from_points(debug=False)
        if loc is not None:
            print("QUEST OCR simyaci location:", loc)
            print("Simyacı görevi bulunuyor")

            # MOUSE: 1088 921 ENVANTER
            click_center((1088, 921))
            sleep(0.2)
            # 2) INVENTORY FIND -> bulursa tıkla, bulamazsa else boş
            INV_TEMPLATE_PATH = r"assets\item_template.png"
            inv_pt = find_inventory_center_once(region=INV_REGION, template_path=INV_TEMPLATE_PATH)
            if inv_pt is not None:
                click_center(inv_pt)
            else:
                print("envanterde silah bulunamadı. diğer karaktere geçiliyor")
                karakterdegis(sayac,toplamsayi) # karakter değiş attığında karakter değiş fonksiyonu çalışacak ve daha sonrasında while başa dönecek
                break 

            # 3) YOLO (MSS) -> biraz dene, bulursa söyle (istersen tıkla), bulamazsa else boş
            simyaci_pt = get_simyaci_center_retry_mss(tries=YOLO_TRIES, delay_s=YOLO_DELAY, debug_show=True)
            if simyaci_pt is not None:
                print("YOLO simyaci center:", simyaci_pt)
                # buraya INV_TEMPLATE_PATH = r"assets\item_template.png" ile bulduğumuz envanter öğesini tıklayarak simyacı bulduğu yere sürüklemesini istiyorum
                # click_center(simyaci_pt)  # istersen aç
                shutdown_yolov8()  # MSS+renkli modda sürekli çalıştırmak stabilite sorunlarına yol açabilir, o yüzden kapatıyorum
            else:

                shutdown_yolov8()

            sleep(0.25)
        else:
            print("görev bulunamadı. diğer karaktere geçiliyor")
            continue # görev bulunamazsa diğer karaktere geçmek için while başa döner

        # ---- buraya kendi işlemlerini eklersin ----




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        close_debug_windows()



# MOUSE: 1088 921 ENVANTER
# MOUSE: 1030 911 KARAKTER
# MOUSE: 382 783 GÖREVLER
# MOUSE: 159 498 SİMYACI GÖREV SEKMESİ
# MOUSE: 391 713 GÖREVLERİN EN AŞAĞISI
# MOUSE: 1186 918 ESC
# MOUSE: 625 532 KARAKTER DEĞİŞTİRME BUTONU
# MOUSE: 223 486 DİĞER KARAKTERE GEÇİŞ
# MOUSE: 557 867 KARAKTER BAŞLATMA
# MOUSE: 610 406 itemi simyacınınn üzerine sürükledikten sonra onaylama tıklaması