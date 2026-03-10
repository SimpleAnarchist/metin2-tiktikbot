# c_ile_bolge_ocr.py

import re
import time
import tkinter as tk
import numpy as np
import cv2
import mss
import pytesseract
import keyboard
from PIL import Image
from ctypes import windll, byref
from ctypes.wintypes import POINT

# --------------------------------------------------
# DPI scaling problemi olmasin
# --------------------------------------------------
try:
    windll.user32.SetProcessDPIAware()
except:
    pass

# --------------------------------------------------
# Tesseract yolu
# Kendi sistemine gore degistir
# --------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# OCR dili
OCR_LANG = "tur"

# OCR config
# psm 6 = birden fazla satir iceren tek metin blogu gibi oku
OCR_CONFIG = r"--oem 3 --psm 6"


def get_mouse_pos():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return pt.x, pt.y


def normalize_region(p1, p2):
    x1 = min(p1[0], p2[0])
    y1 = min(p1[1], p2[1])
    x2 = max(p1[0], p2[0])
    y2 = max(p1[1], p2[1])

    w = x2 - x1
    h = y2 - y1

    return x1, y1, w, h


def capture_region(region):
    x, y, w, h = region

    if w <= 0 or h <= 0:
        return None

    with mss.mss() as sct:
        monitor = {
            "left": x,
            "top": y,
            "width": w,
            "height": h
        }
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        return img


def preprocess_for_ocr(pil_img):
    img = np.array(pil_img)

    # RGB -> Gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Kucuk yazilari buyut
    gray = cv2.resize(
        gray,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    # Hafif blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh


def clean_lines(text):
    raw_lines = text.splitlines()
    result = []

    for line in raw_lines:
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            result.append(line)

    return result


class CKeyOCRApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("C ile Bolge OCR")
        self.root.geometry("470x190")
        self.root.resizable(False, False)

        self.start_point = None
        self.overlay = None
        self.canvas = None
        self.rect_id = None
        self.running_overlay_loop = False

        self.status_var = tk.StringVar()
        self.status_var.set("Hazir. C ile secim baslat.")

        title = tk.Label(
            self.root,
            text="C ile Bolge OCR",
            font=("Consolas", 14, "bold")
        )
        title.pack(pady=(12, 6))

        info = tk.Label(
            self.root,
            text=(
                "Kullanim:\n"
                "C -> baslangic noktasi al\n"
                "C -> bitis noktasi al + OCR yap\n"
                "Q -> cikis\n\n"
                "Secim sirasinda fareyi hareket ettirdikce\n"
                "kirmizi cerceve canli gorunur."
            ),
            justify="left",
            font=("Consolas", 10)
        )
        info.pack(padx=14, anchor="w")

        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            fg="blue",
            font=("Consolas", 10, "bold")
        )
        status.pack(padx=14, pady=10, anchor="w")

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(padx=14, pady=8, anchor="w")

        tk.Button(
            btn_frame,
            text="Secim Baslat / Bitir (C)",
            width=24,
            command=self.handle_c_press
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            btn_frame,
            text="Cikis (Q)",
            width=12,
            command=self.shutdown
        ).pack(side="left")

        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)

        # Tk icinde de tuslar calissin
        self.root.bind("<c>", lambda event: self.handle_c_press())
        self.root.bind("<C>", lambda event: self.handle_c_press())
        self.root.bind("<q>", lambda event: self.shutdown())
        self.root.bind("<Q>", lambda event: self.shutdown())

        # Global hotkey
        keyboard.add_hotkey("c", self.handle_c_press, suppress=False)
        keyboard.add_hotkey("q", self.shutdown, suppress=False)

    def set_status(self, msg):
        self.status_var.set(msg)
        print(f"[INFO] {msg}")

    def create_overlay(self):
        if self.overlay is not None:
            return

        self.overlay = tk.Toplevel(self.root)
        self.overlay.attributes("-fullscreen", True)
        self.overlay.attributes("-topmost", True)
        self.overlay.attributes("-alpha", 0.15)
        self.overlay.configure(bg="black")
        self.overlay.overrideredirect(True)

        self.canvas = tk.Canvas(
            self.overlay,
            bg="black",
            highlightthickness=0,
            cursor="cross"
        )
        self.canvas.pack(fill="both", expand=True)

        self.running_overlay_loop = True
        self.update_rectangle_loop()

    def destroy_overlay(self):
        self.running_overlay_loop = False

        if self.overlay is not None:
            try:
                self.overlay.destroy()
            except:
                pass

        self.overlay = None
        self.canvas = None
        self.rect_id = None

    def update_rectangle_loop(self):
        if not self.running_overlay_loop:
            return

        if self.overlay is None or self.canvas is None or self.start_point is None:
            return

        current_mouse = get_mouse_pos()

        x1, y1 = self.start_point
        x2, y2 = current_mouse

        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)

        self.rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="red",
            width=3
        )

        self.overlay.after(16, self.update_rectangle_loop)

    def handle_c_press(self):
        # Ilk C: baslangic noktasi
        if self.start_point is None:
            self.start_point = get_mouse_pos()
            self.create_overlay()

            self.set_status(
                f"Baslangic alindi: {self.start_point}. "
                "Fareyi gotur, tekrar C bas."
            )
            return

        # Ikinci C: bitis noktasi + OCR
        end_point = get_mouse_pos()
        region = normalize_region(self.start_point, end_point)

        self.set_status(
            f"Bitis alindi: {end_point}. OCR baslatiliyor..."
        )

        self.destroy_overlay()
        self.run_ocr(region)
        self.start_point = None

        self.set_status("OCR tamamlandi. Yeni secim icin C bas.")

    def run_ocr(self, region):
        x, y, w, h = region

        print("\n==============================================")
        print(f"[OCR ALANI] x={x}, y={y}, w={w}, h={h}")
        print("==============================================")

        if w < 5 or h < 5:
            print("[HATA] Secilen alan cok kucuk.")
            return

        screenshot = capture_region(region)
        if screenshot is None:
            print("[HATA] Ekran goruntusu alinamadi.")
            return

        processed = preprocess_for_ocr(screenshot)

        text = pytesseract.image_to_string(
            processed,
            lang=OCR_LANG,
            config=OCR_CONFIG
        )

        lines = clean_lines(text)

        print("\n========== SATIR SATIR OCR SONUCU ==========")
        if not lines:
            print("[SONUC] Metin bulunamadi.")
        else:
            for i, line in enumerate(lines, start=1):
                print(f"{i}. {line}")
        print("============================================\n")

    def shutdown(self):
        try:
            keyboard.unhook_all_hotkeys()
        except:
            pass

        self.destroy_overlay()

        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

    def run(self):
        self.root.mainloop()


def main():
    app = CKeyOCRApp()
    app.run()


if __name__ == "__main__":
    main()