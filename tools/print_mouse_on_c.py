import pyautogui
import keyboard

print("Hazır. 'c' -> mouse (x,y) yazdırır | 'q' -> çıkış")

while True:
    if keyboard.is_pressed("c"):
        x, y = pyautogui.position()
        print(f"MOUSE: {x} {y}")
        keyboard.wait("c")  # tuşu bırakmadan spam olmasın

    if keyboard.is_pressed("q"):
        break


# MOUSE: 1088 921 ENVANTER
# MOUSE: 1030 911 KARAKTER
# MOUSE: 382 783 GÖREVLER
# MOUSE: 159 498 SİMYACI GÖREV SEKMESİ
# MOUSE: 391 713 GÖREVLERİN EN AŞAĞISI
# MOUSE: 1186 918 ESC
# MOUSE: 625 532 KARAKTER DEĞİŞTİRME BUTONU
# MOUSE: 223 486 DİĞER KARAKTERE GEÇİŞ
# MOUSE: 557 867 KARAKTER BAŞLATMA