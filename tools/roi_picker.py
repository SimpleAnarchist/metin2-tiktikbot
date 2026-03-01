import pyautogui
import time

print("5 saniye içinde haritanın SOL-ÜST köşesine mouse'u götür...")
time.sleep(5)
x1, y1 = pyautogui.position()
print("TOP_LEFT:", x1, y1)

print("5 saniye içinde haritanın SAĞ-ALT köşesine mouse'u götür...")
time.sleep(5)
x2, y2 = pyautogui.position()
print("BOTTOM_RIGHT:", x2, y2)

x_min, x_max = sorted([x1, x2])
y_min, y_max = sorted([y1, y2])

print("\nROI:")
print("MAP_X =", x_min)
print("MAP_Y =", y_min)
print("MAP_W =", x_max - x_min)
print("MAP_H =", y_max - y_min)
