import cv2
import numpy  # cv2 ve numpy temel görüntü işleme ve matris işlemleri için
import dxcam  # DirectX ekran yakalama, OpenCV ile birlikte kullanmak için) kütüphaneyi yaznan kişi baya sağlam şekilde yazmış obs den fazlasıyla esinlenilmiş
from paddle import th_dll_path
import pyautogui

""" PROJE TEMEL AMACI YAPILACAK ŞEYLERİ EN OPTİMİZE HALE GETİRMEK, BU YÜZDEN KÜTÜPHAELER ÖNEMLİ BİR NOKTA, PYAUTOGUI BAZEN HIZLI HAREKETLERDE TAKILABİLİR, BU DURUMDA PYWIN32 GİBİ DAHA DÜŞÜK SEVİYE KÜTÜPHANELER DENENEBİLİR"""

MAP_X = 950
MAP_Y = 409
MAP_W = 242
MAP_H = 434
REGION = (MAP_X, MAP_Y, MAP_X + MAP_W, MAP_Y + MAP_H)  # left, top, right, bottom

# konumlar sabit, bu yüzden ekranın neresinde olursa olsun aynı bölgeyi taranacak
# MAP_X VE MAP_Y TARANACAK BÖLGENİN EN SOL PİKSELİ
# MAP_W GENİŞLİK, MAP_H YÜKSEKLİK
# DXCAM REGION FORMATI (LEFT, TOP, RIGHT, BOTTOM) OLDUĞU İÇİN MAP_X + MAP_W VE MAP_Y + MAP_H KULLANILIR

TEMPLATE_PATH = r"item_template.png"  # RESMİN OLDUĞU YER
tmpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
# RESMİ GRİ TONLARINA ÇEVİRMEK İÇİN
# Template matching griyle daha hızlı ve stabil olur (çoğu durumda).

if tmpl is None:
    raise RuntimeError("Template okunamadı: item_template.png")

th, tw = tmpl.shape[:2]  # RESMİN YÜKSEKLİĞİ VE GENİŞLİĞİ

# coarse to fine ayarları (hız için) önce küçükte bul, sonra büyüğünde netleştir

scale = 0.5
