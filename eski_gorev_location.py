import dxcam  # Windows’ta ekran görüntüsü almak için hızlı bir capture kütüphanesi (DirectX tabanlı).
import numpy as np

# from PIL import Image # Ne işe yarar? Görüntüyü göstermek/kaydetmek gibi işler.
import pytesseract  # OCR (Optical Character Recognition) için kullanılan bir kütüphane. Görüntüdeki metni tanımak için kullanılır. -github
from difflib import SequenceMatcher
from pytesseract import Output
from pyautogui import moveTo, click  # fare click hareket i çin
from time import sleep


def sim(a, b):
    return SequenceMatcher(
        None, a, b
    ).ratio()  # SequenceMatcher(...).ratio() bu iki metnin benzerliğini verir. 0 ile 1 arasında bir değer döner, 1 tam eşleşme anlamına gelir. OCR hataları için “tolerans” sağlar.


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # tesseract motorun yolu
)

cam = dxcam.create(
    output_color="GRAY"
)  # ekranı gri olarak yakalıyoruz. ocr gri renkte daha stabil olacağı için gri seçtim
points = [
    ((87, 259), (255, 279)),
    ((86, 344), (337, 362)),
    ((87, 424), (238, 446)),
    ((85, 509), (198, 528)),
    ((86, 592), (291, 614)),
    ((89, 678), (374, 696)),
]  # OCR yapılacak ekran bölgeleri (x1, y1), (x2, y2) formatında. görevlerin belirli kısımlarındaki metinler okunacağı için hepsine ayrı lokasyon bilgisini girdim
# ek olarak bölgeleri küçülterek gürültüyü azaltıp doğruluğu artırdım
x1, x2, y1, y2 = points[0][0][0], points[0][1][0], points[0][0][1], points[0][1][1]
# bölge ataması örneği: points[0] -> ((87, 259), (255, 279)) -> x1=87, y1=259, x2=255, y2=279

for i in range(len(points)):
    (x1, y1), (x2, y2) = points[i]  # sırayla bölgeleri almak için döngü başında

    frame = cam.grab(
        region=(x1, y1, x2, y2)
    )  # left, top, right, bottom -> (x1, y1, x2, y2) formatında bölge tanımlaması yaparak ekran görüntüsü alma

    if frame is None:  # klasik frame none kontrol
        raise RuntimeError("Görüntü alınamadı.")

    frame = np.asarray(frame)  # dxcam’in döndürdüğü görüntüyü numpy array’e çevirme
    # (H, W, 1) -> (H, W)
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]
        # üçüncü boyutu tek kanala indirerek (H, W) formatına getiriyoruz. bu şekilde ocr için daha temiz bir görüntü elde ediyoruz.

    # Image.fromarray(frame).show() resimleri kontrol için

    # tamamen kontrol amaçlı kütüphaneyi de yorum satırına aldım görüntüyü düzgün çekti mi diye kontrol amaçlı eklendi

    text = pytesseract.image_to_string(frame, config="--psm 7")
    # --psm 7: OCR motoruna tek bir satır metin olduğunu söylüyor. bu görevlerde tek satır metinler var, bu yüzden doğruluğu artırmak için bu modu seçtim.

    # daha fazla satır varsa --psm 6 (tek blok metin) veya --psm 3 (tam sayfa) gibi modlar denenebilir. aksi taktirde psm7 bozulmaya başlar

    data = pytesseract.image_to_data(frame, config="--psm 7", output_type=Output.DICT)
    # image_to_data, metni tanımakla kalmaz, aynı zamanda her kelimenin konumunu (bounding box) ve diğer bilgileri de verir. bu sayede benzerlik kontrolü yaparken kelimenin tam konumunu da elde edebiliriz.

    target = "disassemble"  # bu aradığım kelime. özel bir anlamı yok görevin içerisinde geçiyor sadece

    words = [w.strip().lower() for w in data["text"]]
    # words: OCR’den gelen kelimeleri temizleyip küçük harfe çeviriyor.
    # strip() baş/son boşlukları atar.
    # lower() büyük/küçük harf farkını yok eder

    for i, w in enumerate(words):  # (words, start=1 yapsaydım 1. indexten  başlardı)
        # enumerate() şunu yapar: Bir iterable’ın (liste, tuple, string…) elemanlarını gezerken sana hem index’i (sıra numarası) hem de değeri birlikte verir.
        # 0 1.kelime 1 2.kelime diye sırasıyla gider index ve karşısındaki kelime

        if sim(w, target) >= 0.9:  # → kelime “disassemble”e %90+ benziyorsa yakala.
            # sim 0.9 sıkı ama yine de çalışıyor
            # todo eğer görev olduğu halde okumuyorsa bunu düşür

            x = data["left"][i]
            y = data["top"][i]

            print("buldum (yakın eşleşme)", x, y)
            w_box = data["width"][i]
            h_box = data["height"][i]

            x_screen = x1 + x
            y_screen = y1 + y
            cx_screen = x_screen + w_box // 2
            cy_screen = y_screen + h_box // 2
            # görevin merkezi için hesaplama eklendi ama bölgeler sabit olduğu için merkezden ziyade herhangi bir noktasına tıklamak yeterli olabilir. yine de merkez hesaplaması ekledim. (bence gereksiz ama ekledim)
            break
    print(f"Extracted Text from Region {i + 1}: {text.strip()}")

    # github açıklaması için aşağıdaki akış diyagramını kullanacağım unutmam umarım
    """
    [points: ekran bölgeleri]
        |
        v
[dxcam ile ROI (region) ekran görüntüsü al]
        |
        v
[numpy array'e çevir + tek kanala indir (GRAY)]
        |
        v
[tesseract OCR: image_to_string + image_to_data]
        |
        v
[data["text"] içinden kelimeleri dolaş]
        |
        v
[SequenceMatcher ile "disassemble" benzerliği ölç]
        |
        v
[Benzerlik >= 0.9 ise -> kelimenin bbox'ı + ekrandaki koordinatını hesapla]

    """
