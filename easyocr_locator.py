import numpy as np

# from PIL import Image # Ne işe yarar? Görüntüyü göstermek/kaydetmek gibi işler.

# import pytesseract  # OCR (Optical Character Recognition) için kullanılan bir kütüphane. Görüntüdeki metni tanımak için kullanılır. -github
# from pytesseract import Output
# ↑↑ Tesseract yerine EasyOCR’a geçiyoruz; bu yüzden pytesseract importlarını kapattım.

import easyocr  # ✅ EasyOCR: Model bir kere yüklenir (GPU/CPU bellekte kalır), her çağrıda process aç-kapa yok.
from difflib import SequenceMatcher
from pyautogui import moveTo, click  # fare click hareket i çin
from time import sleep

from capture_service import grab_gray  # ✅ DXCam artık burada değil, servis modülünde


def sim(a, b):
    return SequenceMatcher(
        None, a, b
    ).ratio()  # SequenceMatcher(...).ratio() bu iki metnin benzerliğini verir. 0 ile 1 arasında bir değer döner, 1 tam eşleşme anlamına gelir. OCR hataları için “tolerans” sağlar.


# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # tesseract motorun yolu
# )
# ↑↑ EasyOCR kullandığımız için tesseract yoluna artık ihtiyaç yok.

# ✅ EasyOCR motoru: BUNU BİR KERE oluşturuyoruz (resident gibi).
# ['en'] -> İngilizce model. (Senin target "disassemble" olduğu için yeterli.)
# gpu=True -> CUDA varsa GPU kullanır; yoksa CPU'ya düşer.
reader = easyocr.Reader(['en'], gpu=True)

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


def get_location_from_points(target="disassemble", threshold=0.9, debug=False):
    # Bu fonksiyon main dosyasından çağırılacak.
    # Bulursa (cx_screen, cy_screen) döndürür, bulamazsa None döndürür.

    for i in range(len(points)):
        (x1, y1), (x2, y2) = points[i]  # sırayla bölgeleri almak için döngü başında

        frame = grab_gray(
            region=(x1, y1, x2, y2)
        )  # left, top, right, bottom -> (x1, y1, x2, y2) formatında bölge tanımlaması yaparak ekran görüntüsü alma

        if frame is None:  # klasik frame none kontrol
            continue

        # Image.fromarray(frame).show() resimleri kontrol için
        # tamamen kontrol amaçlı kütüphaneyi de yorum satırına aldım görüntüyü düzgün çekti mi diye kontrol amaçlı eklendi

        # text = pytesseract.image_to_string(frame, config="--psm 7")
        # --psm 7: OCR motoruna tek bir satır metin olduğunu söylüyor. bu görevlerde tek satır metinler var, bu yüzden doğruluğu artırmak için bu modu seçtim.
        # ↑↑ EasyOCR tarafında --psm yok. EasyOCR kendi detector + recognizer pipeline’ı ile çalışır.

        # data = pytesseract.image_to_data(frame, config="--psm 7", output_type=Output.DICT)
        # image_to_data, metni tanımakla kalmaz, aynı zamanda her kelimenin konumunu (bounding box) ve diğer bilgileri de verir.
        # ↑↑ EasyOCR karşılığı: reader.readtext() -> (bbox, text, confidence) listesi döner.

        # ✅ EasyOCR okuma:
        # results: [ (bbox, text, conf), ... ]
        # bbox: 4 nokta -> [[x_tl,y_tl],[x_tr,y_tr],[x_br,y_br],[x_bl,y_bl]]
        results = reader.readtext(frame, detail=1)

        # target = "disassemble"  # bu aradığım kelime. özel bir anlamı yok görevin içerisinde geçiyor sadece

        # words = [w.strip().lower() for w in data["text"]]
        # ↑↑ EasyOCR’da data["text"] yok; results içindeki text’leri toplayacağız.
        words = [str(r[1]).strip().lower() for r in results]
        # words: OCR’den gelen kelimeleri temizleyip küçük harfe çeviriyor.
        # strip() baş/son boşlukları atar.
        # lower() büyük/küçük harf farkını yok eder

        # ✅ Burada çok önemli bir bug vardı: senin kodda hem dış döngü hem iç döngü "i" kullanıyordu.
        # Yorumlarını kaybetmemek için yapıyı korudum ama iç döngüde index adını değiştirdim: word_i
        for word_i, w in enumerate(words):  # (words, start=1 yapsaydım 1. indexten  başlardı)
            # enumerate() şunu yapar: Bir iterable’ın (liste, tuple, string…) elemanlarını gezerken sana hem index’i (sıra numarası) hem de değeri birlikte verir.
            # 0 1.kelime 1 2.kelime diye sırasıyla gider index ve karşısındaki kelime

            # ✅ EasyOCR çoğu zaman tek parça string döndürür: "disassemble an item with"
            # Bu yüzden target'ı sadece benzerlik ile değil, içeriyor mu + kelime kelime benzerlik ile kontrol ediyoruz.
            hit = False
            if target in w:
                hit = True
            else:
                for token in w.split():
                    if sim(token, target) >= threshold:
                        hit = True
                        break

            if hit:  # → kelime “disassemble”e %90+ benziyorsa yakala.
                # sim 0.9 sıkı ama yine de çalışıyor
                # todo eğer görev olduğu halde okumuyorsa bunu düşür

                # ✅ EasyOCR’da left/top/width/height yok.
                # Bunun yerine bbox var: 4 köşe noktası.
                bbox = results[word_i][0]
                raw_text = str(results[word_i][1])
                conf = float(results[word_i][2])

                # bbox noktalarından ROI içi merkezi hesapla:
                xs = [float(p[0]) for p in bbox]
                ys = [float(p[1]) for p in bbox]
                cx_roi = int(sum(xs) / len(xs))
                cy_roi = int(sum(ys) / len(ys))

                if debug:
                    print("buldum (yakın eşleşme)", raw_text, "conf:", conf, "roi_center:", cx_roi, cy_roi)

                # ROI içindeki merkezi ekran koordinatına çevir:
                cx_screen = x1 + cx_roi
                cy_screen = y1 + cy_roi

                # görevin merkezi için hesaplama eklendi ama bölgeler sabit olduğu için merkezden ziyade herhangi bir noktasına tıklamak yeterli olabilir.
                # yine de merkez hesaplaması ekledim. (bence gereksiz ama ekledim)
                return (cx_screen, cy_screen)

        # Tesseract'ta text değişkeni vardı; EasyOCR'da results'tan birleştirerek benzer debug çıktısı üretelim:
        extracted_text = " ".join([str(r[1]) for r in results]).strip()
        if debug:
            print(f"Extracted Text from Region {i + 1}: {extracted_text}")

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
[EasyOCR: reader.readtext -> (bbox, text, confidence)]
        |
        v
[results içinden kelimeleri dolaş]
        |
        v
[SequenceMatcher ile "disassemble" benzerliği ölç]
        |
        v
[Benzerlik >= 0.9 ise -> bbox merkezini hesapla + ekrandaki koordinata çevir]
        """

    return None


if __name__ == "__main__":
    # Bu dosyayı tek başına çalıştırırsan test amaçlı:
    print(get_location_from_points(debug=True))