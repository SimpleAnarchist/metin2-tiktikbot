# metin2-tiktikbot

Windows üzerinde ekran yakalama + görüntü işleme pipeline denemeleri.
Ana hedef: oyundan (veya ekrandan) belirli bölgeleri yakalayıp OCR/template ile bilgi çıkarmak ve bunu main akışına bağlamak.

## Repo yapısı (önerilen okuma sırası)

- `main2.py` -> Ana giriş / deneme akışı (buradan devam ediliyor)
- `core/` -> Tekrar kullanılabilir modüller (capture/OCR/envanter tespiti vb.)
- `assets/` -> Görseller / template’ler (örn. `item_template.png`)
- `tools/` -> Yardımcı araçlar (ROI/koordinat seçme gibi)
- `archive/` -> Eski/rafa kalkmış denemeler (çalışmayabilir)

## Kurulum

### Seçenek A (Önerilen): Conda

```bash
conda env create -f environment.yml
conda activate yolov8cuda
```
