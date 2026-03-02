import inspect
import dxcam

import core.capture_service as cs


def ensure_dxcam(preferred_output_color: str = "GRAY", test_region=(0, 0, 32, 32)) -> tuple[bool, str]:
    """
    capture_service.py değişmeden dxcam'i başlatıp cs._cam içine koyar.
    Farklı device_idx/output_idx kombinasyonlarını dener.

    return: (ok, msg)
    """
    # Zaten hazırsa dokunma
    if getattr(cs, "_cam", None) is not None:
        return True, "dxcam zaten hazır (cs._cam dolu)."

    # dxcam.create parametreleri sürüme göre değişebiliyor; signature bakıyoruz
    sig = inspect.signature(dxcam.create)
    supports_device_output = ("device_idx" in sig.parameters) and ("output_idx" in sig.parameters)

    last_err: Exception | None = None

    # 1) Default dene
    try:
        cam = dxcam.create(output_color=preferred_output_color)
        _ = cam.grab(region=test_region)
        cs._cam = cam
        return True, "dxcam default ile açıldı."
    except Exception as e:
        last_err = e

    # 2) device/output kombinasyonlarını tara (hybrid GPU’da çoğu zaman çözüm bu)
    if supports_device_output:
        for device_idx in range(8):
            for output_idx in range(8):
                try:
                    cam = dxcam.create(
                        device_idx=device_idx,
                        output_idx=output_idx,
                        output_color=preferred_output_color,
                    )
                    _ = cam.grab(region=test_region)
                    cs._cam = cam
                    return True, f"dxcam açıldı (device_idx={device_idx}, output_idx={output_idx})."
                except Exception as e:
                    last_err = e

    return False, f"dxcam açılamadı. Son hata: {type(last_err).__name__}: {last_err}"