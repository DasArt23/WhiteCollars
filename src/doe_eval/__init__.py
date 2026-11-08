from .config import DISTANCES, PIXEL, WAVELENGTH, N

__all__ = [
    "DISTANCES",
    "PIXEL",
    "WAVELENGTH",
    "N",
    "dequantize_phase_u8",
    "fresnel_reconstruct",
    "normalize_reconstructions",
    "pearson",
    "pearson_torch",
    "quantize_phase_u8",
    "ssim_global",
    "ssim_global_torch",
    "ssim_skimage",
    "ste_quantize",
]

def __getattr__(name: str):
    """Импорт config/reference_numpy не должен загружать torch или physics.

    Сохраняет публичные from doe_eval import ... для существующих клиентов.
    """
    from importlib import import_module

    if name in {"pearson", "pearson_torch", "ssim_global", "ssim_global_torch", "ssim_skimage"}:
        module = import_module(".metrics", __name__)
    elif name in {"dequantize_phase_u8", "fresnel_reconstruct", "normalize_reconstructions", "quantize_phase_u8", "ste_quantize"}:
        module = import_module(".physics", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value
