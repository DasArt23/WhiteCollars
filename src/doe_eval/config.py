N: int = 128
WAVELENGTH: float = 532e-9
PIXEL: float = 10e-6
DISTANCES: tuple[float, ...] = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)
NUM_PLANES: int = len(DISTANCES)

# Формат папок датасета (согласуется в одном месте): data/<split>/<id>/1.png ... 4.png
IMAGE_FILENAMES: tuple[str, ...] = ("1.png", "2.png", "3.png", "4.png")

DEFAULT_NORM: str = "max"           # "max" | "minmax"
DEFAULT_COMPARE: str = "amplitude"  # "amplitude" | "intensity"

SSIM_K1: float = 0.01
SSIM_K2: float = 0.03
EPS: float = 1e-8