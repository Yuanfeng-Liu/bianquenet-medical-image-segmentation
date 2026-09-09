from pathlib import Path
import sys


UNET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UNET_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESIZED_IMAGE_DIR = PROJECT_ROOT / "data" / "resized_384" / "images"
RESIZED_MASK_DIR = PROJECT_ROOT / "data" / "resized_384" / "masks"
UNET_MODEL_PATH = UNET_DIR / "unet_minimal.pth"
