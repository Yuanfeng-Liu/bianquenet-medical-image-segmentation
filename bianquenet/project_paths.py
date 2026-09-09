from pathlib import Path
import sys


BIANQUENET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BIANQUENET_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESIZED_IMAGE_DIR = PROJECT_ROOT / "data" / "resized_384" / "images"
RESIZED_MASK_DIR = PROJECT_ROOT / "data" / "resized_384" / "masks"
BIANQUENET_MODEL_PATH = BIANQUENET_DIR / "bianquenet_minimal.pth"
BIANQUENET_BEST_MODEL_PATH = BIANQUENET_DIR / "bianquenet_minimal_best.pth"
BIANQUENET_MFF_MODEL_PATH = BIANQUENET_DIR / "bianquenet_mff_minimal.pth"
BIANQUENET_MFF_BEST_MODEL_PATH = BIANQUENET_DIR / "bianquenet_mff_minimal_best.pth"
BIANQUENET_STSC_MODEL_PATH = BIANQUENET_DIR / "bianquenet_stsc_minimal.pth"
BIANQUENET_STSC_BEST_MODEL_PATH = BIANQUENET_DIR / "bianquenet_stsc_minimal_best.pth"
BIANQUENET_STSC_HISTORY_PATH = BIANQUENET_DIR / "bianquenet_stsc_training_history.csv"
BIANQUENET_STSC_LOSS_CURVE_PATH = BIANQUENET_DIR / "bianquenet_stsc_loss_curve.png"
BIANQUENET_STSC_DICE_CURVE_PATH = BIANQUENET_DIR / "bianquenet_stsc_dice_curve.png"
