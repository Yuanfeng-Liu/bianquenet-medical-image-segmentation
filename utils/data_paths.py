from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
RAW_IMAGE_DIR = RAW_DIR / "images"
RAW_MASK_DIR = RAW_DIR / "masks"
OVERVIEW_CSV = RAW_DIR / "overview.csv"
RADIOLOGICAL_GRADINGS_CSV = RAW_DIR / "radiological_gradings.csv"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_IMAGE_DIR = PROCESSED_DIR / "images"
PROCESSED_MASK_DIR = PROCESSED_DIR / "masks"


def raw_image_path(name: str) -> Path:
    return RAW_IMAGE_DIR / name


def raw_mask_path(name: str) -> Path:
    return RAW_MASK_DIR / name


def ensure_processed_dirs() -> None:
    PROCESSED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_MASK_DIR.mkdir(parents=True, exist_ok=True)
