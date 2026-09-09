import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .data_paths import PROCESSED_IMAGE_DIR, PROCESSED_MASK_DIR
except ImportError:
    # Allow `python utils/processed_dataset.py` from the repo root.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.data_paths import PROCESSED_IMAGE_DIR, PROCESSED_MASK_DIR


class ProcessedSpineDataset(Dataset):
    def __init__(self, image_dir: Path = PROCESSED_IMAGE_DIR, mask_dir: Path = PROCESSED_MASK_DIR):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_paths = sorted(self.image_dir.glob("*.npy"))

        if not self.image_paths:
            raise FileNotFoundError(f"No processed images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.mask_dir / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found for {image_path.name}")

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.int64)

        # Simple per-image standardization for stable optimization.
        image_mean = image.mean()
        image_std = image.std()
        image = (image - image_mean) / max(image_std, 1e-6)

        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask)  # (H, W)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": image_path.stem,
        }


def main() -> None:
    dataset = ProcessedSpineDataset()
    sample = dataset[0]

    print("dataset length:", len(dataset))
    print("sample name   :", sample["name"])
    print("image shape   :", tuple(sample["image"].shape))
    print("image dtype   :", sample["image"].dtype)
    print("mask shape    :", tuple(sample["mask"].shape))
    print("mask dtype    :", sample["mask"].dtype)


if __name__ == "__main__":
    main()
