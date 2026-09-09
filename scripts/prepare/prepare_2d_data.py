from pathlib import Path
import sys

import SimpleITK as sitk
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import PROCESSED_IMAGE_DIR, PROCESSED_MASK_DIR, RAW_IMAGE_DIR, ensure_processed_dirs, raw_mask_path


def clear_npy_files(directory: Path) -> None:
    for path in directory.glob("*.npy"):
        path.unlink()


def remap_mask(mask_slice: np.ndarray) -> np.ndarray:
    new_mask = np.zeros_like(mask_slice, dtype=np.uint8)

    # Vertebra labels 1-99 -> class 1
    new_mask[(mask_slice >= 1) & (mask_slice < 100)] = 1

    # Spinal canal label 100 -> class 2
    new_mask[mask_slice == 100] = 2

    # Disc labels 200-299 -> class 3
    new_mask[(mask_slice >= 200) & (mask_slice < 300)] = 3

    return new_mask


def iter_t2_image_paths():
    for image_path in sorted(RAW_IMAGE_DIR.glob("*.mha")):
        stem = image_path.stem.lower()
        if stem.endswith("_t2"):
            yield image_path


def extract_middle_slice(volume: np.ndarray) -> np.ndarray:
    if volume.ndim == 2:
        return volume

    if volume.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {volume.shape}")

    slice_axis = int(np.argmin(volume.shape))
    mid_idx = volume.shape[slice_axis] // 2
    return np.take(volume, mid_idx, axis=slice_axis)


def main() -> None:
    ensure_processed_dirs()
    clear_npy_files(PROCESSED_IMAGE_DIR)
    clear_npy_files(PROCESSED_MASK_DIR)
    count = 0

    for image_path in iter_t2_image_paths():
        mask_path = raw_mask_path(image_path.name)
        if not mask_path.exists():
            continue

        image_itk = sitk.ReadImage(str(image_path))
        mask_itk = sitk.ReadImage(str(mask_path))

        image_arr = sitk.GetArrayFromImage(image_itk)  # (z, y, x)
        mask_arr = sitk.GetArrayFromImage(mask_itk)

        if image_arr.shape != mask_arr.shape:
            raise ValueError(
                f"Image and mask shape mismatch for {image_path.name}: "
                f"{image_arr.shape} vs {mask_arr.shape}"
            )

        image_slice = extract_middle_slice(image_arr)
        mask_slice = extract_middle_slice(mask_arr)

        np.save(PROCESSED_IMAGE_DIR / f"{image_path.stem}.npy", image_slice.astype(np.float32))
        np.save(PROCESSED_MASK_DIR / f"{image_path.stem}.npy", remap_mask(mask_slice).astype(np.uint8))
        count += 1

    print(f"saved {count} 2D T2 samples")
    print("processed images folder:", PROCESSED_IMAGE_DIR)
    print("processed masks folder :", PROCESSED_MASK_DIR)


if __name__ == "__main__":
    main()
