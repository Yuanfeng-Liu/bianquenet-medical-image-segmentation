from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
src_image_dir = PROJECT_ROOT / "data" / "processed" / "images"
src_mask_dir = PROJECT_ROOT / "data" / "processed" / "masks"
dst_image_dir = PROJECT_ROOT / "data" / "resized_384" / "images"
dst_mask_dir = PROJECT_ROOT / "data" / "resized_384" / "masks"


def clear_npy_files(directory: Path) -> None:
    for path in directory.glob("*.npy"):
        path.unlink()


def main() -> None:
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)
    clear_npy_files(dst_image_dir)
    clear_npy_files(dst_mask_dir)

    image_paths = sorted(src_image_dir.glob("*.npy"))
    for image_path in image_paths:
        mask_path = src_mask_dir / image_path.name
        image = np.load(image_path)
        mask = np.load(mask_path)
        resized_image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask, (384, 384), interpolation=cv2.INTER_NEAREST)
        np.save(dst_image_dir / image_path.name, resized_image)
        np.save(dst_mask_dir / image_path.name, resized_mask)

    print(f"saved {len(image_paths)} resized samples")


if __name__ == "__main__":
    main()


