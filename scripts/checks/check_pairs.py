from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import OVERVIEW_CSV, RADIOLOGICAL_GRADINGS_CSV, RAW_IMAGE_DIR, RAW_MASK_DIR


def main() -> None:
    image_files = sorted(RAW_IMAGE_DIR.glob("*.mha"))
    mask_files = sorted(RAW_MASK_DIR.glob("*.mha"))

    image_names = {path.name for path in image_files}
    mask_names = {path.name for path in mask_files}

    common_names = sorted(image_names & mask_names)
    only_in_images = sorted(image_names - mask_names)
    only_in_masks = sorted(mask_names - image_names)

    print("raw image dir:", RAW_IMAGE_DIR)
    print("raw mask dir :", RAW_MASK_DIR)
    print("overview csv  :", OVERVIEW_CSV, OVERVIEW_CSV.exists())
    print("gradings csv  :", RADIOLOGICAL_GRADINGS_CSV, RADIOLOGICAL_GRADINGS_CSV.exists())
    print()

    print(f"number of image files: {len(image_files)}")
    print(f"number of mask files: {len(mask_files)}")
    print(f"number of matched pairs: {len(common_names)}")
    print(f"number only in images: {len(only_in_images)}")
    print(f"number only in masks: {len(only_in_masks)}")

    print("\nfirst 10 matched names:")
    for name in common_names[:10]:
        print(name)

    print("\nfirst 10 only in images:")
    for name in only_in_images[:10]:
        print(name)

    print("\nfirst 10 only in masks:")
    for name in only_in_masks[:10]:
        print(name)


if __name__ == "__main__":
    main()
