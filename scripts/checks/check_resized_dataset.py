from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.processed_dataset import ProcessedSpineDataset


def main() -> None:
    dataset = ProcessedSpineDataset(
        PROJECT_ROOT / "data" / "resized_384" / "images",
        PROJECT_ROOT / "data" / "resized_384" / "masks",
    )
    sample = dataset[0]

    print("dataset length:", len(dataset))
    print("sample name   :", sample["name"])
    print("image shape   :", tuple(sample["image"].shape))
    print("mask shape    :", tuple(sample["mask"].shape))
    print("image dtype   :", sample["image"].dtype)
    print("mask dtype    :", sample["mask"].dtype)
    print("mask unique values:", sample["mask"].unique())


if __name__ == "__main__":
    main()
