from pathlib import Path
import sys

import torch
from torch.utils.data import random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.processed_dataset import ProcessedSpineDataset


def count_class_pixels(subset, num_classes=4):
    counts = torch.zeros(num_classes, dtype=torch.long)

    for sample in subset:
        mask = sample["mask"]
        counts += torch.bincount(mask.reshape(-1), minlength=num_classes)

    total_pixels = counts.sum().item()
    percentages = counts.float() / total_pixels * 100.0
    return counts, percentages


def print_stats(split_name, counts, percentages):
    print(f"\n{split_name}")
    print(f"total pixels: {counts.sum().item()}")
    for cls in range(len(counts)):
        print(
            f"class {cls}: pixels = {counts[cls].item()}, percentage = {percentages[cls].item():.4f}%"
        )


def main():
    dataset = ProcessedSpineDataset(
        PROJECT_ROOT / "data" / "resized_384" / "images",
        PROJECT_ROOT / "data" / "resized_384" / "masks",
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_counts, train_percentages = count_class_pixels(train_dataset)
    val_counts, val_percentages = count_class_pixels(val_dataset)

    print(f"train samples: {len(train_dataset)}")
    print(f"val samples  : {len(val_dataset)}")
    print_stats("Train class pixel stats", train_counts, train_percentages)
    print_stats("Val class pixel stats", val_counts, val_percentages)


if __name__ == "__main__":
    main()
