import matplotlib
import torch
from torch.utils.data import random_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project_paths import RESIZED_IMAGE_DIR, RESIZED_MASK_DIR, UNET_DIR, UNET_MODEL_PATH
from models.unet import UNet
from utils.processed_dataset import ProcessedSpineDataset


dataset = ProcessedSpineDataset(RESIZED_IMAGE_DIR, RESIZED_MASK_DIR)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

num_samples = min(8, len(val_dataset))

model = UNet()
model.load_state_dict(torch.load(UNET_MODEL_PATH))
model.eval()

fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

with torch.no_grad():
    for sample_index in range(num_samples):
        sample = val_dataset[sample_index]
        image = sample["image"].unsqueeze(0)
        true_mask = sample["mask"]
        name = sample["name"]

        output = model(image)
        pred_mask = output.argmax(dim=1).squeeze(0)

        axes[sample_index, 0].imshow(image.squeeze(0).squeeze(0), cmap="gray")
        axes[sample_index, 0].set_title(f"{sample_index}: {name} Image")
        axes[sample_index, 0].axis("off")

        axes[sample_index, 1].imshow(true_mask, cmap="viridis")
        axes[sample_index, 1].set_title("True Mask")
        axes[sample_index, 1].axis("off")

        axes[sample_index, 2].imshow(pred_mask, cmap="viridis")
        axes[sample_index, 2].set_title("Pred Mask")
        axes[sample_index, 2].axis("off")

        print(f"processed sample {sample_index}: {name}")

plt.tight_layout()

output_path = UNET_DIR / "predictions_val_0_7.png"
plt.savefig(output_path, dpi=150)
plt.close()

print("saved figure:", output_path)
