import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from project_paths import (
    BIANQUENET_MFF_BEST_MODEL_PATH,
    BIANQUENET_MFF_MODEL_PATH,
    RESIZED_IMAGE_DIR,
    RESIZED_MASK_DIR,
)
from models.bianquenet import BianqueNetMiniMFF
from utils.processed_dataset import ProcessedSpineDataset


torch.manual_seed(42)


def dice_loss(logits, targets, num_classes=4, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

    intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def per_class_dice_scores(logits, targets, num_classes=4, eps=1e-6):
    preds = logits.argmax(dim=1)
    dice_scores = []

    for cls in range(1, num_classes):
        pred_mask = (preds == cls).float()
        target_mask = (targets == cls).float()

        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()

        dice = (2 * intersection + eps) / (union + eps)
        dice_scores.append(dice)

    return torch.stack(dice_scores)


dataset = ProcessedSpineDataset(RESIZED_IMAGE_DIR, RESIZED_MASK_DIR)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print("train samples:", len(train_dataset))
print("val samples  :", len(val_dataset))

model = BianqueNetMiniMFF()
class_weights = torch.tensor([0.2, 1.0, 3.0, 4.0])
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
best_val_dice = -1.0
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        images = batch["image"]
        masks = batch["mask"]

        optimizer.zero_grad()
        outputs = model(images)
        ce_loss = criterion(outputs, masks)
        loss = ce_loss + dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_class_dice = torch.zeros(3)
    val_true_counts = torch.zeros(4, dtype=torch.long)
    val_pred_counts = torch.zeros(4, dtype=torch.long)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"]
            masks = batch["mask"]

            outputs = model(images)
            ce_loss = criterion(outputs, masks)
            loss = ce_loss + dice_loss(outputs, masks)

            val_loss += loss.item()
            class_dice = per_class_dice_scores(outputs, masks)
            val_dice += class_dice.mean().item()
            val_class_dice += class_dice.cpu()

            preds = outputs.argmax(dim=1)
            val_true_counts += torch.bincount(masks.reshape(-1), minlength=4).cpu()
            val_pred_counts += torch.bincount(preds.reshape(-1), minlength=4).cpu()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    avg_val_class_dice = val_class_dice / len(val_loader)
    val_true_percent = val_true_counts.float() / val_true_counts.sum() * 100.0
    val_pred_percent = val_pred_counts.float() / val_pred_counts.sum() * 100.0

    print(
        f"epoch {epoch + 1}, "
        f"train loss: {avg_train_loss:.4f}, "
        f"val loss: {avg_val_loss:.4f}, "
        f"val mean Dice: {avg_val_dice:.4f}, "
        f"class 1 Dice: {avg_val_class_dice[0].item():.4f}, "
        f"class 2 Dice: {avg_val_class_dice[1].item():.4f}, "
        f"class 3 Dice: {avg_val_class_dice[2].item():.4f}"
    )
    print(
        "val true pixels : "
        f"class 0 = {val_true_counts[0].item()} ({val_true_percent[0].item():.2f}%), "
        f"class 1 = {val_true_counts[1].item()} ({val_true_percent[1].item():.2f}%), "
        f"class 2 = {val_true_counts[2].item()} ({val_true_percent[2].item():.2f}%), "
        f"class 3 = {val_true_counts[3].item()} ({val_true_percent[3].item():.2f}%)"
    )
    print(
        "val pred pixels : "
        f"class 0 = {val_pred_counts[0].item()} ({val_pred_percent[0].item():.2f}%), "
        f"class 1 = {val_pred_counts[1].item()} ({val_pred_percent[1].item():.2f}%), "
        f"class 2 = {val_pred_counts[2].item()} ({val_pred_percent[2].item():.2f}%), "
        f"class 3 = {val_pred_counts[3].item()} ({val_pred_percent[3].item():.2f}%)"
    )

    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        best_epoch = epoch + 1
        torch.save(model.state_dict(), BIANQUENET_MFF_BEST_MODEL_PATH)
        print(f"saved best model to {BIANQUENET_MFF_BEST_MODEL_PATH}")

torch.save(model.state_dict(), BIANQUENET_MFF_MODEL_PATH)
print(f"saved model to {BIANQUENET_MFF_MODEL_PATH}")
print(f"best val mean Dice: {best_val_dice:.4f} at epoch {best_epoch}")
