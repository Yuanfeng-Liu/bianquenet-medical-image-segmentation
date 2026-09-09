import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project_paths import (
    BIANQUENET_STSC_DICE_CURVE_PATH,
    BIANQUENET_STSC_HISTORY_PATH,
    BIANQUENET_STSC_LOSS_CURVE_PATH,
)


def read_history():
    with open(BIANQUENET_STSC_HISTORY_PATH, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "val_mean_dice": float(row["val_mean_dice"]),
                "class_1_dice": float(row["class_1_dice"]),
                "class_2_dice": float(row["class_2_dice"]),
                "class_3_dice": float(row["class_3_dice"]),
            }
            for row in reader
        ]


history = read_history()
epochs = [row["epoch"] for row in history]

plt.figure(figsize=(8, 5))
plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train loss")
plt.plot(epochs, [row["val_loss"] for row in history], marker="o", label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BianqueNetMiniSTSC Loss Curve")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(BIANQUENET_STSC_LOSS_CURVE_PATH, dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, [row["val_mean_dice"] for row in history], marker="o", label="mean Dice")
plt.plot(epochs, [row["class_1_dice"] for row in history], marker="o", label="class 1 Dice")
plt.plot(epochs, [row["class_2_dice"] for row in history], marker="o", label="class 2 Dice")
plt.plot(epochs, [row["class_3_dice"] for row in history], marker="o", label="class 3 Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title("BianqueNetMiniSTSC Dice Curve")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(BIANQUENET_STSC_DICE_CURVE_PATH, dpi=150)
plt.close()

print("saved loss curve:", BIANQUENET_STSC_LOSS_CURVE_PATH)
print("saved dice curve:", BIANQUENET_STSC_DICE_CURVE_PATH)
