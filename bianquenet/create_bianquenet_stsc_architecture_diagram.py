import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from project_paths import BIANQUENET_DIR


OUTPUT_PATH = BIANQUENET_DIR / "bianquenet_stsc_architecture.png"


def add_box(ax, xy, width, height, text, facecolor):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.5,
        edgecolor="#222222",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
    )


def add_arrow(ax, start, end, color="#333333", connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 15.4)
ax.set_ylim(0, 5)
ax.axis("off")

boxes = [
    ((0.3, 2.0), 1.25, 0.75, "Input\n1x384x384", "#e8f1ff"),
    ((1.95, 2.0), 1.35, 0.75, "Enc1\n32x384x384", "#e8fff1"),
    ((3.7, 2.0), 1.35, 0.75, "Enc2\n64x192x192", "#e8fff1"),
    ((5.45, 2.0), 1.35, 0.75, "Enc3\n128x96x96", "#e8fff1"),
    ((7.2, 2.0), 1.35, 0.75, "DFE\nPSP + ASPP", "#fff5d8"),
    ((8.95, 2.0), 1.35, 0.75, "ST-SC\nshifted window", "#f2e8ff"),
    ((10.7, 2.0), 1.35, 0.75, "MFF\nmulti-scale fuse", "#ffe8ef"),
    ((12.45, 2.0), 1.1, 0.75, "Head\n1x1 conv", "#eeeeee"),
    ((13.9, 2.0), 1.25, 0.75, "Output\n4x384x384", "#e8f1ff"),
]

for xy, width, height, text, color in boxes:
    add_box(ax, xy, width, height, text, color)

main_y = 2.375
arrow_pairs = [
    ((1.55, main_y), (1.95, main_y)),
    ((3.3, main_y), (3.7, main_y)),
    ((5.05, main_y), (5.45, main_y)),
    ((6.8, main_y), (7.2, main_y)),
    ((8.55, main_y), (8.95, main_y)),
    ((10.3, main_y), (10.7, main_y)),
    ((12.05, main_y), (12.45, main_y)),
    ((13.55, main_y), (13.9, main_y)),
]

for start, end in arrow_pairs:
    add_arrow(ax, start, end)

add_arrow(ax, (2.62, 2.0), (11.05, 2.0), color="#2a7f62", connectionstyle="arc3,rad=-0.35")
add_arrow(ax, (4.38, 2.0), (11.25, 2.0), color="#2a7f62", connectionstyle="arc3,rad=-0.25")

ax.text(6.9, 0.95, "skip feature from Enc1", ha="center", va="center", fontsize=9, color="#2a7f62")
ax.text(7.9, 1.25, "skip feature from Enc2", ha="center", va="center", fontsize=9, color="#2a7f62")

ax.text(
    7.0,
    4.25,
    "BianqueNetMiniSTSC architecture used in this project",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
)
ax.text(
    7.0,
    3.75,
    "Encoder extracts multi-scale features; DFE adds PSP/ASPP context; ST-SC models local window context; MFF fuses shallow and deep features.",
    ha="center",
    va="center",
    fontsize=10,
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=180)
plt.close()

print("saved architecture diagram:", OUTPUT_PATH)
