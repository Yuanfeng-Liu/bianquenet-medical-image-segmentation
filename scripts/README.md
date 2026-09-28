# Data preparation

These scripts turn the SPIDER MRI volumes and reference masks into the 2D
samples used for training. See the [project README](../README.md#data-preparation)
for the download link and input folder structure.

Run these commands from the repository root:

```powershell
python scripts/prepare/prepare_2d_data.py
python scripts/prepare/resize_processed_data.py
python scripts/checks/check_resized_dataset.py
```

1. `prepare_2d_data.py` reads images whose filenames end in `_t2`, extracts a
   middle slice from each matching image-mask pair, and maps the instance labels
   to background, vertebral body, spinal canal, and disc. It writes NumPy arrays
   to `data/processed/`.
2. `resize_processed_data.py` resizes those arrays to `384 x 384` and writes them
   to `data/resized_384/`. Images use linear interpolation; masks use
   nearest-neighbor interpolation to preserve class labels.
3. `check_resized_dataset.py` checks the prepared samples before training.

The preparation scripts clear existing `.npy` files from their output folders
before rebuilding the dataset. The original `.mha` files in `data/raw/` are left
unchanged.
