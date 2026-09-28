# Lumbar MRI Segmentation with BianqueNet-inspired Models

**Yuanfeng Liu**

I built a PyTorch pipeline to segment vertebral bodies, the spinal canal, and
intervertebral discs in sagittal T2-weighted MRI. The project compares a standard
2D U-Net with three lightweight models adapted from ideas in BianqueNet.

On a fixed validation split of 42 samples, the final model reached a mean
foreground Dice of **0.8940**, compared with **0.7180** for the U-Net baseline.
The experiments use one middle slice from each of 210 SPIDER T2 series.

## Implementation

My work covered data preparation, model implementation, training, and evaluation:

- Read the SPIDER `.mha` images and masks with SimpleITK, extracted middle
  slices, and remapped the instance labels into four semantic classes.
- Built the U-Net baseline, then added multi-scale context, feature fusion,
  and shifted-window attention in successive model variants.
- Trained with weighted cross-entropy and Dice loss, compared foreground-class
  scores, and inspected predicted masks alongside the reference labels.

The model definitions are in [`models/bianquenet.py`](models/bianquenet.py).

| Stage | Model | Main idea |
| --- | --- | --- |
| 1 | `UNet` | Standard encoder-decoder baseline with skip connections |
| 2 | `BianqueNetMini` | Lightweight encoder followed by DFE |
| 3 | `BianqueNetMiniMFF` | DFE plus multi-scale feature fusion |
| 4 | `BianqueNetMiniSTSC` | DFE, simplified shifted-window context, and MFF |

DFE combines pyramid pooling and atrous spatial pyramid pooling to collect
context at several scales. MFF fuses shallow, middle, and deep features. The
ST-SC block adds simplified shifted-window attention. These are my lightweight
adaptations; the repository is independent of the paper's official implementation.

![BianqueNetMiniSTSC architecture](bianquenet/bianquenet_stsc_architecture.png)

## Validation results

Each model was trained for 10 epochs on the same split: **168 training samples
and 42 validation samples**, with seed `42` and images resized to `384 x 384`.
Mean Dice excludes the background class.

| Model / checkpoint | Val loss | Mean Dice | Vertebral body | Spinal canal / CSF | Disc |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Net | 0.5102 | 0.7180 | 0.7268 | 0.7831 | 0.6441 |
| BianqueNetMini | 0.2285 | 0.8914 | 0.8967 | 0.9186 | 0.8590 |
| BianqueNetMiniMFF (best) | 0.2532 | 0.8893 | 0.8864 | 0.9006 | 0.8807 |
| BianqueNetMiniSTSC (best) | 0.2287 | **0.8940** | 0.9094 | 0.9067 | 0.8660 |

ST-SC had the highest mean Dice in this run, although its advantage over the
simpler Mini model was small: 0.0026. MFF had the highest disc Dice but a lower
overall mean. The [detailed comparison](bianquenet/compare_unet_bianquenet_results.md)
also records the predicted class distributions.

These results come from one validation split. Repeated seeds and a separate
test set are needed to assess whether the differences hold. The data preparation
and model variants differ from the original paper, so these scores are not a
direct reproduction of its test results or results on the hidden SPIDER test set.

![ST-SC validation Dice curves](bianquenet/bianquenet_stsc_dice_curve.png)

### Example predictions

Each row shows an MRI slice, its reference mask, and the ST-SC model's prediction
from the validation split.

![MRI, reference masks, and predicted masks](bianquenet/predictions_stsc_best_grid_0_3.png)

## Installation

Python 3.9 was used for the recorded experiment. From the repository root:

```powershell
conda create -n bianquenet python=3.9 -y
conda activate bianquenet
python -m pip install -r requirements.txt
```

The model runs on CPU; a CUDA-enabled PyTorch installation is recommended for
training. If needed, install the correct PyTorch build for your platform from
the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Data preparation

The data are not included in this repository. Download the SPIDER dataset from
[Zenodo](https://doi.org/10.5281/zenodo.10159290), then arrange the extracted
files as follows:

```text
data/
└── raw/
    ├── images/                    # MRI .mha files
    ├── masks/                     # reference-mask .mha files
    ├── overview.csv
    └── radiological_gradings.csv
```

Run the preparation and validation scripts from the repository root:

```powershell
python scripts/prepare/prepare_2d_data.py
python scripts/prepare/resize_processed_data.py
python scripts/checks/check_resized_dataset.py
```

`prepare_2d_data.py` keeps standard T2 series, extracts one middle sagittal
slice per series, and maps the original instance labels to the four semantic
classes: background (`0`), vertebral body (`1`), spinal canal / CSF (`2`), and
disc (`3`). Image resizing uses linear interpolation; mask resizing uses
nearest-neighbor interpolation.

## Running the model

First run a data-free shape check:

```powershell
python bianquenet/check_bianquenet_stsc_forward.py
```

After preparing the data, train and evaluate the ST-SC model:

```powershell
python bianquenet/train_bianquenet_stsc_minimal.py
python bianquenet/evaluate_bianquenet_stsc.py
python bianquenet/save_bianquenet_stsc_predictions.py
```

Training creates the checkpoint used by the evaluation and prediction scripts.
Checkpoints and generated outputs are kept locally; the repository includes
selected figures from the recorded experiment.

## Repository structure

```text
models/             Model definitions for U-Net and BianqueNetMini variants
scripts/prepare/    Raw-data conversion and resize scripts
scripts/checks/     Data-integrity and tensor-contract checks
unet/               U-Net baseline experiments
bianquenet/         Incremental module checks, training, evaluation, and figures
utils/              Shared paths and PyTorch Dataset implementation
```

## Scope

The experiments cover 2D segmentation, with a short training schedule and no
systematic hyperparameter search. They do not evaluate full-volume segmentation
or disc-degeneration grading. This is a research project and has not been
validated for clinical use.

## References and data attribution

- BianqueNet inspiration: Zheng et al., *Deep learning-based high-accuracy
  quantitation for lumbar intervertebral disc degeneration from MRI*, Nature
  Communications (2022). [DOI](https://doi.org/10.1038/s41467-022-28387-5)
- Dataset: van der Graaf et al., *Lumbar spine segmentation in MR images: a
  dataset and a public benchmark*, Scientific Data (2024).
  [Paper](https://doi.org/10.1038/s41597-024-03090-w) ·
  [Dataset](https://doi.org/10.5281/zenodo.10159290) ·
  [SPIDER challenge](https://spider.grand-challenge.org/)

The SPIDER dataset is available under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
The qualitative figure above is derived from SPIDER data; modifications include
middle-slice extraction, semantic label remapping, resizing, and model
prediction by Yuanfeng Liu.

## Code license

The original code in this repository is released under the
[MIT License](LICENSE). Dataset rights remain with the SPIDER dataset authors
under CC BY 4.0.
