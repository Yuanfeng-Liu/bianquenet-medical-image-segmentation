# U-Net and BianqueNet-inspired Model Results

These results compare four models on the same processed SPIDER data. Each was
trained for 10 epochs, using a fixed random split with seed 42.

- Processed samples: 210
- Train samples: 168
- Val samples: 42
- Image size: 384 x 384
- Classes: 0 background, 1 vertebral body, 2 spinal canal/CSF, 3 disc

## Validation metrics

Mean Dice is averaged across the three foreground classes. The MFF model's
final-epoch and best-checkpoint results are both shown because its score fell
at the end of training.

| Model | Val Loss | Val Mean Dice | Class 1 Dice | Class 2 Dice | Class 3 Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Net | 0.5102 | 0.7180 | 0.7268 | 0.7831 | 0.6441 |
| BianqueNetMini | 0.2285 | 0.8914 | 0.8967 | 0.9186 | 0.8590 |
| BianqueNetMiniMFF final | 0.2398 | 0.8712 | 0.9050 | 0.9020 | 0.8068 |
| BianqueNetMiniMFF best checkpoint | 0.2532 | 0.8893 | 0.8864 | 0.9006 | 0.8807 |
| BianqueNetMiniSTSC best checkpoint | 0.2287 | 0.8940 | 0.9094 | 0.9067 | 0.8660 |

## Pixel counts on the validation set

The counts below show how much of the validation set each model assigns to each
class. They help identify over- or underprediction of a class, but do not measure
whether the predicted regions are in the correct locations; the Dice scores
above account for overlap with the reference masks.

True pixels:

| Class | Count | Percent |
| --- | ---: | ---: |
| 0 | 5,340,547 | 86.23% |
| 1 | 534,802 | 8.64% |
| 2 | 210,904 | 3.41% |
| 3 | 106,899 | 1.73% |

U-Net predicted pixels:

| Class | Count | Percent |
| --- | ---: | ---: |
| 0 | 4,952,495 | 79.97% |
| 1 | 743,623 | 12.01% |
| 2 | 296,632 | 4.79% |
| 3 | 200,402 | 3.24% |

BianqueNetMini predicted pixels:

| Class | Count | Percent |
| --- | ---: | ---: |
| 0 | 5,256,223 | 84.87% |
| 1 | 587,530 | 9.49% |
| 2 | 220,206 | 3.56% |
| 3 | 129,193 | 2.09% |

BianqueNetMiniMFF final predicted pixels:

| Class | Count | Percent |
| --- | ---: | ---: |
| 0 | 5,257,428 | 84.89% |
| 1 | 549,449 | 8.87% |
| 2 | 241,869 | 3.91% |
| 3 | 144,406 | 2.33% |

BianqueNetMiniSTSC best predicted pixels:

| Class | Count | Percent |
| --- | ---: | ---: |
| 0 | 5,292,295 | 85.45% |
| 1 | 536,910 | 8.67% |
| 2 | 235,292 | 3.80% |
| 3 | 128,655 | 2.08% |

## Prediction figures

The [published prediction grid](predictions_stsc_best_grid_0_3.png) shows the
ST-SC model's output alongside the MRI slices and reference masks. The following
filenames refer to additional figures generated locally during the experiments;
they are not included in this repository.

- U-Net: `unet/predictions_val_0_7.png`
- BianqueNetMini: `bianquenet/predictions_val_0_7.png`
- BianqueNetMiniMFF final: `bianquenet/predictions_mff_val_0_7.png`
- BianqueNetMiniMFF best: `bianquenet/predictions_mff_best_val_0_7.png`
- BianqueNetMiniSTSC best: `bianquenet/predictions_stsc_best_val_0_7.png`

## Observations

In this run, BianqueNetMini improved mean Dice by 0.1734 over the U-Net baseline.
It also predicted fewer excess foreground pixels, particularly for the disc class.

MFF's best checkpoint, saved at epoch 9, had the highest disc Dice (0.8807),
but its overall mean was slightly below the Mini model's. Its final-epoch score
was lower still, which is why the comparison includes both checkpoints.

ST-SC had the highest mean Dice at 0.8940, only 0.0026 above Mini. This small
difference needs repeated runs and a separate test set before attributing a
consistent benefit to the added attention module.

The ST-SC training script saves the selected checkpoint locally as
`bianquenet/bianquenet_stsc_minimal_best.pth`. These are single-split validation
results for a 2D adaptation, not the original paper's benchmark results.
