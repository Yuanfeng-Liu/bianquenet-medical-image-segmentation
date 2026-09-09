# U-Net vs BianqueNet Comparison

Data version: cleaned 2D T2 dataset

- Processed samples: 210
- Train samples: 168
- Val samples: 42
- Image size: 384 x 384
- Classes: 0 background, 1 vertebral body, 2 spinal canal/CSF, 3 disc

## Final Validation Metrics

| Model | Val Loss | Val Mean Dice | Class 1 Dice | Class 2 Dice | Class 3 Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Net | 0.5102 | 0.7180 | 0.7268 | 0.7831 | 0.6441 |
| BianqueNetMini | 0.2285 | 0.8914 | 0.8967 | 0.9186 | 0.8590 |
| BianqueNetMiniMFF final | 0.2398 | 0.8712 | 0.9050 | 0.9020 | 0.8068 |
| BianqueNetMiniMFF best checkpoint | 0.2532 | 0.8893 | 0.8864 | 0.9006 | 0.8807 |
| BianqueNetMiniSTSC best checkpoint | 0.2287 | 0.8940 | 0.9094 | 0.9067 | 0.8660 |

## Pixel Distribution On Validation Set

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

## Visual Outputs

- U-Net: `unet/predictions_val_0_7.png`
- BianqueNetMini: `bianquenet/predictions_val_0_7.png`
- BianqueNetMiniMFF final: `bianquenet/predictions_mff_val_0_7.png`
- BianqueNetMiniMFF best: `bianquenet/predictions_mff_best_val_0_7.png`
- BianqueNetMiniSTSC best: `bianquenet/predictions_stsc_best_val_0_7.png`

## Takeaway

BianqueNetMini is clearly stronger than U-Net on the cleaned dataset. It improves mean Dice by 0.1734 over U-Net and gives a predicted pixel distribution closer to the validation ground truth, especially for the small foreground classes.

The first MFF version does not beat BianqueNetMini in this 10-epoch run. The best checkpoint is saved from epoch 9. It gets close to BianqueNetMini and improves class 3 Dice, but its mean Dice is still slightly lower.

BianqueNetMiniSTSC is the current best mean-Dice model in this set of experiments. It reaches 0.8940 mean Dice, slightly above BianqueNetMini's 0.8914, and keeps the predicted validation pixel distribution close to the true distribution.

Current best checkpoint:

`bianquenet/bianquenet_stsc_minimal_best.pth`
