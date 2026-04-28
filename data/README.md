# 📂 Dataset

The dataset is **not stored in this repository** due to its large size.

## Download Links


| **Total Dataset** | Raw horn audio recordings for all car models | [Google Drive](https://drive.google.com/drive/folders/1wIjJkNlkF47JE0VTlO8Cvfe8SVdVRLC1?usp=drive_link) |
| **Split Dataset** | Preprocessed Mel Spectrogram & MFCC images, split into train/val/test | [Google Drive](https://drive.google.com/drive/folders/11Yii8IWjF6t0zIyM5xfv00euW_XGN8i_?usp=drive_link) |

## Expected Folder Structure

After downloading the split dataset, place it in your Google Drive as:

```
MyDrive/
└── dataset/
    ├── train/
    │   ├── CarModel_A/
    │   └── CarModel_B/
    ├── val/
    │   ├── CarModel_A/
    │   └── CarModel_B/
    └── test/
        ├── CarModel_A/
        └── CarModel_B/
```

Each class folder contains `.png` files of Mel Spectrograms and MFCC images generated from audio chunks.
