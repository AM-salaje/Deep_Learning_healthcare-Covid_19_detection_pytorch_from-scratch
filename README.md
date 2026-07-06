# COVID-19 Detection from Chest X-Rays — Deep Learning (PyTorch)

Deep learning models for detecting COVID-19 from chest X-ray images, implemented
in PyTorch. Covers both **binary classification** (COVID vs. Normal) and
**multi-class classification** (COVID vs. Normal vs. Pneumonia), using CNNs built
from scratch as well as transfer learning.

## Approaches

| Model | Task | Notebook |
|-------|------|----------|
| VGG-16 (from scratch) | Binary | `Vgg_16_from_scratch_binaryclass.ipynb` |
| VGG-16 (transfer learning) | Binary | `VGG_16_transfer_learning_Binary_classf.ipynb` |
| VGG-16 (transfer learning) | Multi-class | `multiclass__transfer_learning_Vgg16.ipynb` |
| ResNet (from scratch) | Binary | `binary_classif_RESnet_from_scratch.ipynb` |
| ResNet (transfer learning) | Binary | `binary_classification_transfer_learning_RESnet.ipynb` |
| ResNet (transfer learning) | Multi-class | `multiclass_classif_transfer_learning_RESnet.ipynb` |

## Techniques used

- Convolutional neural networks: **VGG-16** and **ResNet**
- **Transfer learning** from ImageNet-pretrained backbones
- Optimization & regularization: **Data Augmentation**, **Adam** optimizer,
  **Batch Normalization**, **Dropout**, **Early Stopping**, and gradient checking

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

The chest X-ray dataset is **not included** in this repo (image data is large and
should be downloaded separately). Organize the images into class subfolders, e.g.:

```
data/
├── train/
│   ├── COVID/
│   ├── NORMAL/
│   └── PNEUMONIA/   # multi-class only
└── test/
    ├── COVID/
    ├── NORMAL/
    └── PNEUMONIA/
```

Update the data path at the top of each notebook to point to your `data/` folder.

## Usage

Open any notebook in Jupyter and run the cells:

```bash
jupyter notebook
```

Each notebook is self-contained — it loads the data, defines the model, trains it,
and reports evaluation metrics.

## Requirements

Python 3.8+ with PyTorch. See `requirements.txt` for the full list. A CUDA-capable
GPU is recommended for training.
