# X-ray Classification using Convolutional Neural Networks (CNN)

> Binary classification of chest X-rays (pleural effusion vs. normal) using two custom CNN architectures trained from scratch with TensorFlow/Keras.

---

## Overview

Pleural effusion, the abnormal accumulation of fluid in the pleural space, can indicate serious conditions such as heart failure or tumours. Chest X-rays are the standard diagnostic method, but their interpretation is subjective and observer-dependent.

This project develops and compares two CNN architectures to automatically classify chest X-rays into two categories: **effusion** and **normal**, using a balanced dataset of 700 images from the NIH Chest X-ray Dataset.

---

## Dataset

| Split | Effusion | Normal | Total |
|-------|----------|--------|-------|
| Train | 283      | 277    | 560   |
| Test  | 67       | 73     | 140   |

- Source: [NIH Chest X-ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
- Original resolution: 512×512×3 px
- Preprocessing: resized to **64×64**, converted to **grayscale**, normalised to **[0, 1]**

---

## Models

### Model A — Lightweight CNN (Sequential API)
```
[Conv2D(32) → BatchNorm → ReLU → MaxPool] ×3
→ GlobalAveragePooling2D → Dense(64, ReLU) → Dropout(20%) → Sigmoid
```
~102K parameters · Learning rate: 1e-3

### Model B — Deep CNN with Data Augmentation (Functional API)
```
RandomFlip + RandomRotation
→ [Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool]
→ [Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool]
→ GlobalAveragePooling2D → Dense(64, ReLU) → Dropout(20%) → Sigmoid
```
~70K parameters · Learning rate: 1e-4

Both models use **Adam** optimiser, **binary cross-entropy** loss, and are trained for up to 50 epochs with:
- `EarlyStopping` (patience=8, monitors val_AUC)
- `ReduceLROnPlateau` (patience=4, factor=0.5)

Each model is trained **5 times** with different random seeds; the best run by AUC is kept.

---

## Results

| Model              | Accuracy | AUC    | Sensitivity | Specificity | Precision | FP | FN |
|--------------------|----------|--------|-------------|-------------|-----------|----|----|
| Model A — Lightweight | 0.8500 | 0.8996 | 0.8493    | 0.8507      | 0.8611    | 10 | 11 |
| Model B — Deep     | 0.8000   | 0.8536 | 0.8356      | 0.7612      | 0.7922    | 16 | 12 |

<img width="1589" height="396" alt="image" src="https://github.com/user-attachments/assets/7a529d71-b233-4559-a741-f98d877ee8f9" />

**Model A outperforms Model B across all metrics.** Despite being architecturally simpler, it achieves AUC ~0.90 and fewer false negatives, the most clinically critical error type. The deeper Model B struggles with the limited dataset size, showing signs of slow convergence and overfitting.

---

## Project Structure

```
.
├── chest-xray-cnn-classifier.ipynb    # Main notebook (data loading, training, evaluation)
├── rxtorax/                           # Dataset directory
│   ├── effusion/
│   └── normal/
├── model_A.keras            # Saved Model A
├── model_B.keras            # Saved Model B

```

---

## Requirements

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

Or all at once:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn pathlib
```

Tested with Python 3.10+ and TensorFlow 2.x.

> `os`, `random`, and `pathlib` are part of the Python standard library — no installation needed.

---

## Usage

1. Clone the repository and place the `rxtorax/` dataset folder in the root directory.
2. Open `chest-xray-cnn-classifier.ipynb` and run all cells sequentially.
3. Trained models will be saved as `model_A.keras` and `model_B.keras`.

---

## Key Takeaways

- A simpler architecture can outperform a more complex one when data is scarce.
- With only 700 images, double convolutional blocks + data augmentation (Model B) slow down initial learning rather than helping generalisation.
- Both models exceed **AUC 0.85** trained from scratch, with no transfer learning.
- Proposed improvements: higher input resolution, transfer learning (e.g. EfficientNet), threshold tuning to prioritise sensitivity.

### Incorrect predictions

| Model A | Model B |
|---------|---------|
| <img width="1197" height="494" alt="image" src="https://github.com/user-attachments/assets/101a22bd-a0a4-4baf-a100-61566a1d6bfe" /> | <img width="1197" height="494" alt="image" src="https://github.com/user-attachments/assets/e0f23828-61a7-4d73-a2cb-f170a2c0c84f" />|

