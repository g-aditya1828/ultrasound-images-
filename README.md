# 🦾 MuscleVision — Automated Muscle Architecture Analysis from Ultrasound Images

> *Automatically measuring what the human eye struggles to measure consistently.*

---

## 📋 What Is This Project?

Skeletal muscles are made up of bundles of fibres called **fascicles**, which run diagonally between two sheets of connective tissue called **aponeuroses**. How these structures are arranged — their angle, length, and how thick the whole muscle is — tells us a great deal about muscle health, athletic performance, injury risk, and recovery.

Traditionally, measuring these architectural parameters from ultrasound scans required a trained clinician to manually annotate every image — a process that is slow, tedious, and inconsistent between observers.

**MuscleVision** is a deep learning pipeline that automates this entire process. Given a raw grayscale ultrasound image of a skeletal muscle, it segments the internal structures using two neural networks and then uses geometric calculations to extract three key clinical measurements automatically.

| Parameter | Symbol | What it measures |
|-----------|--------|-----------------|
| **Pennation Angle** | PA (degrees) | The angle between muscle fascicles and the deep aponeurosis |
| **Fascicle Length** | FL (mm) | The length of a muscle fascicle from one aponeurosis to the other |
| **Muscle Thickness** | MT (mm) | The perpendicular distance between the superficial and deep aponeuroses |

The core idea: *if you can accurately segment the aponeuroses and fascicles in an ultrasound image, you can derive all three architectural parameters with simple geometry.*

---

## 🧠 How It Works — Technical Overview

The pipeline has three major stages:

```
Ultrasound Image
      │
      ▼
┌─────────────────────────────┐
│  Stage 1: Fascicle Model    │  ResNet34 U-Net → binary fascicle mask
│  (Thin-line segmentation)   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Stage 2: Aponeurosis Model │  ResNet34 U-Net → binary aponeurosis mask
│  (Sheet segmentation)       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Stage 3: Geometry Engine   │  Line fitting + trigonometry → PA, FL, MT
└─────────────────────────────┘
```

---

### Model Architecture — ResNet34 Attention U-Net

Both segmentation models share the same architecture:

- **Encoder**: ImageNet-pretrained ResNet34 backbone adapted for grayscale input (RGB weights collapsed to 1 channel by summation)
- **Decoder**: 4 upsampling blocks with **Attention Gates** on each skip connection — filters out irrelevant background features and focuses on the thin structures of interest
- **Full-resolution stream**: A lightweight parallel branch at native image resolution that preserves fine detail (1–3 px fascicle lines) which would otherwise be lost deep in the feature maps
- **Head**: Final `1×1` convolution producing a single-channel probability map

### Why Two Separate Models?

Fascicles (thin diagonal lines, ~1–3 px wide) and aponeuroses (thick horizontal sheets, ~5–15 px wide) have fundamentally different visual characteristics. A single shared model is forced to compromise between both. Two specialised models, each with tuned loss functions and augmentation strategies, produce much cleaner masks.

---

### Loss Function — ComboLoss

A composite loss designed for heavily imbalanced binary segmentation of thin structures:

$$\mathcal{L} = \underbrace{\text{BCE}_{\text{weighted}}}_{\text{class balance}} + \underbrace{(1 - \text{TI})^{0.75}}_{\text{Focal-Tversky}} + \underbrace{w_{\text{cl}} \cdot \mathcal{L}_{\text{clDice}}}_{\text{fascicle only}}$$

- **Weighted BCE** — upweights the rare foreground pixels (fascicles occupy less than 1% of image area)
- **Focal-Tversky** — recall-oriented loss that penalises missed detections more than false alarms; much better than plain Dice for rare foreground
- **clDice** (fascicle model only) — a differentiable loss on the *skeleton* of the predicted and ground-truth masks; rewards connected, continuous line predictions over fragmented blobs

---

### Key Engineering Decisions for Thin Lines

The biggest challenge in this project is that fascicle lines are extremely thin. Several design decisions address this directly:

1. **Full native resolution** — the canvas follows actual image size (up to 1024 px); resizing everything to a fixed small canvas destroys most fascicle pixels
2. **Area-averaged downscaling + low threshold (0.15)** — standard nearest-neighbour resizing loses 1–2 px lines entirely
3. **Mask dilation (2 px)** during training — makes thin targets learnable without affecting geometric measurements at inference
4. **Foreground-centred random crops** (384×384) — 80% of training crops are centred near a foreground pixel so the model sees useful signal rather than empty muscle tissue
5. **Hysteresis thresholding** at inference — repairs broken fascicle segments by keeping low-confidence pixels that are connected to high-confidence ones

---

### Geometry Engine

After segmentation, classical computer vision extracts the architecture parameters:

1. **Fit aponeuroses**: Connected components → per-column median y-coordinate → `np.polyfit` → two parallel lines (superficial, deep)
2. **Muscle Thickness**: Median perpendicular gap between the two fitted aponeurosis lines
3. **Fascicle statistics**: For each fascicle component → `cv2.fitLine` → compute angle relative to deep aponeurosis (PA) → intersect fascicle line with both aponeurosis lines → compute length (FL)
4. **Robust aggregation**: Length-weighted median across all detected fascicles reduces noise from spurious detections

---

## 📁 Project Structure

```
ultrasound-images-/
├── colab_muscle_segmentation_IMPROVED2.ipynb   ← Main pipeline notebook
└── README.md                                   ← This file
```

### Notebook Sections

| Section | Contents |
|---------|----------|
| **1. Setup** | Imports, `CFG` hyperparameter dataclass, device selection, seed |
| **2. Data** | Google Drive mount, dataset discovery, zip extraction |
| **3. Data Audit** | Paired file matching, mask sanity checks, stroke-width measurement, overlay visualisation |
| **4. Group-wise Split** | Train/val/test split that respects subject/sequence boundaries to prevent data leakage |
| **5. Canvas, Cache, Augmentation** | Letterbox resizing, numpy cache, foreground-centred random crops, augmentation pipeline |
| **6. Model, Loss, Metrics** | ResNet34 U-Net, AttentionGate, ComboLoss, Dice/IoU/boundary-tolerant F1 |
| **7. Training** | Training loop with mixed-precision (AMP), OneCycleLR, early stopping, val prediction viewer |
| **8. Threshold Tuning** | Sweep binarisation threshold on validation set; report honest hold-out test score |
| **9. Clinical Geometry** | Aponeurosis line fitting, MT calculation, fascicle angle and length extraction |
| **10. Inference** | Segment all test images, fill failed measurements with median, export results CSV |
| **11. Visual Check** | Side-by-side overlays of segmentation masks and computed parameters |

---

## 🛠️ Dependencies

```
Python 3.9+
torch >= 2.0
torchvision
numpy
pandas
opencv-python (cv2)
Pillow
matplotlib
```

The notebook is designed to run on **Google Colab** with GPU acceleration. The dataset is loaded from **Google Drive**.

---

## ⚙️ Configuration (`CFG`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_SIDE` | 1024 | Maximum long-side of the canvas (px) |
| `PX_PER_MM` | 8.0 | Pixel-to-mm conversion — ⚠️ verify against your dataset's scale bar |
| `BATCH` | 12 | Training batch size (384×384 crops) |
| `EPOCHS_FASC` | 40 | Fascicle model training epochs |
| `EPOCHS_APO` | 30 | Aponeurosis model training epochs |
| `FASC_DILATE` | 2 | Fascicle mask dilation radius (px) |
| `TOL_PX` | 3 | Tolerance (px) for boundary-tolerant F1 metric |
| `LR` | 3e-4 | Peak learning rate |
| `PATIENCE` | 10 | Early stopping patience (epochs without val-Dice gain) |
| `TTA` | True | Horizontal-flip test-time augmentation |
| `SEED` | 42 | Global random seed for reproducibility |

---

## 📊 Expected Dataset Layout

The dataset should be in Google Drive at `MyDrive/ml data/` with the following structure:

```
ml data/
├── fasc_imgs/       ← Fascicle training images
├── fasc_masks/      ← Fascicle binary masks
├── apo_imgs/        ← Aponeurosis training images
├── apo_masks/       ← Aponeurosis binary masks
└── test_imgs/       ← Images for inference
```

---

## 📓 Development Journal — Division of Work Between Two People

This project naturally splits into two independent but connected workstreams that two people can develop in parallel, integrating at key checkpoints.

---

### 👤 Person A — Model Development & Training

**Owns**: Everything inside the neural network — architecture design, loss functions, training pipelines, and segmentation quality.

---

#### Week 1–2 · Data Understanding & Baseline

- [ ] Set up the Colab environment and mount Google Drive
- [ ] Run the data audit (Section 3) — understand image sizes, mask quality, stroke widths
- [ ] Implement the group-wise train/val/test split (Section 4) to prevent data leakage
- [ ] Build a minimal baseline: standard U-Net with BCE loss on downscaled images
- [ ] Measure baseline Dice and tolF1 scores to establish a starting point

**Key questions to answer during this phase:**
- What is the typical fascicle stroke width? (drives the dilation radius setting)
- Are masks stored as 0/255 or 0/1? (affects the `binarize_mask` logic)
- Are there any duplicate images that would inflate validation metrics?

---

#### Week 3–4 · Model Architecture Improvements

- [ ] Swap plain U-Net for **ResNet34 U-Net** with ImageNet encoder (Section 6)
- [ ] Add **Attention Gates** on all skip connections
- [ ] Implement the **full-resolution stream** (`self.hr`) for thin-line preservation
- [ ] Convert encoder input from 3-channel RGB to 1-channel grayscale using the weight summation trick
- [ ] Verify the architecture produces sensible outputs on a single batch before full training

---

#### Week 5–6 · Loss Function & Training Loop

- [ ] Implement **ComboLoss** (weighted BCE + Focal-Tversky + clDice)
- [ ] Tune `w_cl` (clDice weight) for the fascicle model; keep `w_cl=0` for aponeurosis
- [ ] Implement differential learning rates: encoder at `0.3×LR`, decoder at `1×LR`
- [ ] Set up **OneCycleLR** scheduler with `pct_start=0.1`
- [ ] Add gradient clipping (`max_norm=1.0`) and mixed-precision training (AMP)
- [ ] Add early stopping on validation Dice with patience=10
- [ ] Log training curves (loss, Dice, IoU, tolF1) and save best checkpoints

---

#### Week 7 · Evaluation & Threshold Tuning

- [ ] Implement **boundary-tolerant F1** (`tol_f1`, ±3 px) alongside standard Dice/IoU
- [ ] Sweep binarisation thresholds on the validation set — do NOT use the test set for this
- [ ] Report the hold-out test score with the tuned threshold (this is the honest performance estimate)
- [ ] Inspect validation predictions visually using `show_val_predictions`
- [ ] Document final val and test scores for both models in a results table

**Deliverable**: Two trained model checkpoints (`fascicle_best.pth`, `aponeurosis_best.pth`) with documented performance metrics.

---

### 👤 Person B — Data Pipeline & Geometry Engine

**Owns**: Everything outside the neural network — data loading, augmentation, post-processing, and the geometric extraction of clinical parameters.

---

#### Week 1–2 · Data Pipeline

- [ ] Implement robust paired-file matching by stem (handles `.png` ↔ `.tif` mismatches)
- [ ] Implement the `binarize_mask` fix (handles both 0/255 and 0/1 stored masks)
- [ ] Implement the letterbox resize (`letterbox`) that preserves aspect ratio and pads with zeros
- [ ] Implement `prep_mask`: area-averaged downscaling + low threshold + optional dilation
- [ ] Build the numpy cache (`build_cache`) to avoid re-reading images every epoch
- [ ] Implement `sample_window`: random crops biased towards foreground pixels

---

#### Week 3–4 · Augmentation & Dataset Classes

- [ ] Implement the full augmentation pipeline (`augment`):
  - Horizontal flip (50%)
  - Small affine transforms: rotation ±8°, scale 0.9–1.1, translation ±5% (80%)
  - Gamma and contrast/brightness jitter (70%)
  - Speckle-like multiplicative + additive noise (50%)
  - Gaussian blur (20%)
- [ ] Implement `SegDataset` with train/val modes (crops vs. full canvas)
- [ ] Set up `DataLoader` with `num_workers=2`, `pin_memory=True`, `persistent_workers=True`
- [ ] Verify augmented samples visually — masks and images must stay aligned after every transform

---

#### Week 5–6 · Geometry Engine

- [ ] Implement `fit_aponeuroses`: connected components → per-column median → linear regression → classify superficial vs. deep by mean y-coordinate
- [ ] Implement `muscle_thickness_px`: perpendicular distance between fitted aponeurosis lines
- [ ] Implement `fascicle_stats`: `cv2.fitLine` → compute angle → filter implausible fascicles (3°–60°) → intersect with aponeurosis lines → compute FL → length-weighted median PA
- [ ] Implement `calculate_clinical_parameters`: compose the above into PA/FL/MT with NaN fallbacks
- [ ] Validate geometry on a few manually annotated images — does the computed PA match what you'd measure by eye?

---

#### Week 7 · Inference Pipeline & Output

- [ ] Implement `hysteresis`: connect broken fascicle segments at inference time
- [ ] Implement `segment_full_res`: letterbox → predict → crop padding → resize back to original resolution → threshold
- [ ] Run the full inference loop over new images
- [ ] Implement robust NaN filling: failed measurements filled with the **median** of successful predictions (not a hard-coded constant)
- [ ] Implement the visual check overlay (Section 11) — green for fascicles, pink for aponeuroses, cyan for fitted aponeurosis lines

**Deliverable**: A working end-to-end inference pipeline that takes unseen images and outputs PA, FL, and MT for each one.

---

### 🤝 Integration Points (Both Work Together)

| Checkpoint | When | What to align on |
|------------|------|-----------------|
| Canvas size | End of Week 2 | `CANVAS` dict — must match between training and inference |
| `PX_PER_MM` | Before Week 5 | Verify against dataset scale bar — directly affects FL and MT output |
| Threshold values | End of Week 7 | Person A tunes `THR` → Person B uses it in `segment_full_res` |
| Checkpoint paths | Week 7 | Agree on where `.pth` files live in Drive |

---

### 📅 Timeline Summary

| Week | Person A | Person B |
|------|----------|----------|
| 1–2 | Baseline model + data audit | Paired file matching + letterbox + cache |
| 3–4 | ResNet34 U-Net + Attention Gates | Augmentation pipeline + SegDataset |
| 5–6 | ComboLoss + training loop | Geometry engine (aponeuroses + fascicles) |
| 7 | Threshold tuning + evaluation | Inference pipeline + result export |
| 8 | Code review + documentation | Code review + documentation |

---

## ⚠️ Important Notes

> [!IMPORTANT]
> **Verify `PX_PER_MM`** before running inference. The default value of `8.0` is a placeholder. The actual pixel-to-mm scale depends on the ultrasound probe settings used when each image was acquired. Always check the scale bar in the images or the accompanying metadata.

> [!WARNING]
> **Data leakage risk**: The group-wise split tries to keep consecutive frames from the same video sequence in the same split. If your filenames don't encode subject/sequence IDs, the code falls back to blocks of 50 consecutive frames. Always visually confirm that your val/test sets contain genuinely unseen subjects.

> [!TIP]
> After a Colab runtime disconnect, the numpy cache is preserved on Drive. The `build_cache` function checks for an existing file and skips re-extraction — so reconnecting is fast with no data re-loading.

---

## 🙏 Acknowledgements

- **DL_Track_US** — open-source ultrasound muscle architecture analysis tool (inspiration for the geometry engine)
- **Soft Skeleton / clDice** — Shit et al., 2021 — topology-preserving loss for tubular structures
- **Attention U-Net** — Oktay et al., 2018 — attention-gated skip connections for medical image segmentation
- **ResNet** — He et al., 2016 — backbone encoder with ImageNet pre-training

---

## 📄 License

This project is released under the **MIT License**.
