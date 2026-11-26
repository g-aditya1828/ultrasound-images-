# 🚀 Aspirational Methods — What We Wanted to Do But Couldn't (And Why)

> This document is honest about the gap between what this project does and what a well-resourced team with more time, data, and compute would do. Each section explains the method, why it would help, and exactly what constraint prevented its use.

---

## 1. Transformer-Based Segmentation (Swin-UNet / SegFormer)

### What it is
Vision Transformers (ViTs) replaced convolutional encoders with self-attention — the same mechanism that powers large language models. Applied to segmentation, Swin-UNet uses a Swin Transformer encoder (hierarchical shifted-window attention) in place of ResNet, achieving state-of-the-art on many medical image benchmarks. SegFormer is a more lightweight variant.

### Why it would help
- **Global context**: Self-attention computes relationships between every pair of image patches simultaneously. A ResNet encoder at the deepest layer has a receptive field of fixed size; Swin-Transformer sees the entire image at every layer. For aponeurosis detection, knowing "the structure at the top of the image is geometrically parallel to what I see at the bottom" is valuable global context — ResNet can only infer this implicitly.
- **No hard inductive bias**: CNNs assume local features (locality bias). Fascicle orientation is a global property (many parallel lines spanning the image) — transformers can reason about this more naturally.

### Why we didn't use it

| Constraint | Detail |
|-----------|--------|
| **Data** | ViT-based models need 10,000+ images to reach their potential; below that, the CNN's inductive bias wins. Our dataset has hundreds of images. |
| **Compute** | Swin-UNet at reasonable resolution requires a GPU with >12 GB VRAM for training. Colab's T4 has 16 GB but at our native canvas sizes (up to 1024px), memory is already tight with ResNet34. |
| **Pretraining** | Good Swin Transformer segmentation pretraining is on ImageNet-22K (14M images) — not freely available as a simple torchvision download, requires additional setup. |
| **Complexity** | For a two-person project, adding a transformer backbone significantly increases debugging surface area and training instability risk. |

**What we'd do with more resources**: Replace the ResNet34 encoder with a Swin-T (tiny) or Swin-S (small) encoder pretrained on ImageNet-22K, keeping the attention-gated decoder. Expected Dice gain on fascicle: ~5–8%.

> **Reference**: Cao et al. (2022) — *Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation* [⁹]

---

## 2. Semi-Supervised Learning (Pseudo-Labelling / Mean Teacher)

### What it is
Semi-supervised learning uses both labelled images (with ground-truth masks) and unlabelled images (no masks) during training. The idea: train on labelled data first, then use the model to generate "pseudo-labels" for unlabelled images, then retrain on the full set. The **Mean Teacher** variant maintains a "teacher" model (exponential moving average of the student) to generate more stable pseudo-labels.

### Why it would help
Annotation of ultrasound images is expensive — it requires a trained biomechanist to manually trace every fascicle and aponeurosis. The UMUD database contains many more raw unlabelled ultrasound images than annotated ones. Semi-supervised learning could leverage those unlabelled images to learn better feature representations.

### Why we didn't use it

| Constraint | Detail |
|-----------|--------|
| **Data access** | We worked with the provided annotated dataset only; unlabelled raw sequences were not provided in the same format |
| **Stability** | Pseudo-label quality depends on the base model quality. For thin-line segmentation where the base model Dice is ~0.52, pseudo-labels would be noisy enough to hurt training rather than help |
| **Complexity** | Requires maintaining two models simultaneously (teacher + student), doubled memory requirements |
| **Time** | The iterative pseudo-labelling loop adds multiple full training cycles |

**What we'd do with more resources**: Apply Mean Teacher semi-supervised training using the unlabelled frames from the same video sequences, with consistency regularisation between augmented views of the same image.

> **Reference**: Tarvainen & Valpola (2017) — *Mean Teachers are Better Role Models* [¹⁰]

---

## 3. Temporal Consistency Across Video Frames

### What it is
The test dataset includes sequences of 5 consecutive video frames. Currently, we process each frame independently. A temporal model would process all 5 frames together, enforcing that:
- Muscle thickness doesn't jump suddenly between adjacent frames (smoothness prior)
- Fascicle angle changes gradually (temporal consistency loss)
- Structures tracked across frames should be the same physical structure

This is essentially a 2D segmentation problem extended with a time dimension — using recurrent networks (ConvLSTM) or 3D convolutions over the temporal axis.

### Why it would help
- **Noise reduction**: Averaging predictions across 5 frames gives much smoother PA/FL/MT estimates
- **Error correction**: If the model misidentifies a structure in one frame but correctly identifies it in adjacent frames, temporal context can resolve the ambiguity
- **Stability metric**: The temporal smoothness of predictions is itself a quality indicator

### Why we didn't use it

| Constraint | Detail |
|-----------|--------|
| **Training data format** | Ground-truth annotations are per-frame, not per-sequence; temporal supervision would require sequence-level annotations with frame correspondence |
| **Architecture complexity** | ConvLSTM or 3D U-Net dramatically increases model complexity and memory usage |
| **Data structure** | The 5-frame sequences are only in the test set; training data is frame-by-frame, so there's no temporal training signal available |
| **Inference complexity** | Would require batching images in groups of 5 with sequence identity information — complicates the inference pipeline significantly |

**What we'd do with more resources**: Train a lightweight ConvLSTM temporal refinement module that takes 5 consecutive frame predictions and outputs temporally-smoothed PA/FL/MT estimates.

---

## 4. Uncertainty Quantification (Monte Carlo Dropout / Deep Ensembles)

### What it is
Standard neural networks give a single point prediction — a probability map — with no indication of how confident the model is. Uncertainty quantification methods provide a **confidence interval** around each prediction.

- **MC Dropout**: Enable dropout at inference time and run the model N times. The variance across runs estimates prediction uncertainty.
- **Deep Ensembles**: Train K independent models. The variance across their predictions estimates uncertainty.
- **Practical value**: Flag images where the model is uncertain (e.g., Dice uncertainty > threshold) for manual review rather than blindly trusting automated measurements.

### Why it would help
For clinical use, knowing *when to not trust the model* is as important as the predictions themselves. If the model predicts PA=15° with high confidence, that's usable. If it predicts PA=15° but uncertainty is ±12°, that result should be reviewed by a human.

### Why we didn't use it

| Constraint | Detail |
|-----------|--------|
| **Inference time** | MC Dropout requires N forward passes (typically 20–50) per image. Combined with TTA (already 2 passes), this would be 40–100 passes per image — impractical for large datasets |
| **Deep ensembles** | Training K=5 independent models at our canvas size would require 5× the training time and 5× the storage. With 8 weeks of project time, this is not feasible |
| **Calibration** | Uncertainty estimates from dropout are known to be poorly calibrated for segmentation tasks without additional temperature scaling |

**What we'd do with more resources**: Train a 3-model ensemble (different random seeds) and use the inter-model variance as a pixel-wise uncertainty map. Flag images where mean uncertainty exceeds a threshold for human review.

> **Reference**: Lakshminarayanan et al. (2017) — *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* [¹¹]

---

## 5. nnU-Net (Self-Configuring Segmentation)

### What it is
nnU-Net ("no-new-Net") is a framework that automatically configures U-Net architecture, preprocessing, training, and post-processing based on dataset properties. Given a dataset, it:
- Automatically determines optimal input resolution and patch size
- Automatically selects 2D vs. 3D training configuration
- Automatically tunes all hyperparameters
- Has achieved state-of-the-art results across dozens of medical segmentation benchmarks without manual tuning

### Why it would help
Our project manually tuned canvas size, dilation, loss weights, batch size, and training duration. nnU-Net would systematically find better configurations than manual search.

### Why we didn't use it

| Constraint | Detail |
|-----------|--------|
| **Black box** | nnU-Net automates decisions we needed to understand, explain, and justify as a learning project |
| **Customisation** | Adding clDice loss, attention gates, and the full-resolution stream inside nnU-Net's framework is non-trivial |
| **Environment** | nnU-Net expects a specific dataset format (MSD format) and has non-trivial installation requirements that conflicted with our Colab-first setup |
| **Learning value** | The goal of this project included understanding *why* each design choice works — nnU-Net abstracts that away |

**What we'd do with more resources**: Run nnU-Net as a strong baseline to benchmark our custom architecture against. If nnU-Net outperforms our model, understand which of its auto-configurations we should adopt.

> **Reference**: Isensee et al. (2021) — *nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation* [¹²]

---

## 6. Test-Time Augmentation Beyond Horizontal Flip

### What it is
Test-Time Augmentation (TTA) runs multiple augmented versions of each test image through the model and averages the predictions. Currently we apply only horizontal flip (2 passes per image). A full TTA suite would include:
- Vertical flip
- 90°, 180°, 270° rotations
- Multiple scale factors (0.9×, 1.0×, 1.1×)
- Slight brightness variations

### Why it would help
Averaging 8–10 augmented views reduces prediction variance and typically improves Dice by 1–3% at no training cost — only inference cost.

### Why we didn't use it

| Constraint | Detail |
|-----------|--------|
| **Geometric validity** | Vertical flip and 90° rotation change the direction of gravity — in ultrasound, the superficial aponeurosis is always at the top. Flipping vertically would put the deep aponeurosis on top, confusing the geometry engine. |
| **Inference time** | Each additional TTA pass multiplies inference time linearly. For large test sets, 8× TTA at native resolution is slow. |
| **Rotation realism** | Real ultrasound images are always captured roughly horizontally — rotating 90° produces an unrealistic orientation the model was never trained on. |

**Practical resolution**: Horizontal flip is the only geometrically valid TTA for horizontal ultrasound images. It is already applied. This is not a significant limitation.

---

## 7. Pixel-Wise Scale Calibration from Image Metadata

### What it is
The `PX_PER_MM` constant (currently hardcoded as 8.0) converts pixel distances to millimetres. In practice, this value varies per image — it depends on the ultrasound probe frequency, depth setting, and zoom level. The scale bar embedded in each ultrasound image encodes the actual calibration.

An ideal system would:
1. Detect the scale bar region in the image using a secondary detector
2. Parse the scale bar markings (e.g., "10 mm")
3. Count how many pixels span the labelled distance
4. Compute `PX_PER_MM` automatically per image

### Why it would help
Any systematic error in `PX_PER_MM` scales FL and MT outputs proportionally. If the true scale is 7.5 px/mm but we use 8.0, all measurements are off by 6.7% — a clinically meaningful error.

### Why we didn't use it

| Constraint | Detail |
|-----------|--------|
| **OCR complexity** | Reading the scale bar text ("10 mm") requires optical character recognition — a separate model or library (Tesseract) with its own dependencies and failure modes |
| **Scale bar location variability** | Different ultrasound machines place the scale bar in different corners; detecting it reliably requires another detection step |
| **Image crop variation** | Some images in the dataset have the scale bar cropped out, making per-image calibration impossible for a subset |
| **Time** | Implementing a robust scale bar parser is a significant engineering task beyond the project scope |

**What we'd do with more resources**: Train a lightweight detector to localise the scale bar region, extract it, and use a template-matching or OCR approach to read the calibration. This is the single highest-impact missing feature for clinical-grade accuracy.
