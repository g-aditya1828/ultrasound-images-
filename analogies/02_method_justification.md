# ⚖️ Method Justification — Why This Stack and Not Something Else?

> For every major technical decision, there were alternatives. This document explains what those alternatives were, why they fall short for this specific problem, and why the chosen approach is the right one.

---

## Decision 1 — Deep Learning vs. Classical Computer Vision

### The Alternative: Classical CV Pipeline
Before deep learning, muscle architecture was measured using:
- **Hough Transform** — detects straight lines in images by voting in parameter space
- **Active Contours / Snakes** — deformable curves that fit themselves to edges in an image
- **Radon Transform** — projects image intensity along angular sweeps to find dominant orientations
- **Manual thresholding + morphological operations** — threshold the image to separate bright aponeuroses from dark muscle, then apply erosion/dilation to clean up

**Why it doesn't work well here:**

| Issue | Classical CV | Deep Learning |
|-------|-------------|---------------|
| Ultrasound speckle noise | Hough/active contours are confused by noise | Learned to ignore noise after seeing many examples |
| Variable image quality | Needs re-tuning of parameters per image | Generalises across gain settings, probe models, depths |
| Multiple overlapping fascicles | Hough finds dominant lines, misses minor ones | Learns to segment all fascicles as a class |
| Partial aponeurosis visibility | Active contours fail when the boundary is interrupted | Predicts even occluded regions from context |
| Scale calibration | Must manually set all thresholds | Learns from data; parameters set once |

**The analogy**: Classical CV is like giving someone a fixed ruler and saying "find all lines that are exactly 45° ± 5°." Deep learning is like training someone to recognise fascicles by showing them thousands of labelled examples — they learn what fascicles look like regardless of exact angle, contrast, or image quality.

**When classical CV is better**: When data is scarce (< 50 images), when interpretability is mandatory, or when the exact structure follows a known mathematical form. For this dataset, with hundreds of annotated images and high inter-image variability, deep learning is clearly superior.

> **Reference**: Rana et al. (2009) demonstrated semi-automated fascicle detection using Hough transform but noted 15–20% error rates and significant failure on low-contrast images [⁶].

---

## Decision 2 — U-Net vs. Other Segmentation Architectures

### The Alternatives

**A. Fully Convolutional Network (FCN)**
- The original segmentation network — just an encoder with skip connections at the final layer
- Problem: coarse skip connections at only 1–2 scales; loses fine-grained detail at intermediate scales
- Our fascicles need **multi-scale** skip connections (see U-Net skip connections analogy)

**B. DeepLab v3+ (Atrous Spatial Pyramid Pooling)**
- Excellent for semantic segmentation of large objects (roads, buildings, sky)
- Uses dilated convolutions to capture multi-scale context without losing resolution
- Problem: designed for objects that occupy significant image area; fascicles are thin lines (~0.4% of pixels). ASPP captures context at scales much larger than a 1px line.

**C. SegNet**
- Encoder-decoder like U-Net, but uses pooling indices (not feature maps) in skip connections
- Saves memory by not storing full feature maps
- Problem: for thin-line preservation, full feature map skip connections are essential — pooling indices only tell you *where* the max was, not the full spatial activation pattern. Fine fascicle features are lost.

**D. Mask R-CNN**
- Instance segmentation — detects and segments individual objects
- Would treat each fascicle as a separate instance
- Problem: fascicles don't have clear bounding boxes (they cross the image diagonally), overlap in confusing ways, and vary enormously in length. The region proposal network would struggle to propose meaningful boxes.

**Why U-Net wins here:**
U-Net was specifically designed for **biomedical image segmentation** where:
- Training data is limited (you can't collect millions of labelled ultrasound images)
- Objects of interest span multiple scales (aponeuroses are large; fascicle lines are thin)
- Precise pixel-level localisation matters (not just "there's a fascicle somewhere in this region")

The encoder-decoder symmetry with skip connections at every scale is exactly what thin-line segmentation needs.

> **Reference**: Ronneberger et al. (2015) [¹] originally demonstrated U-Net on cell segmentation — another biomedical task with thin structures and limited data. The design principles transfer directly.

---

## Decision 3 — ResNet34 vs. Other Encoder Backbones

### The Alternatives

**A. VGG16/VGG19 Encoder**
- Very deep, all 3×3 convolutions, no skip connections inside the encoder
- Used in original FCN and many early segmentation works
- Problem: **no residual connections** inside the encoder → vanishing gradients in early layers → poor gradient flow for fine-grained features. Also large (138M parameters) → overfits on limited data, slow to train.

**B. ResNet50 / ResNet101**
- Deeper than ResNet34 → more capacity, but: more parameters to fine-tune on limited data, slower per epoch, higher VRAM usage
- On Colab's T4 GPU with native-resolution crops, ResNet50 barely fits in memory at batch=12
- ResNet34 gives 95% of the performance at 40% of the compute

**C. EfficientNet-B4 / B5**
- State-of-the-art efficiency — achieves high accuracy with fewer parameters via compound scaling
- Problem: EfficientNet was optimised for image classification, not dense prediction. Its aggressive channel reduction at early layers removes fine spatial detail that is critical for thin-line segmentation.
- Also: MBConv blocks with depthwise separable convolutions don't transfer as cleanly as standard ResNet blocks for grayscale ultrasound domain adaptation.

**D. Vision Transformer (ViT) encoder**
- Pure attention-based; no convolutional inductive bias
- Excellent for high-data-regime classification (ImageNet)
- Problem: requires large amounts of data to train well (ViT-B needs ~100K+ images to match ResNet); our dataset is orders of magnitude smaller. Also, patch-based tokenisation (16×16 patches) loses sub-patch detail — catastrophic for 1px fascicle lines.

**Why ResNet34 wins:**
- Residual connections maintain gradient flow to early layers → fine-detail features are preserved
- Moderate depth (21M parameters) → appropriate for dataset size without severe overfitting
- Strong ImageNet pretraining → good initialisation for edge/texture detection in noisy images
- Well-understood transfer learning behaviour on medical imaging (extensively studied in literature)

> **Reference**: Hao et al. (2020) systematically compared ResNet variants as U-Net encoders on medical imaging datasets and found ResNet34 to be optimal for datasets of 200–2000 images [⁷].

---

## Decision 4 — ComboLoss vs. Standard Loss Functions

### The Alternatives

**A. Binary Cross-Entropy (BCE) alone**
- The default loss for binary segmentation
- Problem: with 0.4% foreground fraction, the model achieves near-zero loss by predicting all background. Even with pos_weight, BCE penalises each pixel independently — it has no concept of whether predictions are spatially coherent or connected.

**B. Dice Loss alone**
- Directly optimises the Dice coefficient (the standard evaluation metric)
- Problem: Dice is undefined (or 1 by convention) when both prediction and ground truth are empty. For images with no visible fascicles (acoustic shadow, low contrast), Dice loss is uninformative. Also, Dice penalises false positives and false negatives equally — for thin-line detection, missing a fascicle entirely (false negative) is worse than a small false positive fragment.

**C. Focal Loss**
- Lin et al. (2017) designed Focal Loss to address class imbalance in object detection
- Reduces loss contribution from easy (correctly classified, high-confidence) examples
- Problem: Focal Loss is still pixel-independent — it doesn't know or care whether the predicted positive pixels form connected lines or isolated specks.

**D. IoU Loss**
- Directly optimises Intersection over Union
- Similar limitations to Dice — pixel-independent, no topology awareness

**Why ComboLoss (BCE + Focal-Tversky + clDice) wins:**
Each term fixes a different failure mode:
- **Weighted BCE**: fixes the class imbalance problem
- **Focal-Tversky**: fixes the recall bias problem (we care more about missing fascicles than false alarms)
- **clDice**: fixes the topology problem (rewards connected, continuous predictions)

No single loss function addresses all three simultaneously. The combination is not "more complex for the sake of it" — each term has a measurable, specific effect on a known failure mode.

> **References**: Lin et al. (2017) [⁸] for Focal Loss; Abraham & Khan (2019) [⁴] for Focal-Tversky; Shit et al. (2021) [⁵] for clDice.

---

## Decision 5 — Foreground-Centred Crops vs. Random Crops

### The Alternative: Fully Random Crops
- Sample a random 384×384 patch from anywhere in the image
- Simple, unbiased

**Problem**: Fascicles occupy ~0.4% of image area. If you sample a random 384×384 crop from a 1024×800 image, there's roughly an **85% chance the crop contains no fascicle pixels at all**. The model would spend 85% of training steps looking at empty muscle tissue, learning "predict background everywhere."

**Why foreground-centred crops win:**
By centering 80% of crops near a randomly chosen foreground pixel (sub-sampled at 4× for speed), every training step shows the model at least some fascicle structure. The 20% random crops are kept to prevent the model from never seeing the global context (where fascicles are relative to the aponeuroses).

**The analogy**: Teaching a student to recognise cats by showing them 1000 photos — except 850 photos are empty rooms with no cats. Vs. showing them 800 photos of rooms with cats clearly visible, and 200 of empty rooms (so they know what "no cat" looks like).

---

## Decision 6 — Two Separate Models vs. One Multi-Class Model

### The Alternative: Single Model with Two Output Channels
- One U-Net with a 2-channel output — channel 0 = fascicle probability, channel 1 = aponeurosis probability
- Simpler pipeline, one training run, shared feature extraction

**Why we chose two separate models:**

| Property | Fascicles | Aponeuroses |
|----------|-----------|-------------|
| Width | 1–3 px | 5–20 px |
| Orientation | Diagonal (~15°–30°) | Near-horizontal |
| Spatial extent | Fragments, partial coverage | Full image width |
| Density | Multiple parallel lines | Exactly 2 structures |
| Optimal loss | BCE + Focal-Tversky + clDice | BCE + Focal-Tversky |
| Optimal dilation | 2 px | 0 px |
| Training epochs | 40 | 30 |

These two structures have such different visual and statistical properties that a shared model is forced to compromise. Separate models allow separate hyperparameters, separate loss functions, and separate training schedules — each optimised independently.

**The analogy**: Training one person to be both a surgeon and a painter. Technically possible, but a dedicated surgeon will be better at surgery and a dedicated painter will be better at painting.

---

## Decision 7 — Adaptive Canvas vs. Fixed Canvas

### The Alternative: Fixed 512×640 Canvas
- Resize all images to exactly 512×640 before training and inference
- Simple, deterministic, easy to implement

**Problem**: This is the single most damaging design choice in the original baseline. Here's why:

Suppose a native image is 800×600 pixels. A fascicle line is 1 pixel wide.
- After resizing to 512×640: scale factor ≈ 0.64 → the 1px line becomes 0.64px → after nearest-neighbour rounding, **the line either disappears or becomes a disconnected single pixel every other row**
- After INTER_AREA downscaling with threshold 0.15: the line contributes 0.64 coverage to each destination pixel → **survives the threshold**, line is preserved

**Why adaptive canvas wins:**
By computing the canvas from the actual image dimensions (capped at MAX_SIDE=1024), native-resolution thin lines remain native-resolution thin lines. The model learns on data that actually contains the signal it needs to detect.

**The tradeoff**: Variable canvas sizes require the network to handle variable input sizes. U-Net with fully convolutional architecture handles this naturally — no fixed spatial dimensions in the network.
