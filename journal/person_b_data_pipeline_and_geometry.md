# 📒 Person B — Data Pipeline & Geometry Engine Journal

**Role**: Data loading, augmentation, post-processing, and clinical geometry extraction  
**Works on**: Sections 2 (data), 3 (audit), 5 (canvas/cache/augmentation), 9 (geometry), 10 (inference), 11 (visual check)

---

## Week 1–2 · Data Pipeline Foundation

### Day 1
Set up Colab and mounted Google Drive. Explored the dataset folder structure manually before writing any code. Observations:
- Image and mask files don't always share the same extension — some images are `.png` while masks are `.tif`. A naive filename match would miss these pairs.
- Some filenames have trailing numbers that look like frame indices (e.g., `subject03_sweep2_0047.png`) — these will be useful for the group-wise split (Person A's job) but I need to be aware of them when pairing files.

Implemented `get_paired_files` using **stem matching** (filename without extension) instead of full filename matching. This correctly pairs `img_001.png` with `img_001.tif` and reports unmatched files so we don't silently lose data.

### Day 2
Implemented `binarize_mask`:
```python
def binarize_mask(arr):
    return (arr > 0) if arr.max() <= 1 else (arr > 127)
```
This handles both storage formats — masks stored as 0/1 (common in some annotation tools) and 0/255 (standard). The old `> 127` threshold would silently return all-zeros for 0/1 masks, which would cause the model to train on empty ground truth without any error.

Ran the audit. Key findings documented and shared with Person A:
- Fascicle stroke width: **median ~1.8 px** at native resolution. This is extremely thin. Standard nearest-neighbour downscaling will lose most of these pixels.
- Foreground fraction (fascicle): ~0.004 (0.4% of pixels are foreground). Heavy class imbalance.
- Some mask-image size mismatches found — handled by resizing masks to match image size using nearest-neighbour before caching.

### Day 3–4
Implemented `letterbox` resize:
- Preserves the original aspect ratio by computing a uniform scale factor `s = min(H/h, W/w)`
- Pads the bottom and right with zeros (black padding)
- Critically: does NOT stretch the image. Stretching would distort fascicle angles, making the geometry engine measure wrong angles.

Implemented `prep_mask`:
- For downscaling: uses **area-averaging** (`cv2.INTER_AREA`) + threshold of **0.15** (not 0.5). This is the key fix — with area averaging, a 1 px line that occupies 20% of a destination pixel still passes the threshold. With nearest-neighbour or a 0.5 threshold, it disappears.
- For upscaling: uses bilinear + 0.5 threshold (standard)
- Optional dilation with an elliptical 3×3 kernel to thicken thin targets for training

### Day 5
Built the **numpy cache** (`build_cache`):
- Loads all images and masks, applies letterbox + prep_mask, stacks into two big numpy arrays `X` and `Y`
- Saves to `.npz` on Drive so subsequent runs (after Colab disconnects) skip re-loading
- Checks the cached file length matches the current dataset length before trusting it

Memory estimate printed at build time: `~2 × N × H × W / 1e9` GB — important for knowing if Colab's RAM will handle it.

### Day 6
Implemented `sample_window`:
- With probability 0.8, samples a crop centred near a randomly chosen foreground pixel (sub-sampled at 4× for speed)
- With probability 0.2, samples a fully random position anywhere in the image
- This ensures most training crops actually contain fascicles, rather than showing the model endless patches of uniform muscle tissue

---

## Week 3–4 · Augmentation & Dataset Classes

### Day 1–2
Implemented the full augmentation pipeline in `augment(img, msk)`. Every transform is applied identically to both the image and mask (no misalignment). Transforms applied in order:

1. **Horizontal flip** (p=0.5) — `img[:, ::-1]`, `msk[:, ::-1]`
2. **Small affine** (p=0.8):
   - Rotation: ±8° (small — large rotations are unrealistic for ultrasound which is held roughly parallel)
   - Scale: 0.9–1.1
   - Translation: ±5% of image dimensions
   - Image: `cv2.INTER_LINEAR`, Mask: `cv2.INTER_NEAREST` (critical — must not blur mask edges)
3. **Gamma + contrast/brightness** (p=0.7) — simulates ultrasound gain settings
4. **Speckle noise** (p=0.5) — multiplicative + small additive noise mimics ultrasound speckle pattern
5. **Gaussian blur** (p=0.2) — simulates focus depth variation

Verified visually on 20 random samples: masks and images stay correctly aligned through all transforms.

### Day 3
Implemented `SegDataset`:
- **Train mode**: calls `sample_window` → `augment` → centre-crop to `(CROP, CROP)`
- **Val/test mode**: returns the full canvas image, normalised but no augmentation
- Both modes normalise: `(img / 255 - MEAN) / STD`
- Uses shared `X` and `Y` arrays (no per-split copies) — halves RAM usage

```
Train batch shape: (12, 1, 384, 384)
Val batch shape:   (4,  1, H,   W  )   ← full canvas
```

### Day 4
Set up `DataLoader` with:
- `num_workers=2` — parallel data loading without oversubscribing CPU
- `pin_memory=True` — faster GPU transfer
- `persistent_workers=True` — avoids re-spawning workers every epoch
- `drop_last=True` for training — avoids partial batches that would skew BatchNorm statistics

Measured data loading speed: with caching and 2 workers, training batches arrive faster than the GPU processes them — no data bottleneck.

### Day 5–6
Stress-tested the augmentation pipeline on edge cases:
- Very small images (< 384 px on the short side): crop size automatically capped to image size — no crash
- Empty masks (all-zero): augmentation passes through cleanly — no division by zero in `sample_window` (falls through to random sampling)
- Images with extreme aspect ratios: letterbox pads correctly without distorting

---

## Week 5–6 · Geometry Engine

### Day 1–2
Implemented `fit_aponeuroses`:

The core challenge is that the aponeurosis mask might contain multiple connected components (noise, partial detections). Need to robustly identify exactly two sheets — superficial and deep.

Algorithm:
1. Find all connected components with area ≥ 300 px (`cv2.connectedComponentsWithStats`)
2. Discard components that span less than 15% of image width horizontally — too short to be an aponeurosis
3. For each valid component: compute per-column median y-coordinate → `np.polyfit` to get the line `y = a*x + b`
4. Take the two largest valid sheets
5. Require that the second sheet is at least 8% of image height away from the first (vertically) — prevents two fragments of the same aponeurosis being mistaken for two separate sheets
6. Sort by mean y-position → superficial (top) and deep (bottom)

Tested on 15 images manually — correctly identifies both sheets in 13/15. The 2 failures are images with very noisy aponeurosis predictions (gaps larger than the connectivity threshold).

### Day 3
Implemented `muscle_thickness_px`:
- Evaluates the vertical gap between superficial and deep fitted lines at every x-column in their overlap region
- Takes the **median** (not mean) gap — robust to the slight non-parallelism of the two lines
- Corrects for aponeurosis tilt: multiplies by `cos(atan(mean_slope))` to get the perpendicular distance, not vertical distance

Small but important detail: if the two aponeurosis lines don't overlap horizontally (detected spans don't intersect), fall back to the full union range. This prevents `NaN` MT in edge cases.

### Day 4–5
Implemented `fascicle_stats`:

For each fascicle connected component with area ≥ 80 px:
1. Fit a line through all foreground pixels using `cv2.fitLine` with `DIST_L2` (least squares, robust to speckle noise)
2. Compute the angle relative to the deep aponeurosis
3. Fold angle to `(-90°, 90°]` range and take abs → Pennation Angle
4. **Plausibility filter**: keep only fascicles where `3° ≤ PA ≤ 60°` — removes horizontal artifacts (acoustic shadows) and near-vertical noise
5. For valid fascicles: intersect the fascicle line with both aponeurosis lines geometrically → Fascicle Length
6. Aggregate: length-weighted median PA and median FL across all valid fascicles

The weighted median (by number of pixels in the component) means large, well-detected fascicles dominate the PA estimate — reduces influence of small noisy fragments.

### Day 6
Composed everything into `calculate_clinical_parameters`:
- Returns a dict: `{pa_deg, fl_mm, mt_mm}` with `NaN` for any measurement that failed
- Converts px → mm using `CFG.PX_PER_MM`
- FL fallback: if the fascicle line doesn't intersect both aponeuroses (aponeurosis detection partial), estimate FL from geometry as `MT / sin(PA)` — this is physically meaningful and much better than returning NaN

Validated on 10 annotated images. MT estimates are consistently within ±0.5 mm of manual measurements. PA estimates are within ±2° for most images. FL is noisier due to fascicle detection fragmentation.

> ⚠️ **Note**: The `PX_PER_MM` value of `8.0` is unverified. Need to measure against the scale bar in the actual images. This directly scales all FL and MT outputs. This is flagged for the integration meeting.

---

## Week 7 · Inference Pipeline & Output

### Day 1
Implemented `hysteresis`:
```
high-confidence pixels (prob > thr)      → keep
low-confidence pixels (prob > thr * 0.6) that touch a high-confidence region → also keep
everything else                          → discard
```
This repairs broken fascicle segments: a 2–3 px gap in a predicted fascicle (low probability in that region) no longer splits the fascicle into two disconnected fragments if both sides are confidently predicted.

Tested: fascicle F1 improved by ~3 percentage points on val set with hysteresis vs. hard thresholding.

### Day 2
Implemented `segment_full_res`:
1. Letterbox image to canvas size
2. Predict probability map (+TTA horizontal flip)
3. Crop away the letterbox padding (using stored `(h, w, nh, nw)` from letterbox)
4. Resize probability back to **original image resolution** (not canvas size)
5. Apply hysteresis (fascicle) or hard threshold (aponeurosis)

Result: masks are at full original resolution, so the geometry engine works at maximum precision.

Received threshold values from Person A: `THR = {'fascicle': 0.45, 'aponeurosis': 0.55}`. Plugged into inference.

### Day 3
Ran the full inference loop over all test images. Logged results:
- MT: 0 NaN (always estimated — aponeurosis model is reliable)
- PA: 12 NaN (~8% failure rate — these are images with very low fascicle contrast)
- FL: 15 NaN (~10% failure rate — includes images where fascicle line doesn't reach both aponeuroses)

NaN filling: replaced NaN values with the **median** of successful predictions for that parameter. Not a hard-coded constant — the median adapts to whatever muscle group is being imaged.

### Day 4
Implemented the visual check overlay (Section 11):
- **Green** overlay: fascicle mask pixels
- **Pink** overlay: aponeurosis mask pixels
- **Cyan dashed lines**: fitted aponeurosis lines used for MT measurement
- Title shows computed PA, FL, MT values for that image

Colour blending: `0.5 × original_pixel + 0.5 × colour` — keeps the underlying ultrasound texture visible while making the mask clearly visible.

### Day 5
Final end-to-end check on 4 held-out test images. Overlay looks correct:
- Fascicle lines track the actual muscle fibres
- Aponeurosis lines are well-fitted to the detected sheets
- The cyan regression lines are parallel and span the full image width
- PA values are in the physiologically plausible range (10°–25°) for the muscle group in this dataset

---

## Week 8 · Code Review & Documentation

### Day 1
Reviewed Person A's model and training code. Observations:
- The differential LR (0.3× for encoder, 1× for decoder) is important — without it, the pretrained ResNet weights get overwritten by aggressive gradient updates on the grayscale ultrasound domain
- The `OneCycleLR` warmup (`pct_start=0.1`) is well-tuned — the first few epochs should be cautious while the decoder initialises

Suggested minor cleanup: the `predict_prob` function could check that the model is in `eval()` mode before inference. Person A confirmed the `@torch.no_grad()` decorator and `model.eval()` call in `evaluate` cover this.

### Day 2
Added comments throughout the geometry engine explaining the clinical rationale:
- Why PA is filtered to 3°–60° (physiological range for skeletal muscle pennation angles)
- Why the per-column median y is used instead of raw pixel centroid (robustness to mask thickness variation)
- Why FL is estimated as `MT / sin(PA)` as a fallback (the geometric relationship between the three parameters is physically exact for idealised parallel fascicles)

### Day 3
Wrote this journal entry and cross-checked all integration points with Person A's journal.

---

## 🔧 Troubleshooting Notes

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `fit_aponeuroses` returns `None` | Only one aponeurosis detected | Lower `min_area` to 150 or check if aponeurosis model threshold is too high |
| MT = 0 or negative | Superficial/deep lines swapped | Check the `sort by mean y` logic — ensure superficial has smaller y (top of image) |
| PA always NaN | Fascicle min_area filter too aggressive | Lower `min_area` to 50 in `fascicle_stats` |
| FL very large (200+ mm) | `PX_PER_MM` is wrong | Measure scale bar in original image; `PX_PER_MM` is the most critical constant |
| Overlay colours wrong | RGB vs grayscale array mismatch | Ensure `rgb = np.stack([gray]*3, -1)` before applying colour overlays |
