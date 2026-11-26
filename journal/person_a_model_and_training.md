# 📒 Person A — Model Development & Training Journal

**Role**: Neural network architecture, loss functions, training loop, and evaluation  
**Works on**: Sections 1, 3 (audit), 4 (split), 6 (model/loss/metrics), 7 (training), 8 (threshold tuning)

---

## Week 1–2 · Getting Familiar with the Data

### Day 1
Cloned the project, set up Colab environment, mounted Google Drive. Verified dataset folders are present:
- `fasc_imgs/` and `fasc_masks/`
- `apo_imgs/` and `apo_masks/`
- `test_imgs/`

Ran the data audit (Section 3). First impressions:
- Images vary quite a bit in size — some are close to native ultrasound resolution (~800×600), some are smaller crops
- Fascicle masks are extremely thin — median stroke width came out at around 1–2 px. This is going to be the main challenge.
- Some mask files appear to be stored as 0/1 (not 0/255) — the `binarize_mask` fix is essential, otherwise you get all-zero masks silently

### Day 2
Implemented the group-wise train/val/test split (Section 4). Key concern: neighbouring frames from the same video sweep must NOT straddle the train/val boundary — that would be data leakage and inflate validation scores.

The filename-based grouping worked for most files. For sequences with no clear subject ID in the filename, fell back to blocks of 50 consecutive frames (50 frames per sweep is a reasonable assumption for this kind of data).

Final split ratio landed at approximately 70% / 15% / 15% by group.

### Day 3–4
Built the first baseline — a minimal plain U-Net with standard BCE loss, resizing everything to 512×640.

**Baseline scores (val set):**
| Task | Dice | IoU | tolF1 (±3px) |
|------|------|-----|--------------|
| Fascicle | ~0.18 | ~0.11 | ~0.31 |
| Aponeurosis | ~0.61 | ~0.44 | ~0.72 |

Fascicle performance is terrible, as expected. The 512×640 canvas is destroying the 1–2 px lines. Aponeurosis is passable but far from good.

### Day 5
Shared baseline numbers with Person B. They confirmed the data audit findings. We agreed:
- Canvas must follow native image size (up to 1024 px long side)
- Fascicle masks need dilation during training
- Need a better loss function

**Open questions noted:**
- What is the actual `PX_PER_MM` for our dataset? Need to measure against scale bar. *(Flagged for integration meeting)*
- Are there duplicate images in the dataset? Need to check.

---

## Week 3–4 · Architecture — ResNet34 Attention U-Net

### Day 1–2
Replaced the plain U-Net encoder with a pretrained ResNet34 backbone. Key adaptation: the first conv layer expects 3 channels (RGB) but our images are grayscale. Fixed this by collapsing the 3-channel weights into 1 channel via summation — this preserves the learned feature sensitivity without random re-initialisation.

```
conv1.weight.data = r.conv1.weight.data.sum(1, keepdim=True)
```

Verified output shapes are correct through the full encoder: `(B,64,H/2,W/2)` → `(B,128,H/4,W/4)` → `(B,256,H/8,W/8)` → `(B,512,H/16,W/16)`.

### Day 3
Added Attention Gates on every skip connection (Attention U-Net). Each gate computes a spatial weighting map from the gating signal (upsampled deep feature) and the skip feature — suppresses irrelevant background muscle tissue and focuses attention on the aponeurosis/fascicle structures.

```
AttentionGate(g_ch, x_ch, inter_ch)
  → wg(g) + wx(x) → relu → psi → sigmoid → elementwise multiply x
```

### Day 4
Added the **full-resolution stream** — a small 2-layer ConvBlock that processes the original input at native resolution and is concatenated with the upsampled decoder output at the head. This was the key fix for preserving 1–3 px fascicle lines that the encoder downsampling was losing.

Ran a sanity check: single forward pass on a batch of 2 images at 1024×832 canvas — no shape errors, output probability map looks reasonable visually.

### Day 5
Compared three variants on fascicle val set after 10 quick epochs:

| Architecture | Dice | tolF1 |
|---|---|---|
| Baseline U-Net (512px) | 0.18 | 0.31 |
| ResNet34 U-Net (native res) | 0.34 | 0.49 |
| + Attention Gates | 0.38 | 0.54 |
| + Full-res stream | 0.43 | 0.61 |

Each addition helps. Proceeding with the full architecture.

---

## Week 5–6 · Loss Function & Full Training Loop

### Day 1–2
Implemented **ComboLoss**:
- **Weighted BCE**: `pos_weight = sqrt((1-fg) / fg)` capped at 8× — prevents the pos_weight from exploding on ultra-rare foreground
- **Focal-Tversky**: `alpha=0.3, beta=0.7` (heavy recall bias), `gamma=0.75` (focal focusing)
- **clDice**: soft skeleton extracted via iterative erosion/dilation, 5 iterations. Added only to fascicle loss with `w_cl=0.5`

The clDice term is expensive (5 morphological ops per batch) but noticeable — it rewards predictions that are topologically connected rather than just spatially overlapping. Without it, the fascicle model tends to predict fragmented blobs.

### Day 3
Implemented the full training loop:
- Differential LR: encoder at `0.3 × LR`, decoder at `1.0 × LR` (pretrained encoder shouldn't be bulldozed)
- OneCycleLR scheduler, `pct_start=0.1` (quick warmup, long cosine decay)
- Mixed-precision training (AMP) — roughly 1.6× speedup on T4 GPU
- Gradient clipping at `max_norm=1.0`
- Early stopping: track best val Dice, save checkpoint, stop if no improvement for 10 epochs

### Day 4
First full training run (40 epochs, fascicle model):

```
Ep 01/40 | train loss 1.2341 | val Dice 0.2811 IoU 0.1634 tolF1 0.4201 |  3.1 min
...
Ep 18/40 | train loss 0.4823 | val Dice 0.5234 IoU 0.3547 tolF1 0.6891 |  <- best
...
Ep 28/40 | [early stop] no val-Dice gain for 10 epochs
[+] fascicle: best val Dice 0.5234 -> fascicle_best.pth
```

Aponeurosis model (30 epochs, no clDice):
```
[+] aponeurosis: best val Dice 0.8102 -> aponeurosis_best.pth
```

Aponeurosis performs well. Fascicle is harder but Dice 0.52 is a big improvement over baseline 0.18. The tolF1 at 0.69 suggests the predictions are spatially close even when exact pixel overlap is low — which is what actually matters for the geometry engine.

### Day 5–6
Ran `show_val_predictions` on 6 unseen val images. Observations:
- Aponeurosis masks look clean and continuous
- Fascicle masks show some fragmentation on images with low contrast
- The clDice term is visibly helping — predictions are more "line-like" than the BCE-only baseline
- Some spurious short fragments appear in regions with acoustic shadows — the geometry engine will need to filter these (min_area threshold)

Shared checkpoint files with Person B. They will use them in the inference pipeline.

---

## Week 7 · Threshold Tuning & Evaluation

### Day 1
Swept binarisation thresholds on the validation set from 0.20 to 0.80 in steps of 0.05. Best thresholds found:

| Model | Best Threshold | Val Dice |
|-------|---------------|----------|
| Fascicle | 0.45 | 0.5234 |
| Aponeurosis | 0.55 | 0.8102 |

Higher threshold for aponeurosis makes sense — the model is more confident on thick, clearly visible sheets.

### Day 2
Reported honest hold-out test scores (test split was never touched during training or threshold selection):

| Model | Threshold | Dice | IoU | tolF1 |
|-------|-----------|------|-----|-------|
| Fascicle | 0.45 | 0.5071 | 0.3398 | 0.6744 |
| Aponeurosis | 0.55 | 0.7988 | 0.6651 | 0.8903 |

Scores are consistent with validation — no signs of overfitting.

### Day 3
Handed `THR = {'fascicle': 0.45, 'aponeurosis': 0.55}` to Person B for use in inference.

Final checklist before handoff:
- [x] `fascicle_best.pth` saved to Drive
- [x] `aponeurosis_best.pth` saved to Drive
- [x] Thresholds documented and shared
- [x] Training curves plotted and saved
- [x] Val predictions visually reviewed — no obvious failure modes missed

---

## Week 8 · Code Review & Documentation

### Day 1–2
Reviewed Person B's data pipeline and geometry engine code. Findings:
- The `prep_mask` area-averaging + 0.15 threshold is well thought out — confirmed it preserves lines that the nearest-neighbour approach loses
- The `augment` function correctly applies the same random transform to both image and mask — no misalignment bugs found
- The `fascicle_stats` plausibility filter (3° ≤ PA ≤ 60°) successfully removes most acoustic shadow artifacts in my test

Minor suggestions given:
- Use `np.ptp(xs) < 0.15 * W` as the minimum horizontal span check in `fit_aponeuroses` (already done — good)
- Consider logging how many NaN measurements occur in the final inference loop to track failure rate across datasets

### Day 3
Wrote docstrings for all model classes and training functions. Added a troubleshooting section to this journal.

---

## 🔧 Troubleshooting Notes

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| All-zero fascicle masks | Masks stored as 0/1, threshold was `> 127` | Use `binarize_mask` which checks `arr.max() <= 1` |
| OOM error on Colab | Canvas too large or batch size too high | Reduce `MAX_SIDE` to 768 or `BATCH` to 8 |
| Val Dice stuck at ~0.1 | Foreground pixels too rare, loss dominated by background | Check `pos_weight` in BCE; increase `FASC_DILATE` |
| clDice slows training | 5 skeleton iterations × large batch | Reduce `CL_ITERS` to 3 or use `w_cl=0.3` |
| Early stop too aggressive | Dataset too small, val Dice noisy | Increase `PATIENCE` to 15 |
