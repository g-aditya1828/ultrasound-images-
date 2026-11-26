# 📒 Shared Integration Log

**Purpose**: Track decisions made together, resolved blockers, integration checkpoints, and anything that affects both workstreams.

---

## Week 1 — Project Kickoff

### Meeting 1 · Day 1
**Attendees**: Person A, Person B  
**Agenda**: Project scope, division of work, shared tooling

**Decisions made:**
- Person A owns: model architecture, loss functions, training, evaluation, threshold tuning
- Person B owns: data loading, augmentation, geometry engine, inference pipeline, visual output
- Both share responsibility for: data audit, the split logic, final integration testing

**Shared conventions agreed:**
- Dataset lives at `MyDrive/ml data/` on Google Drive
- Model checkpoints saved to `MyDrive/ml data/fascicle_best.pth` and `aponeurosis_best.pth`
- Variable naming: `fasc_*` for fascicle pipeline, `apo_*` for aponeurosis pipeline — consistent throughout both notebooks

**Action items:**
- [ ] Person A: run data audit, share stroke width and foreground fraction numbers
- [ ] Person B: implement file pairing and letterbox, share any size/format findings
- [ ] Both: measure scale bar in 3–5 representative images to agree on `PX_PER_MM`

---

### Meeting 2 · Day 5
**Agenda**: Share audit findings, agree on canvas size and dilation settings

**Shared findings:**
- Median fascicle stroke width: **~1.8 px** at native resolution
- Foreground fraction (fascicle): **~0.004** (very sparse)
- Some images stored as 0/1 masks, some as 0/255 — `binarize_mask` fix confirmed necessary
- 3 images have mask–image size mismatch — handled by resizing mask before caching

**Decisions:**
- `MAX_SIDE = 1024` — canvas follows native image size, capped here
- `FASC_DILATE = 2` — dilation by 2 px (using 3×3 ellipse) makes fascicle targets learnable
- `APO_DILATE = 0` — aponeurosis sheets are already thick enough
- `PX_PER_MM` — ⚠️ **UNRESOLVED** — need to measure scale bar (see below)

**PX_PER_MM discussion:**
- Person B measured 3 images: scale bars read 10 mm across approximately 76–84 px → range is 7.6–8.4 px/mm
- Person A suggested using 8.0 as the working value, but flagging this prominently
- **Resolution**: Use `CFG.PX_PER_MM = 8.0` for now. ⚠️ This MUST be verified before treating FL/MT numbers as clinically meaningful. Any systematic error here scales all length outputs proportionally.

---

## Week 2 — Baseline & Pipeline Skeleton

### Meeting 3 · End of Week 2
**Agenda**: Baseline model scores, pipeline integration check

**Person A reports:**
- Baseline Dice (fascicle): 0.18 — terrible, as expected from fixed 512×640 canvas
- Baseline Dice (aponeurosis): 0.61 — passable baseline
- Moving to ResNet34 + native resolution next week

**Person B reports:**
- File pairing, letterbox, and cache all working
- `prep_mask` with area-averaging + 0.15 threshold confirmed to preserve thin lines
- `sample_window` biasing confirmed: 80% foreground-centred crops vs 20% random

**Integration check:**
- Canvas sizes agreed: `task_canvas()` will be called on the paired files and stored in the shared `CANVAS` dict
- Both workstreams must reference `CANVAS['fascicle']` and `CANVAS['aponeurosis']` from the same source — no hardcoded sizes anywhere

**Open items:**
- [ ] Agree on exact `CROP` and `WIN` sizes once canvas is computed
- [ ] Person B: implement `SegDataset` and `DataLoader` before Person A needs them for training

---

## Week 4 — Architecture Locked, Augmentation Ready

### Meeting 4 · End of Week 4
**Agenda**: Architecture review, augmentation sign-off, geometry engine first draft

**Person A presents:** ResNet34 Attention U-Net — full architecture review

Both reviewed the code for `ResNetUNet`. Key things verified:
- Grayscale weight conversion is correct (`sum(1, keepdim=True)` not `mean`)
- Attention gates attend to skip connections, not bottleneck — this is the correct Attention U-Net formulation
- The full-resolution stream (`self.hr`) processes the **original input**, not an intermediate feature — important for preserving native-resolution thin lines
- Output is a single logit channel (not probability) — sigmoid applied only during loss and inference

**Person B presents:** Augmentation pipeline demo

Showed 10 augmented image+mask pairs. Both confirmed:
- Affine transforms apply the same random matrix to image (bilinear) and mask (nearest-neighbour)
- Noise/blur/gamma applied only to image, not mask — correct
- No misalignment visible even at max rotation (8°)

**Person B presents:** Geometry engine draft

Reviewed `fit_aponeuroses` and `fascicle_stats` logic. Discussion points:
- The 15% minimum horizontal span filter (`np.ptp(xs) < 0.15 * W`) is good — prevents spurious vertical blobs from being treated as aponeuroses
- The 8% vertical separation requirement between sheets is a bit aggressive — might miss muscles with very thin cross-sections. Person B will make this configurable.
- PA filter range (3°–60°) debated — some muscles have higher pennation angles at high contraction levels. Agreed to keep 60° upper limit for now but document it as a parameter to revisit.

**Decisions:**
- `CROP = 384`, `WIN = 448` — confirmed for all datasets
- `min_area` for aponeurosis components: 300 px
- `min_area` for fascicle components: 80 px

---

## Week 6 — Training Complete, Geometry Validated

### Meeting 5 · End of Week 6
**Agenda**: Training results, geometry validation, integration of checkpoint → inference

**Person A reports final training scores:**

| Model | Val Dice | Val IoU | Val tolF1 |
|-------|----------|---------|-----------|
| Fascicle | 0.5234 | 0.3547 | 0.6891 |
| Aponeurosis | 0.8102 | 0.6814 | 0.8923 |

Both agreed these are reasonable given the difficulty of fascicle detection. The tolF1 score is more meaningful for this task than Dice — a 1 px positional offset on a 1 px wide line gives Dice ≈ 0 but tolF1 ≈ 1.

**Person B reports geometry validation:**
- MT estimates within ±0.5 mm of manual measurement on 10 test images
- PA within ±2° for most images
- FL noisier — dependent on fascicle detection quality which is the harder problem
- PX_PER_MM still unverified at clinical-grade precision

**Integration task completed:**
- Person B confirmed they can load `fascicle_best.pth` and `aponeurosis_best.pth` from Drive successfully
- `segment_full_res` tested end-to-end: input raw image → output full-resolution binary mask ✅
- Geometry engine runs on the output masks and produces valid PA/FL/MT values ✅

**Remaining open issue:**
- Threshold values not yet finalised — Person A will tune next week and hand over

---

## Week 7 — Final Integration & End-to-End Test

### Meeting 6 · Day 3
**Agenda**: Threshold handover, full end-to-end run, failure analysis

**Threshold values finalised (Person A → Person B):**
```python
THR = {
    'fascicle':    0.45,
    'aponeurosis': 0.55
}
```

**End-to-end test run (Person B):**
- Ran inference on all test images
- Failure rates: PA ~8%, FL ~10%, MT ~0%
- NaN filling applied using within-batch medians
- Visual overlays reviewed on 4 sample images — all look correct

**Joint review of visual check output:**
- Fascicle overlay (green) correctly traces the diagonal muscle fibre bands
- Aponeurosis overlay (pink) covers both sheets cleanly
- Cyan dashed fitted lines are parallel and plausibly positioned
- Computed PA, FL, MT values displayed in image title match what you'd estimate by eye

**Final checklist — both sign off:**

| Item | Person A | Person B |
|------|----------|----------|
| Fascicle checkpoint saved to Drive | ✅ | ✅ verified loadable |
| Aponeurosis checkpoint saved to Drive | ✅ | ✅ verified loadable |
| Thresholds documented | ✅ | ✅ applied in inference |
| Training curves saved | ✅ | — |
| Data audit findings documented | ✅ | ✅ |
| Geometry validated on annotated images | — | ✅ |
| Visual check on test images | — | ✅ |
| NaN handling documented | — | ✅ |
| `PX_PER_MM` warning in README | ✅ | ✅ |

---

## Week 8 — Code Review

### Meeting 7 · Final Review
**Agenda**: Cross-review of each other's code, documentation, final cleanup

**Person A reviewed Person B's code:**
- `prep_mask`: area-averaging + 0.15 threshold logic is sound and well-commented ✅
- `augment`: confirmed identical transforms applied to image and mask ✅
- `fit_aponeuroses`: 15% horizontal span filter and 8% vertical separation checks are good safeguards ✅
- `fascicle_stats`: length-weighted median aggregation is a good robust estimator ✅
- Minor: suggested adding a log line counting how many fascicle components were detected per image — useful for debugging low-PA images

**Person B reviewed Person A's code:**
- `ComboLoss`: pos_weight cap at 8× prevents training instability on very sparse masks ✅
- `ResNetUNet.forward`: confirmed the full-resolution stream `self.hr(x)` uses the original `x`, not any intermediate feature ✅
- Differential LR: encoder group correctly identified by layer name prefix ✅
- Minor: noted that `torch.load` with `map_location=device` already handles GPU → CPU loading correctly ✅

**Agreed final cleanup tasks:**
- [ ] Remove all print statements that were temporary debugging
- [ ] Add `assert` checks at the top of `calculate_clinical_parameters` to catch mis-shaped inputs early
- [ ] Add a `requirements.txt` / environment note to README
- [ ] Archive intermediate experiment checkpoint files — keep only the best

---

## 📌 Standing Decisions & Conventions

| Decision | Value | Rationale |
|----------|-------|-----------|
| `PX_PER_MM` | 8.0 (⚠️ unverified) | Estimated from scale bar; must be confirmed per dataset |
| `MAX_SIDE` | 1024 | Preserves native resolution without exceeding Colab RAM |
| `CROP` | 384 | Large enough to include multiple fascicles; small enough to fit batch=12 |
| `FASC_DILATE` | 2 | Fascicle stroke median ~1.8 px; dilation to ~5 px makes target learnable |
| Threshold (fascicle) | 0.45 | Tuned on val set by Person A |
| Threshold (aponeurosis) | 0.55 | Tuned on val set by Person A |
| NaN fill strategy | Median of valid predictions | Adaptive per-dataset; not a fixed constant |
| Encoder LR multiplier | 0.3× | Pretrained weights should update slowly |
| Early stopping patience | 10 epochs | Generous enough for the OneCycleLR schedule to work |

---

## 🔗 Key File Locations

```
Google Drive / MyDrive / ml data /
├── fasc_imgs/                  ← Fascicle training images
├── fasc_masks/                 ← Fascicle binary masks  
├── apo_imgs/                   ← Aponeurosis training images
├── apo_masks/                  ← Aponeurosis binary masks
├── test_imgs/                  ← Images for inference
├── fascicle_best.pth           ← Best fascicle model checkpoint
└── aponeurosis_best.pth        ← Best aponeurosis model checkpoint
```
