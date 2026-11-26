# Diagram 8 — Work Division & Timeline

```mermaid
flowchart LR
    subgraph A["👤 Person A — Model & Training"]
        A1["Week 1–2\nData audit\nBaseline U-Net\nGroup-wise split"]
        A2["Week 3–4\nResNet34 U-Net\nAttention Gates\nFull-res stream"]
        A3["Week 5–6\nComboLoss\nTraining loop\nAMP · OneCycleLR\nEarly stopping"]
        A4["Week 7\nThreshold tuning\nHold-out test score\nCheckpoint handoff"]
        A5["Week 8\nCode review\nDocstrings"]
        A1 --> A2 --> A3 --> A4 --> A5
    end

    subgraph B["👤 Person B — Data & Geometry"]
        B1["Week 1–2\nFile pairing\nLetterbox resize\nNumpy cache\nsample_window"]
        B2["Week 3–4\nAugmentation pipeline\nSegDataset\nDataLoader"]
        B3["Week 5–6\nGeometry engine\nApo fitting · MT\nFascicle PA · FL"]
        B4["Week 7\nHysteresis\nsegment_full_res\nInference loop\nVisual overlay"]
        B5["Week 8\nCode review\nClinical comments"]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    subgraph SYNC["🤝 Integration Checkpoints"]
        S1["End Week 2\nAgree canvas size\nPX_PER_MM discussion"]
        S2["End Week 4\nArchitecture review\nAugmentation sign-off"]
        S3["End Week 6\nCheckpoints shared\nGeometry validated"]
        S4["Week 7 Day 3\nThreshold handover\nEnd-to-end test"]
        S1 --> S2 --> S3 --> S4
    end

    A2 -.->|canvas size| S1
    B1 -.->|pipeline ready| S1
    A2 -.->|arch review| S2
    B2 -.->|aug review| S2
    A3 -.->|checkpoints| S3
    B3 -.->|geometry| S3
    A4 -.->|THR values| S4
    B4 -.->|inference| S4
```

---

# Diagram 9 — Module Ownership Map

```mermaid
flowchart TD
    NB["colab_muscle_segmentation_IMPROVED2.ipynb"]

    NB --> S1["§1 Setup\nImports · CFG · Device · Seed"]
    NB --> S2["§2 Data — Drive mount\nDataset discovery · ZIP extract"]
    NB --> S3["§3 Data Audit\nPair matching · Stroke width · Overlays"]
    NB --> S4["§4 Group Split\nSubject-aware train/val/test"]
    NB --> S5["§5 Canvas · Cache · Augmentation\nLetterbox · prep_mask · SegDataset · DataLoader"]
    NB --> S6["§6 Model · Loss · Metrics\nResNet34 U-Net · ComboLoss · Dice · tolF1"]
    NB --> S7["§7 Training\nLoop · AMP · Scheduler · Early stop"]
    NB --> S8["§8 Threshold Tuning\nVal sweep · Hold-out test score"]
    NB --> S9["§9 Geometry Engine\nApo fitting · MT · PA · FL"]
    NB --> S10["§10 Inference\nHysteresis · Full-res seg · NaN fill"]
    NB --> S11["§11 Visual Check\nColour overlays · Fitted lines"]

    style S1 fill:#1e2433,stroke:#475569,color:#94a3b8
    style S2 fill:#1a2f50,stroke:#3b82f6,color:#93c5fd
    style S3 fill:#1a2f50,stroke:#3b82f6,color:#93c5fd
    style S4 fill:#1a2f50,stroke:#3b82f6,color:#93c5fd
    style S5 fill:#14532d,stroke:#22c55e,color:#bbf7d0
    style S6 fill:#1a2f50,stroke:#3b82f6,color:#93c5fd
    style S7 fill:#1a2f50,stroke:#3b82f6,color:#93c5fd
    style S8 fill:#1a2f50,stroke:#3b82f6,color:#93c5fd
    style S9 fill:#14532d,stroke:#22c55e,color:#bbf7d0
    style S10 fill:#14532d,stroke:#22c55e,color:#bbf7d0
    style S11 fill:#14532d,stroke:#22c55e,color:#bbf7d0
```

> **Blue sections** → owned by Person A (Model & Training)  
> **Green sections** → owned by Person B (Data & Geometry)
