# Industrial AI Visual Inspection System

An end-to-end computer vision pipeline for detecting and analyzing
surface defects in industrial images.

Built and evaluated on the KolektorSDD2 dataset.

**Computer Vision • Deep Learning • Image Segmentation • Feature Engineering • Decision Systems • Failure Analysis**

> **Portfolio project:** Built to demonstrate production-oriented AI engineering practices rather than model training alone.

---

## Overview

Industrial visual inspection systems need to do more than detect pixels.

A useful inspection pipeline should be able to:

- identify potential defects,
- convert segmentation outputs into measurable features,
- make an image-level decision,
- evaluate performance on unseen data,
- analyze failure modes,
- and provide evidence for where the system needs improvement.

This project implements that workflow on the **Kolektor Surface-Defect Dataset 2 (KolektorSDD2)**.

The current implementation is a **frozen baseline**. Model and inference behavior were intentionally frozen after validation, held-out testing, ablation analysis, and failure investigation.

---

## Problem

The task is to classify industrial images as:

- **Healthy**
- **Defective**

while also using pixel-level defect segmentation as the foundation for the inspection decision.

The system therefore separates the problem into two stages:

```text
Image
  ↓
Defect Segmentation
  ↓
Segmentation-Derived Features
  ↓
Random Forest Decision Layer
  ↓
Healthy / Defective
```

This separation makes it possible to investigate whether failures originate primarily from segmentation or from the downstream decision layer.

---

## Dataset

**KolektorSDD2**

Dataset statistics used in this project:

| Split | Samples | Healthy | Defective |
|---|---:|---:|---:|
| Train | 1,864 | 1,667 | 197 |
| Validation | 467 | 418 | 49 |
| Held-out Test | 1,004 | 894 | 110 |

The original training data contained a duplicated image/ground-truth pair (`10301 (copy)`), which was identified during dataset integrity auditing and excluded from the working manifest.

The held-out test set remained untouched during model development and threshold selection.

---

## System Architecture

```text
Industrial Image
       │
       ▼
Preprocessing / Resize + Padding
       │
       ▼
Multi-Scale Candidate Localization
       │
       ├── 128×128 patches
       ├── 192×192 patches
       └── 256×256 patches
       │
       ▼
U-Net Segmentation
       │
       ▼
Probability Maps
       │
       ▼
Multi-Scale Max Aggregation
       │
       ▼
Binary Defect Mask
       │
       ▼
Segmentation Features
       │
       ├── Pixel statistics
       ├── Probability statistics
       ├── Connected components
       ├── Bounding-box geometry
       └── Shape / density features
       │
       ▼
Random Forest
       │
       ▼
Healthy / Defective
       │
       ▼
Quantitative Error Analysis
```

---

## Segmentation Pipeline

The frozen segmentation pipeline uses:

- U-Net architecture
- Input size: `192 × 192`
- Binary segmentation threshold: `0.5`
- Patch sizes: `128`, `192`, `256`
- Patch stride: `32`
- CPU inference
- Multi-scale probability aggregation using pixel-wise maximum

Candidate patches are generated using a localization score based on:

```text
0.75 × residual + 0.25 × gradient
```

where the residual component is derived from Gaussian-blur differences and the gradient component uses Sobel features.

---

## Feature Engineering

The segmentation output is converted into image-level features before classification.

The current feature set includes:

- predicted pixel count
- prediction density
- maximum probability
- mean probability
- connected-component count
- largest component area
- largest component bounding-box area
- largest component width
- largest component height
- overall bounding-box width
- overall bounding-box height
- overall bounding-box area
- bounding-box aspect ratio

This creates an explicit bridge between **pixel-level computer vision** and **image-level decision making**.

---

## Decision Layer

A Random Forest classifier is used as the image-level decision layer.

Configuration:

```text
n_estimators = 200
class_weight = balanced
random_state = 42
n_jobs = -1
```

The operating threshold was selected using the validation set and then frozen before evaluating the held-out test set.

Final decision threshold:

```text
RF score ≥ 0.30 → Defective
RF score < 0.30 → Healthy
```

The RF output is treated as an **operating score**, not as a calibrated probability.

---

## Validation Evaluation

At the selected validation operating point:

| Metric | Result |
|---|---:|
| TP | 33 |
| TN | 376 |
| FP | 42 |
| FN | 16 |
| Recall | 67.35% |
| Precision | 44.00% |
| F1 | 53.23% |
| False Positive Rate | 10.05% |

The validation results were used for model/operating-point analysis only. The held-out test set was not used for threshold selection.

---

## Candidate Aggregation Ablation

A controlled validation-only ablation compared the frozen baseline against a `TOP-5` candidate selection strategy.

| Strategy | TP | FN | TN | FP | Defect Recall | Healthy Specificity |
|---|---:|---:|---:|---:|---:|---:|
| All Candidates + Max | 49 | 0 | 0 | 418 | 100.00% | 0.00% |
| Top-5 + Max | 48 | 1 | 16 | 402 | 97.96% | 3.83% |

The `TOP-5` strategy reduced false positives but also introduced a false negative.

The baseline therefore retained **all-candidate max aggregation**.

No additional candidate-count tuning was performed after this controlled comparison.

---

## Final Held-Out Test

The final frozen pipeline was evaluated on the untouched test set.

| Metric | Result |
|---|---:|
| Samples | 1,004 |
| Defective | 110 |
| Healthy | 894 |
| TP | 55 |
| TN | 740 |
| FP | 154 |
| FN | 55 |
| Accuracy | 79.18% |
| Precision | 26.32% |
| Recall | 50.00% |
| F1 | 34.48% |
| False Positive Rate | 17.23% |
| ROC-AUC | 0.7416 |
| Average Precision | 0.3982 |

The headline accuracy should not be interpreted in isolation because the dataset is imbalanced and the inspection system produces a substantial number of false positives.

---

## Failure Analysis

A dedicated audit was performed on **all 209 classification errors** from the held-out test set.

```text
209 total errors
├── 154 False Positives
└── 55 False Negatives
```

Observed error patterns:

### False Positives

All 154 false positives corresponded to:

```text
Ground truth: Healthy
Prediction:   Non-empty defect segmentation
```

This indicates that healthy images can trigger substantial false activation in the segmentation stage.

### False Negatives

The quantitative audit produced heuristic segmentation diagnoses:

```text
40  strong over-segmentation
8   weak segmentation overlap
4   strong segmentation failure
2   reasonable segmentation → inspect decision layer
1   mixed segmentation quality
```

These labels are **diagnostic heuristics**, not formal causal proof.

The evidence nevertheless indicates that segmentation and candidate generation are important bottlenecks in the current baseline.

---

## What the Error Distribution Revealed

The error-distribution audit showed that false positives and false negatives often have surprisingly similar prediction geometry.

Median values:

| Feature | False Positive | False Negative |
|---|---:|---:|
| Prediction Density | 19.59% | 19.01% |
| Component Count | 25 | 24 |
| Largest Component Area | 14,478.5 | 14,014 |
| BBox Width | 230 | 230 |
| BBox Height | 634 | 637 |
| BBox Aspect Ratio | 0.3618 | 0.3606 |

This is important because it shows that a simple threshold on prediction density or component count is unlikely to solve the failure mode by itself.

There are also examples in both directions where the downstream classifier receives misleading segmentation evidence:

- A false negative with relatively good segmentation quality but a low RF score.
- A healthy false positive with zero pixel overlap with the ground truth but a high RF score.

This demonstrates why the project includes both **segmentation analysis** and **decision-layer analysis** rather than treating classification accuracy as the only signal.

---

## Main Bottleneck

The current evidence points toward the following engineering bottleneck:

```text
Candidate Localization
        +
Multi-Scale Segmentation
        +
Max Probability Aggregation
        ↓
False Activations
        ↓
Misleading Segmentation Features
        ↓
Decision-Layer Errors
```

This is a working technical interpretation based on the completed audits, not a claim of formal causal proof.

The current baseline is therefore **not production-ready**.

---

## Engineering Practices Demonstrated

This project focuses on the engineering workflow around an AI system:

- dataset integrity validation
- duplicate detection
- deterministic train/validation splitting
- held-out test evaluation
- modular preprocessing
- multi-scale inference
- feature engineering
- separate decision layer
- controlled ablation testing
- threshold analysis
- quantitative error analysis
- representative visual inspection
- frozen-baseline discipline
- documented limitations

The goal is not simply to report a model metric, but to understand **why the system behaves the way it does**.

---

## Limitations

The current baseline has several important limitations:

1. Segmentation produces substantial false activation on healthy images.
2. Held-out test recall is only 50%.
3. False-positive rate is 17.23%.
4. The RF score is not perfectly calibrated.
5. Candidate aggregation remains a major source of segmentation noise.
6. The current pipeline has not been optimized for deployment latency.
7. No claim of production readiness is made.

These limitations are intentionally documented rather than hidden behind a single aggregate metric.

---

## Future Work

Future iterations can investigate:

### Segmentation

- stronger candidate localization
- improved hard-negative mining
- better multi-scale aggregation
- improved segmentation architectures
- stronger handling of tiny defects
- segmentation-aware post-processing

### Decision Layer

- richer geometric features
- segmentation-quality features
- alternative classifiers
- probability calibration
- cost-sensitive decision policies

### Productionization

```text
Model
  ↓
Inference Service
  ↓
Structured Inspection Result
  ↓
Visualization
  ↓
LLM Explanation
  ↓
AI Agent Workflow
```

These improvements are intentionally deferred until the baseline failure modes are addressed.

---

## Reproducibility

The project uses deterministic seeds where applicable and keeps the held-out test set separate from model development.

Key frozen artifacts include:

```text
outputs/
├── checkpoints/
├── segmentation_feature_audit/
├── train_segmentation_feature_audit/
├── decision_layer/
├── final_test/
├── final_test_error_audit/
└── final_test_audit/
```

The repository also contains the scripts used for dataset auditing, feature extraction, decision-layer training, test evaluation, candidate aggregation analysis, and error analysis.

---

## Project Structure

```text
KolektorSDD2/
│
├── train/
├── test/
├── splits/
│
├── scripts/
│   ├── dataset audits
│   ├── segmentation
│   ├── feature extraction
│   ├── decision layer
│   ├── test evaluation
│   └── error analysis
│
├── outputs/
│   ├── checkpoints/
│   ├── decision_layer/
│   ├── final_test/
│   ├── final_test_audit/
│   └── final_test_error_audit/
│
└── FINAL_TECHNICAL_REPORT.md
```

---

## What This Project Demonstrates

This project demonstrates an end-to-end approach to industrial computer vision:

> **Build → Validate → Freeze → Test → Audit → Understand Failure Modes**

The main engineering result is not simply a classifier.

It is the complete inspection pipeline and the evidence-driven analysis of where that pipeline currently fails.

---

## Final Baseline Status

**Status: Frozen Baseline**

The current model and inference pipeline are intentionally frozen.

Further optimization is deferred to a future iteration rather than continuously tuning against the same validation/test evidence.

The next stage is to use the documented failure modes to design a targeted improvement, rather than performing blind hyperparameter search.