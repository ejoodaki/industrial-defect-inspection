# KolektorSDD2 Industrial Defect Inspection
## Final Technical Report — Frozen Baseline

---

## 1. Project Overview

This project develops an industrial defect inspection pipeline for the KolektorSDD2 dataset.

The system combines:

- Image preprocessing
- U-Net based defect segmentation
- Multi-scale candidate localization
- Segmentation-derived geometric features
- Random Forest decision layer
- Quantitative error analysis
- Validation ablation studies
- Final held-out test evaluation

The objective is not only to detect defective samples, but also to analyze where and why the inspection pipeline fails.

The evaluated pipeline is treated as a frozen baseline. No further model tuning was performed after the final test evaluation.

---

## 2. Dataset

The KolektorSDD2 dataset contains industrial surface images with pixel-level defect annotations.

### Test Set

| Property | Value |
|---|---:|
| Total samples | 1004 |
| Defective samples | 110 |
| Healthy samples | 894 |

The test set was kept separate from model development and was used only for final evaluation.

---

## 3. Data Integrity and Split Validation

Dataset integrity checks were performed before model development.

### Test Set Integrity

- Image/mask pairs: 1004
- Missing pairs: 0
- Unreadable files: 0
- Dimension mismatches: 0
- Invalid masks: 0

### Training Data

The training data contained 2333 image files and 2331 valid image/ground-truth pairs.

A duplicated `10301 (copy)` sample was identified and excluded from the manifest.

Final training manifest:

- Total samples: 2331
- Healthy: 2085
- Defective: 246

### Train / Validation Split

A stratified 80/20 split was created with `random_state=42`.

| Split | Total | Healthy | Defective |
|---|---:|---:|---:|
| Training | 1864 | 1667 | 197 |
| Validation | 467 | 418 | 49 |

No train/validation overlap was detected.

---

## 4. System Architecture

The final frozen baseline follows this pipeline:

```text
Industrial Image
      |
      v
Preprocessing / Resize-Pad
      |
      v
Multi-Scale Candidate Localization
      |
      v
U-Net Segmentation
      |
      v
Probability Map
      |
      v
Binary Defect Mask
      |
      v
Segmentation / Geometry Features
      |
      v
Random Forest Decision Layer
      |
      v
Healthy / Defective
```

The system therefore separates pixel-level localization from image-level classification.

---

## 5. Segmentation Pipeline

The segmentation stage uses a U-Net model with a single-channel input and single-channel output.

The frozen inference configuration uses:

- Model input size: 192 × 192
- U-Net probability threshold: 0.5
- Multi-scale patch sizes: 128, 192, 256
- Candidate stride: 32 pixels
- CPU inference
- Maximum-probability aggregation across selected candidates

Candidate patches are generated using a combination of:

- Gaussian-blur residual information
- Sobel gradient information

Candidate scores are computed using:

```text
0.75 × residual + 0.25 × gradient
```

The frozen baseline evaluates all generated candidates and aggregates their restored probability maps using element-wise maximum aggregation.

---

## 6. Segmentation-Derived Features

The segmentation output is converted into image-level features for the decision layer.

The feature set includes:

- Predicted pixel count
- Prediction density
- Maximum probability
- Mean probability
- Component count
- Largest component area
- Largest component bounding-box area
- Largest component width
- Largest component height
- Overall bounding-box width
- Overall bounding-box height
- Overall bounding-box area
- Bounding-box aspect ratio

These features provide geometric and confidence-related information to the image-level classifier.

---

## 7. Decision Layer

A Random Forest classifier was trained using segmentation-derived features.

Configuration:

```text
n_estimators = 200
class_weight = balanced
random_state = 42
n_jobs = -1
```

The selected frozen operating threshold for the RF score is:

```text
RF threshold = 0.30
```

This threshold was selected using validation data before the final test evaluation.

The test set was not used for threshold tuning.

---

## 8. Validation Evaluation

The validation set contained:

- 467 samples
- 49 defective
- 418 healthy

At the validation operating point:

| Metric | Value |
|---|---:|
| TP | 33 |
| TN | 376 |
| FP | 42 |
| FN | 16 |
| Recall | 67.35% |
| Precision | 44.00% |
| F1 | 53.23% |
| FPR | 10.05% |

The validation threshold audit showed a trade-off between defect recall and false-positive rate.

No single threshold was treated as universally optimal.

---

## 9. Candidate Aggregation Ablation

A validation-only ablation study compared:

1. All generated candidates with maximum aggregation
2. Top-5 candidates with maximum aggregation

### Results

| Mode | TP | FN | TN | FP | Defect Recall |
|---|---:|---:|---:|---:|---:|
| All candidates | 49 | 0 | 0 | 418 | 100.00% |
| Top-5 candidates | 48 | 1 | 16 | 402 | 97.96% |

The Top-5 strategy reduced false positives but introduced a missed defect.

The frozen baseline therefore retained all-candidate aggregation.

This experiment was treated as a documented ablation rather than as a tuning loop.

---

## 10. Final Held-Out Test Evaluation

The final test evaluation was performed on the untouched 1004-sample test set.

### Results

| Metric | Result |
|---|---:|
| Accuracy | 79.18% |
| Precision | 26.32% |
| Recall | 50.00% |
| F1 | 34.48% |
| False Positive Rate | 17.23% |
| ROC-AUC | 0.742 |
| Average Precision | 0.398 |

Confusion matrix:

| | Predicted Healthy | Predicted Defective |
|---|---:|---:|
| Actual Healthy | 740 | 154 |
| Actual Defective | 55 | 55 |

The model detected 55 of 110 defective test samples and missed 55.

Among 894 healthy samples, 154 were incorrectly classified as defective.

---

## 11. Final Error Analysis

A quantitative root-cause audit was performed over all 209 final test errors.

### Error Distribution

| Error Type | Count | Percentage of All Errors |
|---|---:|---:|
| Segmentation false activation on healthy | 154 | 73.7% |
| Strong over-segmentation | 40 | 19.1% |
| Weak segmentation overlap | 8 | 3.8% |
| Strong segmentation failure | 4 | 1.9% |
| Reasonable segmentation; decision-layer inspection needed | 2 | 1.0% |
| Mixed segmentation quality | 1 | 0.5% |

### False Negatives

Among 55 false negatives:

- 40 showed strong over-segmentation
- 8 showed weak segmentation overlap
- 4 showed strong segmentation failure
- 2 had relatively reasonable segmentation and warrant decision-layer inspection
- 1 showed mixed segmentation quality

The majority of false negatives therefore exhibited segmentation-related failure patterns.

### False Positives

All 154 false positives were associated with observed segmentation false activation on healthy samples.

These findings indicate that segmentation quality and candidate generation/aggregation are major limitations of the current baseline.

The diagnosis labels are heuristic audit categories and should not be interpreted as formal causal proof.

---

## 12. Main Failure Modes

The final evaluation identified two dominant failure patterns.

### 12.1 False Activation on Healthy Samples

Healthy images can produce substantial segmentation responses.

This creates false defect regions which are subsequently converted into image-level features and can cause false defective predictions.

### 12.2 Over-Segmentation on Defective Samples

Several defective images produced segmentation masks substantially larger or noisier than the annotated defect regions.

This can distort the geometric features supplied to the Random Forest decision layer and contribute to false-negative decisions.

---

## 13. Interpretation

The frozen baseline demonstrates that a complete defect-inspection pipeline can be constructed by combining segmentation, geometric feature extraction, and classical machine-learning decision modeling.

However, the final held-out test results show that the current system is not production-ready for reliable industrial deployment.

The principal evidence from the error audit points toward segmentation and candidate-generation behavior as the dominant technical bottleneck.

The Random Forest decision layer also has identifiable failure cases, but the current audit does not support treating it as the primary source of error.

---

## 14. Limitations

Important limitations of the current baseline include:

- Limited defect recall on the held-out test set
- High false-positive rate for healthy samples
- Segmentation false activations
- Over-segmentation
- Generalization gap between validation and test performance
- CPU-oriented inference during experimentation
- Heuristic candidate generation
- Heuristic error-diagnosis categories

The reported results should therefore be considered a research/engineering baseline rather than a production inspection system.

---

## 15. Final Baseline Status

The following components are considered complete:

- Dataset integrity audit
- Dataset manifest generation
- Train/validation split validation
- Segmentation baseline
- Feature engineering
- Random Forest decision layer
- Validation threshold analysis
- Error analysis
- Error distribution analysis
- Candidate aggregation ablation
- Final held-out test evaluation
- Root-cause-oriented error audit

The model and inference pipeline are now frozen.

Further optimization is intentionally deferred to a future model iteration rather than continuing to modify the current baseline.

---

## 16. Future Improvement Direction

Future iterations should focus primarily on improving the segmentation stage and candidate localization mechanism.

Potential directions include:

- Improved defect localization
- Better handling of small defects
- Reduction of healthy-image false activations
- Improved segmentation loss design
- Better spatial aggregation strategies
- Stronger augmentation
- Improved model architecture
- More robust candidate generation
- End-to-end evaluation of segmentation and decision performance

These are future development directions and were not evaluated in the current frozen baseline.

---

## 17. Conclusion

The project successfully established and evaluated an end-to-end industrial defect-inspection baseline on KolektorSDD2.

The final held-out test result was:

```text
Accuracy:   79.18%
Precision:  26.32%
Recall:     50.00%
F1:         34.48%
ROC-AUC:    0.742
```

The most important engineering outcome is not the metric alone, but the identification of the dominant failure modes through controlled validation experiments and full test-set error analysis.

The current baseline is frozen and provides a measurable starting point for future segmentation-focused improvements.