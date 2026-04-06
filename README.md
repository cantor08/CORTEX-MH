<p align="center">
  <img src="assets/Cortex_MH_logo_design_concept.png" alt="CORTEX-MH Logo" width="280"/>
</p>

<h1 align="center">CORTEX-MH</h1>
<h3 align="center">Crisis Observation, Risk Triage & Explainability for Mental Health</h3>

<p align="center">
  <a href="https://cantor08.github.io/CORTEX-MH/" target="_blank">
    <img src="https://img.shields.io/badge/Live%20Dashboard-CORTEX--MH-blue?style=for-the-badge&logo=github" alt="Live Dashboard"/>
  </a>
</p>

<p align="center">
An end-to-end AI-powered mental health triage system that processes patient intake text and structured clinical responses to classify mental state, compute C-SSRS-aligned severity scores, and assign priority levels (P0–P3) with full explainability and population-level analytics.
</p>

---

## Live Dashboard

**[cantor08.github.io/CORTEX-MH](https://cantor08.github.io/CORTEX-MH/)**

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Tasks](#tasks)
  - [Task 1 — Domain Gate](#task-1--domain-gate--ood-classifier)
  - [Task 2 — Mental-State Classification](#task-2--mental-state-classification-weak-modality)
  - [Task 3 — Crisis Severity Scoring](#task-3--crisis-severity-scoring-strong-modality)
  - [Task 4 — Risk Triage & Fusion](#task-4--risk-triage--prioritization-fusion)
  - [Task 5 — Population Analytics](#task-5--population-mental-health-stress-intelligence)
- [Key Results](#key-results)
- [Dataset Sources](#dataset-sources)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Design Principles](#design-principles)
- [End-to-End Demo](#end-to-end-demo)
- [Skills Demonstrated](#skills-demonstrated)
- [Disclaimer](#disclaimer)

---

## Overview

CORTEX-MH is a full-stack clinical AI pipeline built for mental health clinicians, crisis counselors, and public health teams. It takes raw patient intake text as input and routes it through five sequential tasks — domain validation, mental-state classification, structured severity scoring, multimodal fusion triage, and population-level stress intelligence — producing actionable, explainable outputs at every stage.

**Triage priority levels:**

| Level | Label | Description |
|-------|-------|-------------|
| P0 | Critical | Actual attempt — immediate intervention |
| P1 | High | Behavior present — urgent review |
| P2 | Moderate | Active ideation — scheduled follow-up |
| P3 | Low | Indicator / supportive — monitoring |

---

## Pipeline Architecture

```
Raw Text Input
      │
      ▼
┌─────────────────┐
│   Task 1        │  Domain Gate / OOD Classifier
│   Safety Layer  │  → accept_reject, domain_confidence, rejection_reason
└────────┬────────┘
         │  (accepted inputs only)
         ▼
┌─────────────────┐
│   Task 2        │  Mental-State Classification  (Weak Modality)
│   BERT Model    │  → p_normal, p_anxiety, p_depression, p_suicidal
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Task 3        │  Crisis Severity Scoring  (Strong Modality)
│   C-SSRS Engine │  → severity_score_rule [0–100], risk_band, R¹⁵ features
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Task 4        │  Gated Multimodal Fusion  ← TEXT + CLINICAL
│   GatedFusion   │  → triage_bucket P0–P3, confidence, gate_weight α
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Task 5        │  Population-Level Analytics
│   MHSI Engine   │  → daily trends, anomaly flags, stress index
└─────────────────┘
```

---

## Tasks

---

### Task 1 — Domain Gate / OOD Classifier

**Status:** ✅ Complete

Rejects inputs that are not mental-health related before they reach downstream models, preventing misleading triage outputs. The safety and governance layer of the pipeline.

**Pipeline:**
1. Text sanitation rules (fast pre-check): min length, max repetition, non-language detection, empty/emoji-only, URL-only
2. OOD classifier: TF-IDF + Logistic Regression (baseline)
3. Threshold selection for high in-domain recall
4. Governance metrics reporting

**Outputs:**
- `accept_reject` ∈ {0, 1}
- `domain_confidence` ∈ [0, 1]
- `rejection_reason` ∈ {OOD, invalid, too_short, profanity-only, ...}

**Data:**
- In-domain: `ourafla/Mental-Health_Text-Classification_Dataset` (HuggingFace)
- Out-of-domain: WikiText (general domain)

**Deliverable:** `domain_gate_model.pkl` + threshold + rejection taxonomy + evaluation report

---

### Task 2 — Mental-State Classification (Weak Modality)

**Status:** ✅ Complete

Classifies text into 4 mental health states and produces a probabilistic linguistic risk signal for downstream fusion.

**Dataset:** `ourafla/Mental-Health_Text-Classification_Dataset`

| Split | Samples |
|-------|---------|
| Train | 44,650 |
| Validation | 4,962 |
| Test | 992 (balanced, 248/class) |

**Classes:** Normal · Anxiety · Depression · Suicidal

**Models trained:**

| Model | Val F1 Macro | Test F1 Macro | Suicidal F1 |
|-------|-------------|---------------|-------------|
| TF-IDF + Logistic Regression (baseline) | 0.772 | 0.709 | 0.708 |
| BERT (bert-base-uncased, 3 epochs) | 0.853 | **~0.886** | **~0.930** |

**BERT training config:** lr=2e-5, batch_size=16, weight_decay=0.01, 89 min on T4 GPU

**Output columns fed to Task 4:**
```
task2_p_normal | task2_p_anxiety | task2_p_depression |
task2_p_suicidal | task2_entropy | task2_maxprob
```

**Saved artifacts:**
- `task2_models/bert_mental_health_classifier/` (~440MB)
- `task2_models/baseline_tfidf_vectorizer.pkl`
- `task2_models/baseline_lr_model.pkl`
- `task2_weak_modality_for_task4_named.csv`

---

### Task 3 — Crisis Severity Scoring (Strong Modality)

**Status:** ✅ Complete

Computes a clinically grounded severity score from structured C-SSRS-style intake signals. Features a dual temporal module separating acute recency from chronic persistence. Serves as the strong modality in Task 4.

**Dataset Source:** Zenodo Record 2667859 — "Reddit C-SSRS Suicide Dataset"
- 500 real Reddit users with human-annotated C-SSRS-aligned labels

**Tier mapping:**
```
Supportive → Tier 0    Indicator → Tier 1    Ideation → Tier 2
Behavior   → Tier 3    Attempt   → Tier 4
```

**R¹⁵ Clinical Feature Vector:**

| Group | Features |
|-------|---------|
| Ideation (6) | I_present, I_severity_level, I_freq, I_duration, I_controllability, I_deterrents |
| Intent/Meaning (1) | I_reasons |
| Behavior (4) | B_preparatory, B_aborted, B_interrupted, B_actual_attempt |
| Aggregated (1) | B_any |
| Temporal (2) | days_since_last_event, duration_since_onset |
| Demographic (1) | age_group |

**Scoring engine (rule-based, strictly monotonic):**

```
Total Score = Ideation (0–35) + Intensity (0–20) + Behavior (0–30) + Temporal (0–15)
              Capped at 100
```

| Component | Formula |
|-----------|---------|
| Ideation | 7 × I_severity_level |
| Intensity | 20 × mean(I_freq, I_dur, I_ctrl, I_det, I_reas − 1) / 4 |
| Behavior | Hierarchical max: Prep=8, Abort=12, Interrupt=18, Attempt=30 |
| Acute Recency | 0–7 days → +8 pts, 8–30 days → +4 pts, >30 → 0 |
| Persistence | 0–7d → 0, 8–30d → +2, 31–90d → +4, >90d → +7 |

**Risk bands:**
```
score < 35          → Low
35 ≤ score < 65     → Medium
65 ≤ score < 85     → High
score ≥ 85 OR attempt → Critical
```

**Model (upscaled to 40k rows):**

| Metric | Value |
|--------|-------|
| MAE | 0.319 |
| R² | 0.9996 |
| Band accuracy | 89.57% |
| Critical F1 | 0.984 |
| Low F1 | 0.954 |

**Saved artifacts:**
- `T3/models/task3_score_regressor.pkl`
- `T3/models/task3_feature_cols.pkl`
- `task3_strong_modality_40k.csv`

---

### Task 4 — Risk Triage & Prioritization (Fusion)

**Status:** ✅ Complete

Fuses the weak text signal (Task 2) with the strong clinical signal (Task 3) through a gated multimodal neural network to produce a final P0–P3 triage bucket with per-sample explainability.

**P0–P3 label source:** Real Zenodo `tier_sim` annotations (zero leakage — completely independent of all 19 training features)

**Label distribution (40,012 samples):**

| Label | Count | % |
|-------|-------|---|
| P0 — Critical | 6,728 | 16.81% |
| P1 — High | 3,947 | 9.86% |
| P2 — Moderate | 8,101 | 20.25% |
| P3 — Low | 21,236 | 53.07% |

**GatedFusion Model Architecture (PyTorch):**

```
text_branch:     FC(6→64) → ReLU → LayerNorm → Dropout(0.3) → FC(64→64) → ReLU → LayerNorm
clinical_branch: FC(13→64) → ReLU → LayerNorm → Dropout(0.3) → FC(64→64) → ReLU → LayerNorm
gate_network:    concat[z_text, z_clinical](128) → FC(128→32) → ReLU → FC(32→1) → Sigmoid → α
fusion:          z_fused = α × z_clinical + (1−α) × z_text
classifier:      Dropout(0.3) → FC(64→32) → ReLU → FC(32→4)
```

Gate weight α:
- α → 1.0: model trusts clinical signal more
- α → 0.0: model trusts text signal more
- Learned dynamically per sample

**Training config:**
- AdamW, lr=3e-4, weight_decay=1e-4
- CosineAnnealingLR, T_max=60
- CrossEntropyLoss with inverse-frequency class weights
- Early stopping patience=10, gradient clipping max_norm=1.0
- Data split: 70/15/15 (Train 28,008 / Val 6,002 / Test 6,002)

**Output artifacts (saved to `CORTEX/T4/`):**
```
t4_gated_fusion.pt            t4_xgb_clinical.pkl
t4_test_predictions.csv       t4_ablation_results.csv
t4_evaluation_report.json     t4_confusion_matrix.png
t4_training_curves.png        t4_calibration.png
t4_ablation.png               t4_gate_weights.png
t4_shap_clinical.png
```

---

### Task 5 — Population Mental-Health Stress Intelligence

**Status:** ✅ Complete

Aggregates per-case triage outputs into population-level stress intelligence: daily trends, severity distribution shifts, anomaly detection, and surge alerts.

**MHSI Formula (Mental Health Stress Index):**

```
MHSI_d = w0 × %P0_d + w1 × %P1_d + w2 × avg_severity_d + w3 × Spike_d
```

Where `Spike_d` = z-score of P0/P1 rate vs trailing window (7 or 14 days)

**Outputs:** Daily/weekly P0–P3 counts · EWMA-smoothed MHSI · Spike flags · Severity distribution shifts · Dashboard-ready tables

**Live dashboard:** [cantor08.github.io/CORTEX-MH](https://cantor08.github.io/CORTEX-MH/)

---

## Key Results

| Task | Model | Key Metric | Result |
|------|-------|-----------|--------|
| Task 2 | BERT | Test F1 Macro | ~0.886 |
| Task 2 | BERT | Suicidal F1 | ~0.930 |
| Task 2 | BERT | Depression F1 | ~0.802 |
| Task 2 | TF-IDF + LR | Test F1 Macro | 0.709 |
| Task 3 | Random Forest | MAE | 0.319 |
| Task 3 | Random Forest | R² | 0.9996 |
| Task 3 | Random Forest | Band Accuracy | 89.57% |
| Task 3 | Random Forest | Critical F1 | 0.984 |

---

## Dataset Sources

| Dataset | Source | Used For |
|---------|--------|---------|
| ourafla Mental Health Text Classification | HuggingFace (`ourafla/Mental-Health_Text-Classification_Dataset`) | Task 2 training |
| Reddit C-SSRS Suicide Dataset | Zenodo Record 2667859 | Task 3 risk anchoring, Task 4 ground truth labels |
| WikiText | HuggingFace / Wikipedia | Task 1 OOD negatives |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, HuggingFace Transformers |
| Classical ML | Scikit-learn, XGBoost |
| NLP | BERT (bert-base-uncased), TF-IDF |
| Explainability | SHAP (TreeExplainer) |
| Data Processing | Pandas, NumPy |
| Evaluation | Scikit-learn metrics, calibration_curve, Brier score |
| Visualization | Matplotlib, Seaborn, Tableau |
| Environment | Google Colab, T4 GPU |
| Storage | Google Drive |
| Serialization | PyTorch (`.pt`), Joblib (`.pkl`) |

---

## Project Structure

```
CORTEX-MH/
│
├── assets/
│   └── Cortex_MH_logo_design_concept.png      # Project logo
│
├── T2/                                         # Task 2 — Mental-State Classification
│   ├── Task_1.ipynb
│   ├── Task_1_output.ipynb
│   ├── task2_models.zip
│   ├── task2_results.zip
│   └── task2_weak_modality_for_task4_named.csv
│
├── T3/                                         # Task 3 — Crisis Severity Scoring
│   ├── t3_training.ipynb
│   ├── task3_strong_modality_40k.csv
│   ├── Dataset/
│   │   ├── task_3_dataset_generation.ipynb
│   │   └── task3_synthetic_cssrs_structured.csv
│   └── models/
│       ├── task3_score_regressor.pkl
│       └── task3_feature_cols.pkl
│
├── T4/                                         # Task 4 — Fusion Triage
│   ├── T4.ipynb
│   ├── t4_gated_fusion.pt
│   ├── t4_test_predictions.csv
│   ├── t4_evaluation_report.json
│   └── *.png
│
├── zenodo_500_with_triage_predictions_20260301_195222.csv
└── README.md
```

---

## Design Principles

1. **No longitudinal tracking** — single intake evaluation only, no user history required
2. **Zero data leakage** — triage labels derived exclusively from real human-annotated Zenodo tiers, never from computed features
3. **Monotonic scoring** — more severe clinical responses never reduce the severity score
4. **Dual temporal modeling** — acute recency (immediacy risk) and chronic persistence (burden) are modeled independently
5. **Modality explainability** — gate weight α quantifies per-sample trust in clinical vs text signal
6. **Two-track deployment** — Track B (40k synthetic) for training robustness, Track A (real C-SSRS questionnaire) for clinical demo
7. **Full artifact pipeline** — every task saves structured outputs (CSV, JSON, PKL, PNG) consumed by the next

---

## End-to-End Demo

**Single intake flow:**

```
User submits text
        │
        ▼
Task 1 → accept/reject + domain_confidence + rejection_reason
        │
        ▼
Task 2 → state_label + p_state[4] + text explanation
        │
        ▼
Task 3 → severity_score [0–100] + risk_band + points breakdown
              (ideation: X | intensity: X | behavior: X | temporal: X)
        │
        ▼
Task 4 → triage_bucket (P0/P1/P2/P3) + confidence + α weight + explanation
```

**Population dashboard:** [cantor08.github.io/CORTEX-MH](https://cantor08.github.io/CORTEX-MH/)

---

## Skills Demonstrated

**Machine Learning & AI**
PyTorch · HuggingFace Transformers · BERT Fine-tuning · XGBoost · Random Forest · Gated Neural Networks · Multimodal Fusion · SHAP · Model Calibration

**NLP**
Text Classification · TF-IDF · Probabilistic Language Modeling · Sequence Classification · Mental State Analysis

**Data Science**
Feature Engineering · Synthetic Data Generation · Conditional Probability Modeling · Anomaly Detection · Time Series Analytics · Data Leakage Prevention

**Clinical Domain Knowledge**
C-SSRS Framework · Suicide Risk Assessment · Crisis Severity Scoring · Mental Health Triage

**MLOps**
Model Serialization · GPU Training · Experiment Reproducibility · Versioned Artifact Pipelines

**Evaluation**
Macro-F1 · Calibration Curves · Brier Score · ECE · Confusion Matrix · Ablation Studies

---

## Disclaimer

CORTEX-MH is a portfolio project demonstrating AI system design for mental health triage. It is **not a certified medical device** and is **not intended for clinical use** without appropriate regulatory approval and human oversight. All outputs should be reviewed by qualified mental health professionals. The system is designed to augment, not replace, clinical judgment.

---

<p align="center">Built by <strong>Maulik Raval</strong> · 2025–2026</p>
<p align="center">
  <a href="https://cantor08.github.io/CORTEX-MH/">Live Dashboard</a>
</p>
