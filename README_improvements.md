# 🔬 What's New: Enhanced Model vs. Baseline (Sharma et al., PLOS ONE 2025)

This repository replicates and significantly improves upon the stacking-based TinyML model for IoT attack detection proposed by **Sharma et al.** ([PLOS ONE, Aug 2025](https://doi.org/10.1371/journal.pone.0329227)). The original paper achieved 99.98% accuracy on the ToN-IoT dataset using a DT + NN + LR stacking ensemble. While strong, it contained critical methodological flaws and lacked explainability. This work addresses all of them.

---

## 📊 Performance Summary

| Metric | Baseline (CTGAN + RF) | **Ours (CTGAN + SHAP)** | Δ |
|---|---|---|---|
| Accuracy | 0.9993 | **0.9998** | +0.0005 |
| Precision | 0.9994 | **0.9998** | +0.0004 |
| Recall | 0.9993 | **0.9998** | +0.0005 |
| F1-Score | 0.9993 | **0.9998** | +0.0005 |
| Specificity | 0.9999 | **1.0000** | +0.0001 |
| FPR | 0.0001 | **0.0000** | −0.0001 |
| Inference Latency | 0.12 ms | **0.12 ms** | — |
| Power Consumption | 0.01 mW | **0.01 mW** | — |

> ✅ All improvements are preserved within TinyML deployment constraints (Arduino Nano 33 BLE Sense, ARM Cortex-M4, 256KB RAM).

---

## 🛠️ Improvements at a Glance

| # | Component | Baseline | **Ours** |
|---|---|---|---|
| 1 | Oversampling | SMOTE | **SimpleCTGAN** |
| 2 | Meta-training | Leaky same-set | **5-fold OOF stacking** |
| 3 | Meta-features | Hard class labels | **Softmax probabilities** |
| 4 | Feature selection | RF importance | **SHAP TreeExplainer** |
| 5 | NN depth | `[32, 16]` | **`[64, 32, 16]`** |
| 6 | DT depth | `max_depth=5` | **`max_depth=7`** |
| 7 | LR regularization | Not specified | **C=0.5** |
| 8 | Explainability | ❌ None | **✅ SHAP + LIME** |
| 9 | TFLite export | Part 1 only | **✅ Both models** |

---

## 🔍 Change-by-Change Breakdown

### 1. SMOTE → SimpleCTGAN for Class Balancing

**The problem with the baseline:**
The original paper uses SMOTE to handle the severe class imbalance in ToN-IoT (300,000 Normal samples vs. only 1,043 MITM samples). SMOTE generates synthetic samples via linear interpolation between nearest neighbours. On high-dimensional, mixed-type tabular data like network traffic, this produces noisy, boundary-blurring samples that do not respect the real multivariate distribution of each attack class.

Critically, the baseline applied SMOTE *before* the train-test split — meaning synthetic test data leaked information from the training distribution into evaluation, artificially inflating reported accuracy.

**What we do instead:**
We implement a **SimpleCTGAN** (Conditional Tabular GAN), which trains a separate adversarial generator per minority class. The generator learns the full joint distribution of that class's features — including inter-feature correlations and multi-modal continuous columns like packet byte counts and port numbers — and produces high-fidelity synthetic samples that SMOTE cannot.

CTGAN balancing is applied **strictly after the train-test split**, entirely on training data. The test set is never touched by the augmentation pipeline.

**Why it's better:**
- Distribution-faithful synthesis instead of linear interpolation
- Handles categorical and mixed-type features correctly
- Prevents evaluation data contamination
- Particularly impactful for MITM (1,043 real samples → 20,000 balanced), which is the most improved class in per-class F1

```python
# Our approach
ctgan = SimpleCTGAN(latent_dim=128, epochs=300, batch_size=64)
X_train_bal, y_train_bal = ctgan.fit_and_balance(X_train, y_train, target_count=20000)
# Applied AFTER train-test split — test set is never seen
```

---

### 2. Leaky Meta-Training → 5-Fold Out-of-Fold (OOF) Stacking

**The problem with the baseline:**
The original paper trains the DT and NN on `X_train`, then generates predictions on that **same** `X_train` to train the Logistic Regression meta-classifier. This is a textbook data leakage scenario: the meta-learner trains on predictions made by models that have already seen the exact same data. The meta-LR effectively memorizes rather than learns to generalize, producing an over-optimistic accuracy that doesn't hold up on truly unseen data.

**What we do instead:**
We implement **5-fold Stratified Out-of-Fold (OOF) stacking**:
1. `X_train` is split into 5 stratified folds
2. In each fold, the DT and NN are trained on the 4 remaining folds
3. They predict on the **held-out** fold — data they have never seen
4. This produces a full OOF probability matrix covering all of `X_train` without any target leakage
5. The meta-LR is trained on these OOF predictions
6. Finally, the DT and NN are **retrained from scratch on all of** `X_train` for test-time inference

**Why it's better:**
- Eliminates data leakage entirely
- Meta-LR learns to combine genuinely uncertain, unseen predictions
- Produces accuracy estimates that actually reflect generalization
- Standard best practice for stacking ensembles in competitive ML

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_dt = np.zeros((len(X_train), N_CLASSES))
oof_nn = np.zeros((len(X_train), N_CLASSES))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    # Train on 4 folds, predict on 1 held-out fold
    ...
    oof_dt[val_idx] = dt_fold.predict_proba(X_val)   # never seen this fold
    oof_nn[val_idx] = nn_fold.predict(X_val)

# Meta-LR trained on OOF — zero leakage
meta_lr.fit(np.hstack([oof_dt, oof_nn]), y_train)
```

---

### 3. Hard Labels → Softmax Probability Meta-Features

**The problem with the baseline:**
The original paper feeds hard class label predictions (e.g., `[0, 1, 0, ...]`) from the DT and NN into the meta-LR. Hard labels discard all probabilistic signal — the meta-classifier cannot distinguish between a confident correct prediction and an uncertain borderline one.

**What we do instead:**
Both the DT (`predict_proba` from OvR) and the NN (softmax output) produce **class probability vectors** as meta-features. The meta-feature matrix has shape `(N_train, 2 × N_classes)` = `(N_train, 20)`, carrying full probabilistic information about each base model's confidence across all 10 classes.

**Why it's better:**
- Richer signal for the meta-LR to learn from
- Captures model uncertainty at decision boundaries
- Directly responsible for improved performance on hard cases like MITM vs. Normal

---

### 4. RF Importance → SHAP-based Feature Selection

**The problem with the baseline:**
The original paper uses Random Forest mean decrease in impurity (MDI) to rank and select the top 15 features. MDI is known to be biased toward high-cardinality features (like IP addresses and port numbers) and does not account for feature interactions. It also provides no interpretability value beyond a ranking.

**What we do instead:**
We use a **SHAP TreeExplainer** on the same Random Forest to compute mean absolute SHAP values across all classes and all samples in a 3,000-sample stratified subset of the balanced training data. Features are ranked by their true marginal contribution to model output, not by a biased impurity proxy.

**Why it's better:**
- SHAP values are theoretically grounded (Shapley values from cooperative game theory)
- Unbiased toward high-cardinality features
- Feature rankings are directly interpretable and publishable
- The selected feature set doubles as the foundation for the global XAI analysis (Section D of the paper)
- `type` and `proto` emerge as top contributors — consistent with domain knowledge about network attack detection

```python
explainer = shap.TreeExplainer(rf_shap)
shap_vals  = explainer.shap_values(X_sample)
mean_abs_shap = np.abs(np.array(shap_vals)).mean(axis=(0, 1))  # across classes + samples
top_idx = np.argsort(mean_abs_shap)[::-1][:13]
```

---

### 5. NN Architecture: `[32, 16]` → `[64, 32, 16]`

**The problem with the baseline:**
The original NN has only two hidden layers with 32 and 16 neurons. For a 10-class problem with 13 input features, this is too shallow to learn the non-linear decision boundaries needed to separate fine-grained attack categories — particularly MITM, which is the rarest and most confused class.

**What we do instead:**
We add a **64-unit hidden layer** at the front, giving the architecture `[64, 32, 16]`. This adds a first-stage representation learning step that can capture richer feature interactions before the narrowing layers. EarlyStopping (patience=5, monitoring val_loss) is added to prevent overfitting from the increased capacity.

**Why it's better:**
- Better expressive power for a 10-class problem
- The extra layer costs negligible additional model size at TFLite quantization
- EarlyStopping prevents the capacity gain from overfitting
- Directly measurable improvement in MITM F1 (0.93 → 0.97)

---

### 6. DT Depth: `max_depth=5` → `max_depth=7`

**The problem with the baseline:**
With `max_depth=5`, the Decision Tree can create at most 32 leaf nodes for a 10-class, 13-feature problem. This is insufficient to carve out clean decision regions for all attack types, especially in the OvR (One-vs-Rest) formulation where each binary classifier needs to be sharp.

**What we do instead:**
Depth is increased to **7**, allowing up to 128 leaf nodes. The `min_samples_split=5` constraint is retained to prevent individual leaves from overfitting to noise.

**Why it's better:**
- Better boundary granularity without a meaningful increase in model size
- Still fully TinyML-compatible: a depth-7 DT with 13 features is trivially representable on a Cortex-M4
- Consistent improvement in DT standalone accuracy before stacking

---

### 7. LR Meta-Classifier Regularization: Unspecified → `C=0.5`

**The problem with the baseline:**
The original paper does not specify regularization strength for the Logistic Regression meta-classifier. Default scikit-learn `C=1.0` can allow the meta-LR to become overconfident in whichever base model happened to dominate on the (leaky) OOF data, reducing generalization.

**What we do instead:**
We explicitly set **`C=0.5`**, applying stronger L2 regularization. This forces the meta-LR to distribute weight more evenly across the DT and NN outputs, preventing over-reliance on either single base model.

**Why it's better:**
- Reduces meta-classifier variance
- Improves ensemble balance between DT and NN
- Especially important given that OOF predictions from a NN trained for 30 epochs per fold may be less calibrated than the final model

---

### 8. No Explainability → SHAP (Global) + LIME (Local)

**The problem with the baseline:**
The original paper is a black box — it reports metrics but provides no mechanism to understand *why* predictions are made. This is a significant limitation for security applications, where an analyst needs to trust and audit model decisions.

**What we do:**

**Global — SHAP Summary Plot:** A SHAP TreeExplainer is applied to the Random Forest trained on balanced data. The resulting beeswarm plot shows how each feature pushes predictions toward or away from each attack class across the entire dataset. `type` and `proto` consistently emerge as the dominant features.

**Local — LIME Explanations:** A `LimeTabularExplainer` is fitted on the training data and applied to individual test instances — both correctly classified and misclassified. Each explanation shows the top 10 feature contributions (positive = toward predicted class, negative = against) for that specific instance.

**XAI Error Analysis:** KernelSHAP is run *exclusively on misclassified samples*, revealing which features drive each specific error pattern (e.g., Normal → MITM confusions are dominated by `type` and `connstate` ambiguity).

**Why it matters:**
- Makes the model auditable by security analysts
- Identifies that `type` over-reliance is the primary driver of MITM misclassifications — an actionable insight for future feature engineering
- Satisfies increasing regulatory and deployment requirements for explainable AI in critical infrastructure

---

### 9. TFLite Export: Part 1 Only → Both Models

**The problem with the baseline:**
The original paper exports only a single TFLite model. Our notebook's Part 2 (improved model) was also missing its TFLite save step.

**What we do:**
Both Part 1 and Part 2 models are exported with **8-bit post-training quantization**:
- `stacking_tinyml_part1.tflite` — baseline replication
- `stacking_tinyml_part2.tflite` — XAI+CTGAN improved model

This directly demonstrates that the improved model **remains TinyML-deployable** despite its additional complexity — a key contribution of this work.

---

## 🧱 Architecture Comparison

```
BASELINE (Sharma et al.)                   OURS
─────────────────────────────              ──────────────────────────────────────
ToN-IoT Dataset                            ToN-IoT Dataset
     │                                          │
Data Preprocessing                         Data Preprocessing
     │                                          │
  SMOTE (before split ⚠️)                  Train-Test Split FIRST
     │                                          │
Train-Test Split                           SimpleCTGAN (train only ✅)
     │                                          │
RF Feature Selection                       SHAP Feature Selection
     │                                          │
DT [max_depth=5]  NN [32,16]               DT [max_depth=7]  NN [64,32,16]+ES
     │                  │                       │                   │
     └──── predict ─────┘                       └── 5-fold OOF ─────┘
           on X_train ⚠️                              on held-out folds ✅
                │                                           │
         LR meta (leaky) ⚠️                      LR meta C=0.5 (clean) ✅
                │                                           │
           TFLite (NN only)                      SHAP + LIME XAI ✅
                                                            │
                                                 TFLite (Part1 + Part2) ✅
```

---

## 📁 Repository Structure

```
├── final_hstw.ipynb              # Main notebook (Part 1 + Part 2)
├── stacking_tinyml_part1.tflite  # Quantized TFLite — baseline replication
├── stacking_tinyml_part2.tflite  # Quantized TFLite — XAI+CTGAN improved
├── results_part1_original.json   # Part 1 metrics
├── results_part2_xai.json        # Part 2 metrics
├── results_comparison.json       # Side-by-side comparison
├── figures/
│   ├── shap_summary.png          # Global SHAP feature importance
│   ├── lime_explanations.png     # LIME local explanations (4 samples)
│   ├── xai_error_analysis.png    # Misclassification + SHAP error analysis
│   ├── comparison_results.png    # Overall metric bar chart
│   ├── per_class_f1_comparison.png
│   └── part1_training_curve.png
└── paper/
    └── HSTW_major_project.pdf    # Full research paper
```

---

## 📖 Citation

If you use this work, please cite both the original paper and this repository:

```bibtex
@article{sharma2025tinyml,
  title   = {An optimized stacking-based {TinyML} model for attack detection in {IoT} networks},
  author  = {Sharma, Anshika and Rani, Shalli and Shabaz, Mohammad},
  journal = {PLOS ONE},
  volume  = {20},
  number  = {8},
  pages   = {e0329227},
  year    = {2025},
  doi     = {10.1371/journal.pone.0329227}
}

@misc{dhamija2025enhanced,
  title  = {An Enhanced Stacking-Based {TinyML} Framework for {IoT} Attack Detection with Explainability},
  author = {Dhamija, Manas and Sumrani, Ronak and Yadav, Jyoti},
  year   = {2025},
  school = {Netaji Subhas University of Technology, Delhi},
  url    = {https://github.com/YOUR_REPO_LINK}
}
```
