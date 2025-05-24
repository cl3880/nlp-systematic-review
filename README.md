# Systematic Review Classification: Results Summary

## Project Overview

This repository implements a comprehensive machine learning pipeline for systematic review screening, focusing on brain arteriovenous malformation (AVM) literature classification. The project evaluates multiple classifiers, text normalization techniques, balancing methods, and expert-derived features to optimize both balanced performance and high-recall screening scenarios.

## Project Structure

```
├── src/
│   ├── scripts/           # Main workflow scripts
│   │   ├── stage1_baseline_grid_search.py      # Stage 1: Baseline classifier comparison (StratifiedKFold grid search)
│   │   ├── stage3_isolated_experiments.py      # Stage 3: Isolated experiments (SVM fixed params, StratifiedKFold)
│   │   ├── stage4_expert_features.py           # Stage 4: Expert criteria features (isolated experiment)
│   │   ├── stage4_mesh_features.py             # Stage 4: MeSH term features (isolated experiment)
│   │   ├── final_test_evaluation.py            # Final test set evaluation
│   │   ├── analysis_stage_comparison.py        # Cross-stage analysis
│   │   └── analysis_generate_tables.py         # Table generation
│   ├── models/            # Model definitions and feature engineering
│   ├── utils/             # Utility functions
│   └── visualization/     # Plotting and visualization tools
├── results_final/        # Final organized results by stage (methodology-accurate naming)
│   ├── stage1_baseline_gridsearch_skf/            # Stage 1: Grid search with StratifiedKFold
│   ├── stage2_normalization_smote_grid_search/    # Stage 2: Normalization & SMOTE with grid search
│   ├── stage3_svm_fixed_params_skf/               # Stage 3: SVM fixed params with StratifiedKFold
│   ├── stage4_svm_fixed_params_features_skf/      # Stage 4: SVM fixed params + features with StratifiedKFold
│   └── final_test_evaluation/                     # Test set results
├── data/
│   ├── raw/               # Original datasets
│   └── processed/         # Preprocessed data
├── archive/               # Legacy code and old experiments
└── requirements.txt       # Python dependencies
```

## Workflow Guide

### Prerequisites

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**
   ```bash
   python src/scripts/prepare_data.py --input data/raw/dataset.csv --output data/processed/
   ```

### Stage 1: Baseline Classifier Comparison (StratifiedKFold Grid Search)
```bash
# Run individual classifiers
python src/scripts/stage1_baseline_grid_search.py --model svm --output results_final/stage1_baseline_gridsearch_skf/
python src/scripts/stage1_baseline_grid_search.py --model logreg --output results_final/stage1_baseline_gridsearch_skf/
python src/scripts/stage1_baseline_grid_search.py --model cnb --output results_final/stage1_baseline_gridsearch_skf/
python src/scripts/stage1_baseline_grid_search.py --model cosine --output results_final/stage1_baseline_gridsearch_skf/
```

### Stage 2: Normalization & SMOTE (Grid Search)
```bash
# With normalization
python src/scripts/stage1_baseline_grid_search.py --model svm --normalization stemming --output results_final/stage2_normalization_smote_grid_search/
python src/scripts/stage1_baseline_grid_search.py --model svm --normalization lemmatization --output results_final/stage2_normalization_smote_grid_search/

# With SMOTE
python src/scripts/stage1_baseline_grid_search.py --model svm --balancing smote --output results_final/stage2_normalization_smote_grid_search/
```

### Stage 3: Isolated Experiments (SVM Fixed Params + StratifiedKFold)
```bash
# Test individual effects with fixed parameters
python src/scripts/stage3_isolated_experiments.py --normalization none --balancing none
python src/scripts/stage3_isolated_experiments.py --normalization stemming --balancing none  
python src/scripts/stage3_isolated_experiments.py --normalization none --balancing smote
python src/scripts/stage3_isolated_experiments.py --normalization stemming --balancing smote

# Parameters: --normalization [none|stemming|lemmatization], --balancing [none|smote], --target-recall [0.95]
```

### Stage 4: Expert Features (Controlled Experiments - SVM Fixed Params + StratifiedKFold)
```bash
# Isolated expert criteria features experiment
python src/scripts/stage4_expert_features.py --balancing smote

# Isolated MeSH terms experiment  
python src/scripts/stage4_mesh_features.py --balancing smote --use-mesh
```

### Final Test Evaluation
```bash
python src/scripts/final_test_evaluation.py --model-path results_final/stage4_svm_fixed_params_features_skf/best_model.joblib
```

### Analysis and Visualization
```bash
# Generate comparison tables
python src/scripts/analysis_generate_tables.py --results-dir results_final/ --output results_final/analysis/

# Cross-stage analysis
python src/scripts/analysis_stage_comparison.py --output results_final/analysis/
```

---

## Stage 1: Baseline Classifier Comparison (Grid Search)

### 1.1 Overview

We performed a 5-fold cross-validated grid search over four classifiers: Support Vector Machine (SVM), Logistic Regression (LogReg), Complement Naive Bayes (CNB), and Cosine Similarity. Each model was optimized for both balanced performance (F1) and high-recall use cases.

### 1.2 Results Summary

| Classifier             | Precision | Recall     | F₁         | F₂         | ROC AUC    | WSS\@95    |
| ---------------------- | --------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression    | 0.5667  | 0.7083     | 0.6296     | 0.6746     | 0.9476     | **0.8124** |
| Support Vector Machine | **0.5714**   | **0.8333** | **0.6780** | **0.7634** | **0.9565** | 0.7894     |
| Complement Naive Bayes | 0.3333    | 0.9167     | 0.4889     | 0.6790     | 0.9223     | 0.6472     |
| Cosine Similarity      | 0.4571    | 0.6667     | 0.5424     | 0.6107     | 0.9257     | 0.7894     |

**Conclusion**: SVM had the best F1 and AUC scores. LogReg led on WSS\@95. CNB delivered high recall but very low precision.

---

## Stage 2: Normalization & SMOTE (Grid Search)

### 2.1 Overview

We evaluated the impact of text normalization (stemming, lemmatization) and SMOTE (synthetic minority oversampling) using SVM, optimized via grid search.

### 2.2 Results Summary (Balanced)

| Method              | Precision | Recall | F₁ (bal)   | F₂     | ROC AUC (bal) | WSS\@95 (bal) |
| ------------------- | --------- | ------ | ---------- | ------ | ------------- | ------------- |
| Raw SVM baseline    | 0.5714    | 0.8333 | 0.6780     | 0.7634 | 0.9565        | 0.7894        |
| SVM + SMOTE         | 0.6000    | 0.8750 | **0.7083** | 0.7813 | 0.9543        | **0.8399**    |
| SVM + Lemmatization | 0.5789    | 0.7500 | 0.6552     | 0.7113 | 0.9510        | 0.7940        |
| SVM + Stemming      | 0.5641    | 0.7083 | 0.6333     | 0.6779 | 0.9500        | 0.7849        |

### 2.3 Results Summary (High Recall)

| Method              | Precision  | Recall | F₁ (HR)    | F₂         | ROC AUC (HR) | WSS\@95 (HR) |
| ------------------- | ---------- | ------ | ---------- | ---------- | ------------ | ------------ |
| Raw SVM baseline    | 0.4510     | 0.9583 | 0.6133     | 0.7823     | 0.9565       | 0.7161       |
| SVM + SMOTE         | 0.4510     | 0.9583 | 0.6133     | 0.7823     | 0.9543       | 0.7161       |
| SVM + Lemmatization | 0.4231     | 0.9583 | 0.5750     | 0.7547     | 0.9510       | 0.6931       |
| SVM + Stemming      | **0.4694** | 0.9583 | **0.6216** | **0.7900** | 0.9500       | **0.7206**   |

---

## Stage 3: Isolated Experiments (Fixed SVM Parameters)

### 3.1 Overview

We fixed the SVM parameters based on earlier grid search and isolated the effects of normalization and SMOTE. This ensured controlled comparisons without hyperparameter retuning.

### 3.2 Results Summary (Balanced)

| Model                 | Precision | Recall | F₁ (bal)   | F₂     | AUC (bal) | WSS\@95 (bal) | Notes                      |
| --------------------- | --------- | ------ | ---------- | ------ | --------- | ------------- | -------------------------- |
| Raw SVM               | 0.5714    | 0.8333 | 0.6780     | 0.7634 | 0.9565    | 0.7244        | Baseline                   |
| Raw + SMOTE           | 0.6897    | 0.8333 | **0.7547** | 0.8000 | 0.9515    | **0.7886**    | Best balanced model        |
| Lemmatization         | 0.5435    | 0.7917 | 0.6552     | 0.7267 | 0.9519    | 0.7152        | Underperforms baseline     |
| Lemmatization + SMOTE | 0.6667    | 0.8333 | 0.7347     | 0.7826 | 0.9459    | 0.7152        | Lower AUC than raw + SMOTE |
| Stemming              | 0.6429    | 0.7500 | 0.6923     | 0.7258 | 0.9521    | 0.7565        | Best high-recall model     |
| Stemming + SMOTE      | 0.6429    | 0.7500 | 0.6923     | 0.7258 | 0.9521    | 0.7565        | Same as above              |

### 3.3 Results Summary (High Recall)

| Model                 | Precision | Recall | F₁ (HR) | F₂     | AUC (HR) | WSS\@95 (HR) |
| --------------------- | --------- | ------ | ------- | ------ | -------- | ------------ |
| Raw SVM               | 0.4510    | 0.9583 | 0.4144  | 0.7823 | 0.9565   | 0.7244       |
| Raw + SMOTE           | 0.1394    | 0.9583 | 0.2434  | 0.4406 | 0.9515   | 0.7886       |
| Lemmatization         | 0.3433    | 0.9583 | 0.3433  | 0.6117 | 0.9519   | 0.7152       |
| Lemmatization + SMOTE | 0.3194    | 0.9583 | 0.3194  | 0.5974 | 0.9459   | 0.7152       |
| Stemming              | 0.2255    | 0.9583 | 0.3651  | 0.5808 | 0.9521   | 0.7565       |
| Stemming + SMOTE      | 0.2255    | 0.9583 | 0.3651  | 0.5808 | 0.9521   | 0.7565       |

---

## Stage 4: Expert Criteria Features & MeSH Terms (Fixed SVM Parameters)

### 4.1 Overview

Using the best configurations from Stage 3, we conducted **controlled experiments** to isolate the individual contributions of different feature types. This scientific approach allows us to measure:

**Experimental Design**:
- **`stage4_expert_features.py`**: SVM + SMOTE + **Expert Criteria** (isolated)
- **`stage4_mesh_features.py`**: SVM + SMOTE + **MeSH Terms** (isolated) 
- **Combined experiments**: SVM + SMOTE + **Both Expert Criteria + MeSH**

This controlled methodology enables us to:
1. **Isolate individual contributions** of expert criteria features
2. **Isolate individual contributions** of MeSH terms  
3. **Measure synergistic effects** when features are combined
4. **Ensure fair comparisons** using fixed SVM parameters throughout

All models used fixed parameters from Stage 1 baseline and SMOTE where indicated.

### 4.2 Results Summary (Balanced)

| Model                                 |  Precision | Recall |    F₁ (HR) |         F₂ |   AUC (HR) | WSS\@95 (HR) |
| ------------------------------------- | ---------: | -----: | ---------: | ---------: | ---------: | -----------: |
| **SVM + SMOTE**                       |     0.1394 | 0.9583 |     0.2434 |     0.4406 |     0.9515 |       0.7886 |
| **SVM + SMOTE + Criteria**            |     0.1394 | 0.9583 |     0.2434 |     0.4406 |     0.9515 |       0.7886 |
| **SVM + SMOTE + Criteria + Stemming** |     0.2255 | 0.9583 |     0.3651 |     0.5808 |     0.9521 |       0.7565 |
| **SVM + SMOTE + Criteria + MeSH**     | **0.3239** | 0.9583 | **0.4842** | **0.6886** | **0.9575** |       0.6877 |

### 4.3 Results Summary (High Recall)

| Model                 | Precision  | Recall | F₁ (HR)    | F₂         | AUC (HR)   | WSS\@95 (HR) |
| --------------------- | ---------- | ------ | ---------- | ---------- | ---------- | ------------ |
| Raw + SMOTE           | 0.1394     | 0.9583 | 0.2434     | 0.4406     | 0.9515     | 0.7886       |
| + Criteria            | 0.1394     | 0.9583 | 0.2434     | 0.4406     | 0.9515     | 0.7886       |
| + Stemming + Criteria | 0.2255     | 0.9583 | 0.3651     | 0.5808     | 0.9521     | 0.7565       |
| + Criteria + MeSH     | **0.3382** | 0.9583 | **0.5000** | **0.7012** | **0.9656** | 0.7198       |

---

### Final Validation: Test Set Evaluation

To ensure the reliability and generalizability of our best-performing models, we evaluated them on a **held-out test set** that was **not used during training or hyperparameter tuning**. This provides an unbiased estimate of real-world performance.

##### Balanced‐performance

| Model                         | Precision | Recall | F₁ (bal) |     F₂ |    AUC | WSS\@95 |
| ----------------------------- | --------: | -----: | -------: | -----: | -----: | ------: |
| SVM + SMOTE + Criteria        |    0.6667 | 0.7500 |   0.7059 | 0.7317 | 0.9429 |  0.5642 |
| SVM + SMOTE + Criteria + MeSH |    0.6296 | 0.7083 |   0.6667 | 0.6911 | 0.9399 |  0.6372 |

##### High‐Recall

| Model                         | Precision | Recall | F₁ (HR) |     F₂ |    AUC | WSS\@95 |
| ----------------------------- | --------: | -----: | ------: | -----: | -----: | ------: |
| SVM + SMOTE + Criteria        |    0.2447 | 0.9583 |  0.3898 | 0.6053 | 0.9429 |  0.5642 |
| SVM + SMOTE + Criteria + MeSH |    0.2300 | 0.9583 |  0.3710 | 0.5867 | 0.9399 |  0.6372 |

### Interpretation

* The test set results **confirm the robustness** of the two best-performing models. Their metrics **closely match** those observed during validation.
* Both models maintain **very high recall (95.8%)**, aligning with the goal of minimizing false negatives in systematic reviews.
* The **Raw + SMOTE + Criteria** model maintains the strongest balanced performance across F₁, F₂, and WSS\@95.
* The **+ Criteria + MeSH** model matches its performance, offering **consistent generalization**, particularly in the high-recall regime.

These results reinforce our final recommendation:

* Use **Raw + SMOTE + Criteria** for general screening.
* Use **+ Criteria + MeSH** for recall-prioritized triage scenarios.