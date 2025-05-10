# Baseline

| Classifier             | Precision  | Recall     | F₁         | F₂         | ROC AUC    | WSS\@95    |
| ---------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression    | 0.5667     | 0.7083     | 0.6296     | 0.6746     | 0.9476     | **0.8124** |
| Support Vector Machine | **0.5714** | 0.8333     | **0.6780** | **0.7634** | **0.9565** | 0.7894     |
| Complement Naive Bayes | 0.3333     | **0.9167** | 0.4889     | 0.6790     | 0.9223     | 0.6472     |
| Cosine Similarity      | 0.4571     | 0.6667     | 0.5424     | 0.6107     | 0.9257     | 0.7894     |

For detailed analysis, see the full grid search results in the cv_results.csv file.

# Logistic Regression Model Comparison

## Model Configurations

### Balanced Model (Optimized for F1)
- clf__C: 100
- clf__class_weight: balanced
- clf__penalty: l2
- clf__solver: liblinear
- tfidf__max_df: 0.85
- tfidf__max_features: 20000
- tfidf__min_df: 1
- tfidf__ngram_range: (1, 3)

### High-Recall Model (95% Target)
- optimal_threshold: 0.11899092942874724

## Performance Comparison

| Metric | Balanced Model | High-Recall Model |
|--------|---------------|-------------------|
| precision | 0.5667 | 0.4182 |
| recall | 0.7083 | 0.9583 |
| f1 | 0.6296 | 0.5823 |
| f2 | 0.6746 | 0.7616 |
| roc_auc | 0.9476 | 0.9476 |
| wss at 95 | 0.8124 | 0.6977 |

### N-gram Range Analysis

Based on our cross-validation results (see cv_results.csv), we found:

- **(1, 2)** n-grams: average F1 = 0.5222
- **(1, 3)** n-grams: average F1 = 0.5200

The (1,2) n-gram range outperforms (1,3) by **0.2 percentage points** - a modest improvement that should be considered alongside other hyperparameters.

### *ROC Curve*
![b44ad3c4c5b5fa542351556663d64c4f.png](:/980548370cd749f4a44caf3121f76353)

### *Confusion Matrix*
| Balanced Threshold | High-Recall Threshold |
|:------------------:|:---------------------:|
| ![CM balanced](:/0d55cbb0bb914acfb10c858b2a761df0) | ![CM high-recall](:/bdb4d2fef3df4ac28415cda65416e2b7) |

### *PR Curve*
![7f36ad84efe67bcb323692c75e7da42a.png](:/bcd02057da2a4a8c87dd84c79b5f4d3c)

---

# Support Vector Machine Model Comparison

## Model Configurations

### Balanced Model (Optimized for F1)
- clf__C: 1
- clf__class_weight: balanced
- clf__kernel: linear
- tfidf__max_df: 0.85
- tfidf__max_features: 5000
- tfidf__min_df: 1
- tfidf__ngram_range: (1, 2)

### High-Recall Model (95% Target)
- optimal_threshold: 0.15580591415876371

## Performance Comparison

| Metric | Balanced Model | High-Recall Model |
|--------|---------------|-------------------|
| precision | 0.5714 | 0.4510 |
| recall | 0.8333 | 0.9583 |
| f1 | 0.6780 | 0.6133 |
| f2 | 0.7634 | 0.7823 |
| roc_auc | 0.9565 | 0.9565 |
| wss at 95 | 0.7894 | 0.7161 |

### N-gram Range Analysis

Based on our cross-validation results (see cv_results.csv), we found:

- **(1, 2)** n-grams: average F1 = 0.5300
- **(1, 3)** n-grams: average F1 = 0.5292

The (1,2) n-gram range outperforms (1,3) by **0.1 percentage points** - a modest improvement that should be considered alongside other hyperparameters.

For detailed analysis, see the full grid search results in the cv_results.csv file.

### *ROC Curve*
![581e74f225fbc163885351e86fb47705.png](:/8edf58e79ec44e1bb48596506eae8857)

### *Confusion Matrix*
| Balanced Threshold | High-Recall Threshold |
|:------------------:|:---------------------:|
| ![CM balanced](:/ffad4b8053e0498b835df0ee0c1dbba2) | ![CM high-recall](:/70ed5e290fc6409c8a9bb172958e93bb) |

### *PR Curve*
![beba72dee7988a81090483e71b6ad9bf.png](:/0d58d3b011b64e83a10d553ad24222a5)

---

# Complement Naive Bayes Model Comparison

## Model Configurations

### Balanced Model (Optimized for F1)
- clf__alpha: 0.1
- clf__norm: True
- tfidf__max_df: 0.9
- tfidf__max_features: 10000
- tfidf__min_df: 2
- tfidf__ngram_range: (1, 2)

### High-Recall Model (95% Target)
- optimal_threshold: 0.49999564235682203

## Performance Comparison

| Metric | Balanced Model | High-Recall Model |
|--------|---------------|-------------------|
| precision | 0.3333 | 0.2323 |
| recall | 0.9167 | 0.9583 |
| f1 | 0.4889 | 0.3740 |
| f2 | 0.6790 | 0.5897 |
| roc_auc | 0.9223 | 0.9223 |
| wss at 95 | 0.6472 | 0.4959 |

### N-gram Range Analysis

Based on our cross-validation results (see cv_results.csv), we found:

- **(1, 3)** n-grams: average F1 = 0.2656
- **(1, 2)** n-grams: average F1 = 0.2562

The (1,3) n-gram range outperforms (1,2) by **0.9 percentage points** - a modest improvement that should be considered alongside other hyperparameters.

For detailed analysis, see the full grid search results in the cv_results.csv file.

### *ROC Curve*
![8fb8bb3318e07fe4e630ef66d71d3e83.png](:/fe1870032c12494eb58d46fc0f4337e4)

### *Confusion Matrix*
| Balanced Threshold | High-Recall Threshold |
|:------------------:|:---------------------:|
| ![CM balanced](:/5559f5ed3f7b4b838ced6d84c49d3fbc) | ![CM high-recall](:/a74452a518514e4aad70db804e2269d0) |

### *PR Curve*
![69c997bfe713081ffff9a0d67fcd5789.png](:/256eec334d054019b1d862ddef9376e8)

---

# Cosine Similarity Model Comparison

## Model Configurations

### Balanced Model (Optimized for F1)
- clf__threshold: 0.25
- tfidf__max_df: 0.85
- tfidf__max_features: 20000
- tfidf__min_df: 3
- tfidf__ngram_range: (1, 3)

### High-Recall Model (95% Target)
- optimal_threshold: 0.2116999669480205

## Performance Comparison

| Metric | Balanced Model | High-Recall Model |
|--------|---------------|-------------------|
| precision | 0.4571 | 0.3485 |
| recall | 0.6667 | 0.9583 |
| f1 | 0.5424 | 0.5111 |
| f2 | 0.6107 | 0.7099 |
| roc_auc | 0.9257 | 0.9257 |
| wss at 95 | 0.7894 | 0.6472 |

### N-gram Range Analysis

Based on our cross-validation results (see cv_results.csv), we found:

- **(1, 2)** n-grams: average F1 = 0.2962
- **(1, 3)** n-grams: average F1 = 0.2832

The (1,2) n-gram range outperforms (1,3) by **1.3 percentage points** - a modest improvement that should be considered alongside other hyperparameters.

For detailed analysis, see the full grid search results in the cv_results.csv file.

### *ROC Curve*
![2a4c2fc2e654ac49a11a9c1f283c0a19.png](:/22324daebf69456991c4e8a95538edc8)

### *Confusion Matrix*
| Balanced Threshold | High-Recall Threshold |
|:------------------:|:---------------------:|
| ![CM balanced](:/f671c2ca16154746a5ae4124e1fd543d) | ![CM high-recall](:/20c5ce450f934fb98b34197ab5fe649f) |

### *PR Curve*
![17734e269336b6ef54216e4ffe43c984.png](:/62e3916fdc74409dae4e78ca382e31a6)

---
##### *Console Logs*
![338781a13fed3e5dc59f96a271f4f342.png](:/52867ff5884a48b88ddd24b623b174ae)

---
=========================================
---

# Support Vector Machine Model Comparison

## Lemmatization

### Balanced Model (Optimized for F1)
- clf__C: 1
- clf__class_weight: balanced
- clf__kernel: linear
- tfidf__max_df: 0.85
- tfidf__max_features: 5000
- tfidf__min_df: 1
- tfidf__ngram_range: (1, 3)

### High-Recall Model (95% Target)
- optimal_threshold: 0.1269127167245752

## Performance Comparison

| Metric | Balanced Model | High-Recall Model |
|--------|---------------|-------------------|
| precision | 0.5588 | 0.4107 |
| recall | 0.7917 | 0.9583 |
| f1 | 0.6552 | 0.5750 |
| f2 | 0.7308 | 0.7566 |
| roc_auc | 0.9510 | 0.9510 |
| wss at 95 | 0.7940 | 0.6931 |

### N-gram Range Analysis

Based on our cross-validation results (see cv_results.csv), we found:

- **(1, 2)** n-grams: average F1 = 0.5423
- **(1, 3)** n-grams: average F1 = 0.5377

The (1,2) n-gram range outperforms (1,3) by **0.5 percentage points** - a modest improvement that should be considered alongside other hyperparameters.

### *ROC Curve*
![3d5c9c633f654c02db7b74ee50a08f71.png](:/bf1d737e94ed4fe3bb0e34a3624923e9)

### *Confusion Matrix*
| Balanced Threshold | High-Recall Threshold |
|:------------------:|:---------------------:|
| ![CM balanced](:/bd13aef447c345ee87a11a753353dde8) | ![CM high-recall](:/9bfe20ce5983469aa12faac4b435fcdb) |

### *Precision-Recall Curve*
![f1c94c12131eff5189ea9ab54a7d442e.png](:/777c73b99c694b25a0005a90190fa2ac)

## Stemming

### Balanced Model (Optimized for F1)
- clf__C: 1
- clf__class_weight: balanced
- clf__kernel: linear
- tfidf__max_df: 0.9
- tfidf__max_features: 5000
- tfidf__min_df: 3
- tfidf__ngram_range: (1, 3)

### High-Recall Model (95% Target)
- optimal_threshold: 0.1470133749026906

## Performance Comparison

| Metric | Balanced Model | High-Recall Model |
|--------|---------------|-------------------|
| precision | 0.5278 | 0.4600 |
| recall | 0.7917 | 0.9583 |
| f1 | 0.6333 | 0.6216 |
| f2 | 0.7197 | 0.7877 |
| roc_auc | 0.9500 | 0.9500 |
| wss at 95 | 0.7849 | 0.7206 |

## N-gram Range Analysis

Based on our cross-validation results (see cv_results.csv), we found:

- **(1, 2)** n-grams: average F1 = 0.5331
- **(1, 3)** n-grams: average F1 = 0.5294

The (1,2) n-gram range outperforms (1,3) by **0.4 percentage points** - a modest improvement that should be considered alongside other hyperparameters.

### *ROC Curve*
![3d5c9c633f654c02db7b74ee50a08f71.png](:/bf1d737e94ed4fe3bb0e34a3624923e9)

### *Confusion Matrix*
| Balanced Threshold | High-Recall Threshold |
|:------------------:|:---------------------:|
| ![CM balanced](:/033a19be422f45ff9b3e10add4176e36) | ![CM high-recall](:/5a4dc9ba1e06435b8c551538619e0532)) |

### *Precision-Recall Curve*
![221807df042d7a3c7d3abd370c397991.png](:/e6eaea02473c4e1f94dc2d4d61d3e86f)

#### ChatGPT Analysis:
| Normalization       | Precision (bal) | Recall (bal) | F₁ (bal)     | ROC AUC (bal) | Precision (hr) | Recall (hr) | F₁ (hr)      | ROC AUC (hr) |
| ------------------- | --------------- | ------------ | ------------ | ------------- | -------------- | ----------- | ------------ | ------------ |
| **None (baseline)** | 0.5714          | 0.8333       | ***0.6780*** | ***0.9565***  | 0.4510         | 0.9583      | 0.6133       | 0.9565       |
| **Lemmatization**   | 0.5588          | 0.7917       | 0.6552       | 0.9510        | 0.4107         | 0.9583      | 0.5750       | 0.9510       |
| **Stemming**        | 0.5278          | 0.7917       | 0.6333       | 0.9500        | ***0.4600***   | 0.9583      | ***0.6216*** | 0.9500       |

– **Baseline (no norm.)** gives you the highest balanced performance (F₁ = 0.678) and the best ROC AUC (0.9565).
– **Lemmatization** slightly lowers both balanced precision/recall and AUC; at high-recall it underperforms baseline on every metric .
– **Stemming** also dips below baseline in the balanced setting, but **at the high-recall point** it actually *improves* over baseline in precision (0.460 vs 0.451), F₁ (0.6216 vs 0.6133) and F₂—even though AUC is a bit lower (0.9500 vs 0.9565) .

**Conclusions:**

* **No normalization** (raw text) remains your strongest balanced classifier and yields the best overall ranking ability (AUC).
* If your priority is *strictly* hitting ≥95 % recall, **stemming** gives you a slightly higher precision/F₁ at that operating point than either lemmas or raw text.
* **Lemmatization** does not appear to improve performance over the baseline for either balanced or high-recall objectives.

In practice, you’d choose:

1. **Baseline** for general-purpose screening (maximize F₁/AUC).
2. **Stemming + lowered threshold** if you need guaranteed high recall with as little extra manual review as possible.

---
##### *Console Logs*
![6f568db0f88f119e2d77e903f770d5c7.png](:/7680f0091839450f9201443db06de030)

---