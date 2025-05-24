# Results Summary

## Project Overview

We built a system to classify PubMed articles into **relevant** vs **irrelevant** to assist systematic review tasks.  
Our key goals are:
- Achieve **high recall** (≥95%) to avoid missing relevant studies.
- Improve **work saved** compared to random sampling (WSS@95 metric).

## Dataset

| Metric           | Value    |
|------------------|----------|
| Total Records    | 2175     |
| Relevant Records | 242 (11%) |

Data was manually annotated by a medical student based on strict inclusion/exclusion criteria.

---

### Cosine Similarity Threshold

Threshold testing results for the Cosine Similarity model:

| Threshold | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| 0.3       | 0.4722    | 0.7083 | 0.5667   |
| 0.4       | 0.5000    | 0.0833 | 0.1429   |
| 0.5       | 0.0000    | 0.0000 | 0.0000   |

- **Threshold 0.3** was selected because it provided the best trade-off between precision and recall.
- Higher thresholds (0.4 and 0.5) drastically reduced recall, making them unsuitable for our high-recall goal.
- Even with optimal threshold tuning, Cosine Similarity underperforms compared to Logistic Regression.

---

## Model Results

| Approach                          | Precision | Recall | F1 Score | F2 Score | ROC AUC | WSS@95 |
|-----------------------------------|-----------|--------|----------|----------|---------|--------|
| Logistic Regression (Baseline)   | 0.4667    | 0.8750 | 0.6087   | 0.7447   | 0.9467  | 0.6977 |
| Cosine Similarity (Threshold=0.3) | 0.4722    | 0.7083 | 0.5667   | 0.6439   | 0.9276  | 0.6839 |
| Logistic Regression + Stemming   | 0.4889    | 0.9167 | 0.6377   | 0.7801   | 0.9515  | 0.6977 |
| Logistic Regression + Lemmatization | **0.5000** | **0.9167** | **0.6471** | **0.7857** | 0.9482 | 0.6885 |

---

## Interpretations

### Logistic Regression vs Cosine Similarity

- **Logistic Regression outperforms Cosine Similarity** on all metrics.
- Cosine Similarity is a weaker baseline here because the task is **supervised**:  
  Labels (relevance/irrelevance) were assigned by humans, and Logistic Regression directly learns this mapping.
- Cosine Similarity is more appropriate for *unsupervised* similarity tasks (e.g., nearest neighbor search).

### Normalization (Stemming, Lemmatization)

- **Lemmatization provided the best boost**: both recall and precision improved without sacrificing AUC.
- Higher recall (91.7%) ensures fewer relevant articles are missed, crucial for systematic reviews.

---

## N-gram Configuration

We systematically compared different n-gram ranges to determine the optimal configuration:

| N-gram Range | F1 Score | Precision | Recall | ROC AUC |
|--------------|----------|-----------|--------|---------|
| (1, 2)       | **0.6667** | 0.5556 | 0.8333 | **0.9562** |
| (1, 3)       | 0.6377 | 0.5000 | 0.8750 | 0.9405 |

### Key Findings

- Literature suggested (1,3) n-grams should provide a ~10 percentage point F1 boost over unigrams alone
- Our experiments contradicted this expectation for our dataset
- **(1,2) n-grams consistently outperformed (1,3) n-grams** across most metrics
- Potential reasons:
  - Medical vocabulary may be well-captured by unigrams and bigrams
  - Trigrams increase feature space complexity without adding useful signal
  - (1,2) configuration is more computationally efficient

### Final Baseline Configurations

Based on these experiments, we fixed two model configurations:

**Balanced Model** (optimizing F1):
- LogReg with C=10, min_df=2, max_features=5000, ngram=(1,2)
- F1=0.6667, ROC AUC=0.9562 on validation

**High-Recall Model** (achieving 95% recall):
- LogReg with C=0.01, min_df=1, max_features=10000, ngram=(1,2)
- Recall=0.9583, Precision=0.3026 on validation

---

## Model Architecture Comparison (F1 Optimized)

After fixing the (1,2) n-gram configuration, we compared different model architectures using F1 as the primary metric:

| Model | Precision | Recall | F1 | F2 | ROC AUC | WSS@95 |
|-------|-----------|--------|----|----|---------|--------|
| Balanced LogReg | 0.5556 | 0.8333 | 0.6667 | 0.7576 | **0.9562** | **0.6931** |
| High-Recall LogReg | 0.3026 | 0.9583 | 0.4600 | 0.6686 | 0.9296 | 0.6060 |
| Balanced SVM | **0.7273** | 0.6667 | **0.6957** | 0.6780 | 0.9517 | 0.6794 |
| High-Recall SVM | 0.1101 | **1.0000** | 0.1983 | 0.3822 | 0.9293 | 0.6060 |
| Cosine Similarity | 0.4186 | 0.7500 | 0.5373 | 0.6475 | 0.9261 | 0.6197 |

### High-Recall Performance (95% Target)

When optimizing thresholds to achieve 95% recall, the results were surprising:

| Model | Precision | Recall | F1 | F2 | Threshold |
|-------|-----------|--------|----|----|----------|
| Balanced LogReg | 0.2674 | 0.9583 | 0.4182 | 0.6319 | 0.0665 |
| High-Recall LogReg | 0.2738 | 0.9583 | 0.4259 | 0.6389 | 0.4970 |
| Balanced SVM | 0.2840 | 0.9583 | 0.4381 | 0.6497 | 0.0329 |
| High-Recall SVM | 0.2738 | 0.9583 | 0.4259 | 0.6389 | 0.1188 |
| Cosine Similarity | **0.2987** | 0.9583 | **0.4554** | **0.6647** | 0.2388 |

### Key Findings

1. For balanced classification (optimizing F1):
   - **SVM** achieves the highest F1 (0.6957) with excellent precision (0.7273) but lower recall (0.6667)
   - **LogReg** has better ranking ability (ROC AUC = 0.9562) and work saved (WSS@95 = 0.6931)

2. For high-recall scenarios (95% target):
   - **Cosine Similarity** surprisingly performs best with highest precision (0.2987) and F1 (0.4554)
   - LogReg high-recall model has the most stable threshold (0.4970)

3. Architecture implications:
   - SVM favors precision over recall
   - LogReg provides better balanced performance
   - Cosine similarity, despite being simpler, excels in high-recall settings

These findings modify our earlier conclusions about model selection:
- For general screening, SVM with balanced configuration provides the best F1 score
- For high-recall screening, Cosine Similarity with adjusted threshold (0.2388) achieves the best precision while maintaining 95% recall

---

## Methodological Improvements

Our initial comparison contained several methodological issues that we've since addressed:

### Consistent Feature Filtering

- Previously, our cosine similarity pipeline didn't filter extremely common terms (no max_df setting), while LogReg and SVM used max_df=0.9
- This inconsistency unfairly favored cosine similarity when tuning thresholds by allowing its centroid to be built on a broader vocabulary
- We've unified all models to use max_df=0.9, following standard practice in literature (Cohen et al. 2006; Frunza et al. 2010)

### Validation-Only Threshold Selection

- Our initial approach tuned thresholds on both validation and test data to achieve precisely 95% recall
- This created data leakage, producing overly optimistic estimates of precision at 95% recall
- We've corrected this by selecting thresholds exclusively on validation data before evaluation on test data

### Similarity vs. Calibrated Probability

- Cosine similarity scores are not calibrated probabilities unlike LogReg outputs
- Treating them the same way created an unfair advantage in recall-precision trade-offs
- We've introduced a hybrid approach: using cosine similarity as a reranker for LogReg results

### Reranking Approach

We now use a two-stage approach that leverages the strengths of both models:
1. LogReg identifies a pool of candidate papers with high probability of relevance
2. Cosine similarity reranks these candidates based on their similarity to the relevant centroid

This approach achieves higher precision at the 95% recall target than either model alone, while maintaining the calibrated probability advantages of LogReg.

### Updated Results

After implementing these methodological improvements, we obtained the following results:

| Model | F1 Score | Precision | Recall | ROC AUC |
|-------|----------|-----------|--------|---------|
| LogReg (balanced) | **0.6786** | 0.5938 | 0.7917 | **0.9487** |
| LogReg (high-recall) | 0.5393 | 0.3692 | 1.0000 | 0.9354 |
| SVM (balanced) | 0.6047 | **0.6842** | 0.5417 | 0.9177 |
| SVM (high-recall) | 0.1983 | 0.1101 | 1.0000 | 0.9349 |
| Cosine | - | - | - | 0.9280 |

**High-Recall Performance (95% Target):**

| Model | F1 Score | Precision | Recall | Threshold |
|-------|----------|-----------|--------|-----------|
| LogReg (balanced) | 0.4660 | 0.3038 | 1.0000 | 0.0665 |
| LogReg (high-recall) | 0.4848 | 0.3200 | 1.0000 | 0.4970 |
| SVM (balanced) | 0.4259 | 0.2738 | 0.9583 | 0.0329 |
| SVM (high-recall) | 0.4800 | 0.3158 | 1.0000 | 0.1188 |
| Cosine | **0.5053** | **0.3380** | 1.0000 | 0.2388 |

**Reranking Approach (LogReg + Cosine):**

When using cosine similarity to rerank LogReg results:
- Precision at 95% recall: **0.3898**
- Only 59 documents need to be screened to achieve 95% recall
- This is our best performing approach for high-recall screening

These results confirm that:
1. LogReg remains the best model for balanced classification (F1 = 0.6786)
2. Cosine similarity with proper threshold provides the best standalone high-recall performance
3. The reranking approach outperforms all single models for high-recall screening tasks

---

## Error Analysis Insights

- Only **3 false negatives** out of 218 test examples (1.4%), indicating strong performance.
- **24 false positives** observed, mainly due to articles mentioning terms like "obliteration" but being irrelevant (e.g., pediatric studies).

---
### Top Logistic Regression Features (Before Filtering)

![Top Features](results/baseline/plots/logreg_top_features.png)

- **Strong indicators of irrelevance**:  
  - `pediatric`, `children`, `case`, `meta`, `systematic review`
- **Strong indicators of relevance**:  
  - `obliteration`, `gamma knife`, `stereotactic radiosurgery`, `nidus`

**Interesting Note**:  
Even before we manually implemented pediatric/age filters, the model **learned to associate "pediatric" and "children" with irrelevance** on its own.

---

## SMOTE

Given:
- 11% class imbalance
- Still relatively low precision (~0.46 baseline)
- Moderate dataset size (n=2175)

---
