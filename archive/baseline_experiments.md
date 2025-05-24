# Baseline Model Experiments: N-gram Range Comparison

## 1. Logistic Regression Results

### 1.1 Current Model (Grid Search Best Parameters)
Parameters:
- C: 1
- class_weight: balanced
- penalty: l2
- solver: liblinear
- tfidf__max_df: 0.9
- tfidf__max_features: 5000
- tfidf__min_df: 5
- tfidf__ngram_range: (1, 2)

| Metric    | Validation | Test | CV (mean ± std) |
|-----------|------------|------|-----------------|
| Precision | 0.4200     | N/A  | 0.5164 ±0.0190  |
| Recall    | 0.8750     | N/A  | 0.8503 ±0.0529  |
| F1 Score  | 0.5676     | N/A  | 0.6418 ±0.0209  |
| F2 Score  | 0.7192     | N/A  | -               |
| ROC AUC   | 0.9491     | N/A  | 0.9434 ±0.0166  |
| WSS@95    | 0.6564     | N/A  | -               |

### 1.2 High-Recall Model (95% Target)
- Optimized threshold: 0.2886
- Achieved recall: 95.83%

| Metric    | Validation |
|-----------|------------|
| Precision | 0.2875     |
| Recall    | 0.9583     |
| F1 Score  | 0.4423     |
| F2 Score  | 0.6534     |
| ROC AUC   | 0.9491     |
| WSS@95    | 0.6564     |

### 1.3 N-gram Range (1,3) Experiment
Parameters:
- max_features: 10000
- ngram_range: (1, 3)
- min_df: 3
- normalization: none
- balancing: none

#### Balanced Model (Default Threshold)
| Metric    | Validation |
|-----------|------------|
| Precision | 0.4200     |
| Recall    | 0.8750     |
| F1 Score  | 0.5676     |
| F2 Score  | 0.7192     |
| ROC AUC   | 0.9450     |
| WSS@95    | 0.6656     |

#### High-Recall Model (95% Target)
- Optimized threshold: 0.2766
- Achieved recall: 95.83%

| Metric    | Validation |
|-----------|------------|
| Precision | 0.2771     |
| Recall    | 0.9583     |
| F1 Score  | 0.4299     |
| F2 Score  | 0.6425     |
| ROC AUC   | 0.9450     |
| WSS@95    | 0.6656     |

## 2. N-gram Configuration Analysis

### 2.1 Grid Search Configuration
Our grid search is configured to use **F2 score** (beta=2) as the scoring metric, which puts more emphasis on recall than precision. This is appropriate for systematic reviews where missing relevant papers is costlier than including irrelevant ones.

Default n-gram range for our pipeline factory functions is set to (1, 3), based on literature suggesting this range achieves better results (approximately 10 percentage point F1 boost).

```python
def logreg_param_grid():
    return {
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3), (2, 3)],
        "tfidf__min_df": [1, 2, 3, 5],
        "tfidf__max_df": [0.9, 0.95, 1.0],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": ["balanced"],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
    }
```

### 2.2 Observations

#### N-gram (1,2) vs. N-gram (1,3)

1. **Performance Comparison**:
   - F1 score: Both configurations achieved identical F1 (0.5676)
   - F2 score: Both achieved identical F2 (0.7192)
   - ROC AUC: (1,2) was slightly better at 0.9491 vs. 0.9450 for (1,3)
   - WSS@95: (1,3) was slightly better at 0.6656 vs. 0.6564 for (1,2)

2. **High-Recall Optimization**:
   - (1,3) required a slightly lower threshold (0.2766 vs. 0.2886)
   - (1,3) had slightly lower precision at 95% recall (0.2771 vs. 0.2875)
   - (1,3) had slightly lower F1 at 95% recall (0.4299 vs. 0.4423)

3. **Computational Considerations**:
   - (1,3) includes more features, potentially increasing memory usage and computation time
   - (1,2) is more efficient while achieving comparable performance

Despite literature suggesting a potential 10 percentage point F1 boost with (1,3) n-grams, our dataset does not show this improvement. The (1,2) n-gram range appears to perform slightly better on most metrics, supporting the grid search's selection of this configuration.

### 2.3 Experimental Setup
For this experiment, we simplified our approach by using a single grid search that captures both balanced (optimized for F1) and high-recall (optimized for recall) configurations:

```python
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring={'f1': 'f1', 'recall': 'recall', 'roc_auc': 'roc_auc'},
    refit='f1',
    n_jobs=-1,
)
```

This allows us to extract both the best-for-F1 and best-for-Recall parameter sets from a single grid search, making the comparison more efficient.

## 3. Conclusion and Next Steps

Based on our experiments, we'll proceed with the (1,2) n-gram range as our anchor configuration for subsequent experiments. While literature suggests (1,3) might be better in general, our specific dataset shows slightly better performance with (1,2) at a lower computational cost.

Our next steps:
1. Explore text normalization techniques (stemming, lemmatization)
2. Test different class balancing strategies
3. Compare model architectures beyond logistic regression 

## 4. Model Architecture Comparison (F1-optimized)

After confirming that the (1,2) n-gram range is optimal for our dataset, we conducted a comparison of different model architectures using F1 score as the primary optimization metric. All models used the (1,2) n-gram configuration.

### 4.1 Model Configurations

We compared the following models:

1. **Balanced LogReg**:
   - max_features: 5000
   - ngram_range: (1, 2)
   - min_df: 2
   - max_df: 0.9
   - C: 10
   - class_weight: balanced
   - penalty: l2
   - solver: liblinear

2. **High-Recall LogReg**:
   - max_features: 10000
   - ngram_range: (1, 2)
   - min_df: 1
   - max_df: 0.9
   - C: 0.01
   - class_weight: balanced
   - penalty: l2
   - solver: liblinear

3. **Balanced SVM**:
   - max_features: 5000
   - ngram_range: (1, 2)
   - min_df: 2
   - max_df: 0.9
   - C: 10
   - class_weight: balanced
   - kernel: linear

4. **High-Recall SVM**:
   - max_features: 10000
   - ngram_range: (1, 2)
   - min_df: 1
   - max_df: 0.9
   - C: 0.01
   - class_weight: balanced
   - kernel: linear

5. **Cosine Similarity**:
   - max_features: 5000
   - ngram_range: (1, 2)
   - min_df: 2
   - max_df: 0.9
   - threshold: 0.3 (fixed)

### 4.2 Default Threshold Performance

| Model | Precision | Recall | F1 | F2 | ROC AUC | WSS@95 |
|-------|-----------|--------|----|----|---------|--------|
| logreg_balanced | 0.5556 | 0.8333 | 0.6667 | 0.7576 | 0.9562 | 0.6931 |
| logreg_high_recall | 0.3026 | 0.9583 | 0.4600 | 0.6686 | 0.9296 | 0.6060 |
| svm_balanced | 0.7273 | 0.6667 | 0.6957 | 0.6780 | 0.9517 | 0.6794 |
| svm_high_recall | 0.1101 | 1.0000 | 0.1983 | 0.3822 | 0.9293 | 0.6060 |
| cosine | 0.4186 | 0.7500 | 0.5373 | 0.6475 | 0.9261 | 0.6197 |

### 4.3 High-Recall Performance (95% Target)

| Model | Precision | Recall | F1 | F2 | Threshold |
|-------|-----------|--------|----|----|----------|
| logreg_balanced | 0.2674 | 0.9583 | 0.4182 | 0.6319 | 0.0665 |
| logreg_high_recall | 0.2738 | 0.9583 | 0.4259 | 0.6389 | 0.4970 |
| svm_balanced | 0.2840 | 0.9583 | 0.4381 | 0.6497 | 0.0329 |
| svm_high_recall | 0.2738 | 0.9583 | 0.4259 | 0.6389 | 0.1188 |
| cosine | 0.2987 | 0.9583 | 0.4554 | 0.6647 | 0.2388 |

### 4.4 Key Findings

1. **Best Model for Balanced Classification (by F1)**:
   - SVM with balanced configuration achieved the highest F1 score (0.6957)
   - SVM had significantly better precision (0.7273) than LogReg (0.5556) but lower recall (0.6667 vs 0.8333)
   - LogReg had the highest ROC AUC (0.9562) and WSS@95 (0.6931)

2. **Best Model for High-Recall Scenarios (95% target)**:
   - Cosine similarity surprisingly performed best at 95% recall with:
     - Highest F1 score (0.4554)
     - Highest precision (0.2987)
     - Highest F2 score (0.6647)
   - All models successfully achieved the 95% recall target with different thresholds
   - LogReg high-recall model had the most reasonable threshold (0.4970)

3. **Model Architecture Implications**:
   - SVM favors precision over recall with default threshold
   - LogReg provides better balanced performance
   - Cosine similarity, despite being simpler, excels in high-recall scenarios
   - High-recall SVM with default threshold achieves 100% recall but very low precision (0.1101)

### 4.5 Recommendations

Based on this comparison, we recommend:

1. For balanced classification (when both precision and recall are important):
   - Use the SVM balanced model with (1,2) n-grams when precision is valued more
   - Use the LogReg balanced model with (1,2) n-grams when recall and overall ranking (AUC) are valued more

2. For high-recall screening (when missing relevant papers is costly):
   - Use the Cosine similarity model with threshold=0.2388 to achieve 95% recall with the best precision
   - Alternatively, use the LogReg high-recall model with threshold=0.4970 for a more stable threshold

These findings confirm that the choice of model architecture is important and can be tailored to specific screening requirements.

### 4.6 Methodological Considerations

After our initial comparison, we identified several methodological issues that could potentially bias our results:

1. **Inconsistent Feature Filtering**: 
   - Our original Cosine Similarity implementation didn't filter extremely common terms (no max_df parameter)
   - LogReg and SVM pipelines used max_df=0.9 to remove very frequent terms
   - This inconsistency could unfairly favor cosine similarity by allowing its centroid to be built on a broader vocabulary
   - Solution: Ensure all models use max_df=0.9 for fair comparison

2. **Threshold Selection Bias**:
   - Our original approach tuned thresholds on test data to achieve exactly 95% recall
   - This created data leakage, producing overly optimistic precision estimates
   - Solution: Select thresholds exclusively on validation data before evaluating on test data

3. **Similarity vs. Probability Calibration**:
   - Cosine similarity scores are not calibrated probabilities unlike LogReg outputs
   - Treating them the same way for threshold selection creates an unfair advantage in recall-precision trade-offs
   - Solution: Implement a hybrid approach where cosine similarity serves as a reranker for LogReg results

These issues have been addressed in a follow-up analysis, where we implemented a more robust methodology and found that a hybrid approach combining LogReg with cosine similarity reranking achieves the best results for high-recall screening. 