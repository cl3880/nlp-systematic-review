# STEMMING_CRITERIA_SVM Model Comparison

## Model Configurations

### Balanced Model (Optimized for F1)
- clf__C: 1
- clf__class_weight: balanced
- clf__kernel: linear
- features__text_features__tfidf__max_df: 0.85
- features__text_features__tfidf__max_features: 10000
- features__text_features__tfidf__min_df: 5
- features__text_features__tfidf__ngram_range: (1, 3)

### High-Recall Model (95% Target)
- clf__C: 0.1
- clf__class_weight: balanced
- clf__kernel: linear
- features__text_features__tfidf__max_df: 0.85
- features__text_features__tfidf__max_features: 5000
- features__text_features__tfidf__min_df: 10
- features__text_features__tfidf__ngram_range: (1, 2)

## Performance Comparison

| Metric | Balanced Model | High-Recall Model |
|--------|---------------|-------------------|
| precision | 0.6429 | 0.3898 |
| recall | 0.7500 | 0.9583 |
| f1 | 0.6923 | 0.5542 |
| f2 | 0.7258 | 0.7419 |
| roc_auc | 0.9521 | 0.9373 |
| wss at 95 | 0.8216 | 0.6794 |

## N-gram Range Analysis

Based on cross-validation results (see cv_results.csv), we found:

- **(1, 3)** n-grams: average F1 = 0.6523
- **(1, 2)** n-grams: average F1 = 0.6510

The (1,3) n-gram range outperforms (1,2) by **0.1 percentage points**.

## Methodological Notes

The high-recall configuration utilizes the same underlying model as the balanced configuration, with an adjusted decision threshold to prioritize recall (≥95%) at the expense of precision. This approach maintains identical feature coefficients while modifying only the decision boundary.
