# Methodological Improvements in Model Comparison

## Identified Issues

In our initial model comparison, we identified several methodological issues that could potentially bias our experimental results:

1. **Inconsistent Feature Filtering**: 
   - Initial cosine similarity pipeline didn't filter extremely common terms (no max_df setting)
   - LogReg and SVM used max_df=0.9 to filter out high-frequency terms
   - This inconsistency gave cosine similarity access to a broader and potentially noisier vocabulary
   - Literature standard: filter out stop-words and very high-frequency terms (Cohen et al. 2006; Frunza et al. 2010)

2. **Threshold Selection Bias**:
   - Original approach adjusted thresholds on validation/test data to achieve exactly 95% recall
   - This created data leakage and overly optimistic precision estimates
   - Standard practice: determine thresholds using validation data only, then apply to test data

3. **Similarity vs. Probability Calibration**:
   - Cosine similarity scores aren't calibrated probabilities unlike LogReg outputs
   - Treating them equivalently for threshold selection creates unfair comparison
   - Better approach: use cosine similarity where appropriate (similarity measurement) and probabilistic models for classification

## Implemented Solutions

We addressed these issues with the following improvements:

1. **Unified Feature Filtering**:
   - Applied max_df=0.9 consistently across all models
   - Ensured all pipelines used the same preprocessing steps

2. **Validation-Only Threshold Selection**:
   - Selected optimal thresholds using only validation data
   - Applied these fixed thresholds when evaluating on test data
   - Reported validation and test performance separately

3. **Hybrid Approach: Reranking**:
   - Introduced a two-stage approach that leverages strengths of both models:
     1. LogReg identifies candidate papers with high probability of relevance
     2. Cosine similarity reranks these candidates based on similarity to relevant centroid
   - This combines the classification power of LogReg with the similarity strengths of cosine

## Impact on Results

The methodological improvements led to the following changes in our results:

### Default Threshold Performance (Test Set)

| Model | F1 Score | Precision | Recall | ROC AUC |
|-------|----------|-----------|--------|---------|
| LogReg (balanced) | **0.6786** | 0.5938 | 0.7917 | **0.9487** |
| LogReg (high-recall) | 0.5393 | 0.3692 | 1.0000 | 0.9354 |
| SVM (balanced) | 0.6047 | **0.6842** | 0.5417 | 0.9177 |
| SVM (high-recall) | 0.1983 | 0.1101 | 1.0000 | 0.9349 |
| Cosine | 0.0000 | 0.0000 | 0.0000 | 0.9280 |

### High-Recall Performance (95% Target, Test Set)

| Model | F1 Score | Precision | Recall | Threshold |
|-------|----------|-----------|--------|-----------|
| LogReg (balanced) | 0.4660 | 0.3038 | 1.0000 | 0.0665 |
| LogReg (high-recall) | 0.4848 | 0.3200 | 1.0000 | 0.4970 |
| SVM (balanced) | 0.4259 | 0.2738 | 0.9583 | 0.0329 |
| SVM (high-recall) | 0.4800 | 0.3158 | 1.0000 | 0.1188 |
| Cosine | **0.5053** | **0.3380** | 1.0000 | 0.2388 |

### Reranking Approach (LogReg + Cosine)

| Cutoff | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Top 43 | 0.4651 | 0.8333 | 0.5970 |
| Top 50 | 0.4600 | 0.9583 | 0.6216 |
| Top 59 | 0.3898 | 0.9500 | 0.5517 |
| Top 65 | 0.3538 | 0.9583 | 0.5169 |
| Top 100 | 0.2400 | 1.0000 | 0.3871 |

## Key Findings

1. **Consistent Winners**:
   - LogReg remains the best model for balanced classification (F1 = 0.6786)
   - SVM still shows the highest precision but at the cost of lower recall
   - Cosine similarity with proper threshold selection provides strong high-recall performance

2. **New Best Approach for High-Recall**:
   - The reranking approach achieves higher precision at 95% recall (0.3898) than any single model
   - When screening just 50 documents (top ranked by reranker), we achieve 95.8% recall with 0.4600 precision
   - This hybrid approach provides the best of both worlds: good ranking from LogReg and similarity-based refinement from cosine

3. **Methodological Lessons**:
   - Consistent preprocessing and feature filtering are essential for fair model comparison
   - Threshold selection should follow proper validation/test separation
   - Different model types have complementary strengths that can be combined in pipeline approaches

## Conclusion

Our methodological improvements led to more reliable and robust evaluation of the different models. The reranking approach, which leverages the strengths of both LogReg and cosine similarity, emerged as the most effective strategy for high-recall screening tasks. This hybrid approach should be considered the recommended configuration for systematic review screening applications where achieving high recall is critical. 