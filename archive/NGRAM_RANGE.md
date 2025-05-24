
## 1. Baseline Feature Representation

We follow the consensus in recent systematic‐review screening studies by featurizing each document with **TF–IDF over unigrams, bigrams and trigrams** (`ngram_range=(1,3)`):

* **LREC 2020** shows that adding 2- and 3-grams to unigrams yields an ∼10 percentage-point F₁ boost in SVM ranking experiments .
* **Norman et al. (2018)** universally adopt a bag-of-n-grams baseline with n ≤ 3 (TF–IDF + binary indicators, stemmed & unstemmed) and demonstrate it outperforms unigram-only models .
* **Cohen et al. (2009)** likewise extract up to trigrams weighted by TF–IDF in their SVM ranker .

We therefore include trigrams in our **default baseline**, and perform a **single 5-fold GridSearchCV** sweeping over both TF–IDF hyperparameters
(`tfidf__ngram_range`, `min_df`, `max_df`, `max_features`) and classifier settings (`C`, class\_weight, penalty) .

---

## 2. N-gram Ablation Results

| N-gram Range          | Precision | Recall | F₁     | F₂     | ROC AUC | WSS\@95 |
| --------------------- | --------- | ------ | ------ | ------ | ------- | ------- |
| **Balanced (1,2)**    | 0.4200    | 0.8750 | 0.5676 | 0.7192 | 0.9491  | 0.6564  |
| **Balanced (1,3)**    | 0.4200    | 0.8750 | 0.5676 | 0.7192 | 0.9450  | 0.6656  |
| **High-recall (1,2)** | 0.2875    | 0.9583 | 0.4423 | 0.6534 | 0.9491  | 0.6564  |
| **High-recall (1,3)** | 0.2771    | 0.9583 | 0.4299 | 0.6425 | 0.9450  | 0.6656  |

* **Balanced models**: identical F₁/F₂; (1,2) edges out on ROC AUC, while (1,3) wins WSS\@95 by 0.0092 .
* **High-recall models**: (1,2) yields marginally better precision (+0.0104) and F₁ (+0.0124) at 95% recall, though (1,3) again ticks up WSS\@95 slightly .

---

## 3. Discussion & Justification

1. **Literature alignment**
   Although LREC 2020 and Norman et al. recommend trigrams (n≤3) for their robust F₁ gains , our **smaller dataset** contains fewer distinct, high-value trigrams—so their corpus-level benefit largely vanishes here.

2. **Computational trade-off**
   Trigrams roughly **double** the feature space (and memory/runtime cost) for only **marginal** WSS\@95 improvements. Bigrams alone give equal or better ROC AUC and F₁ at a fraction of the cost.

3. **Final choice**

   * **Canonical baseline**: we adopt `(1,3)` to remain consistent with the field’s standard bag-of-n-grams baseline.
   * **Efficiency variant**: we report that `(1,2)` achieves near-identical performance and is the preferred choice when computation is constrained.

---

## 4. Methodological Improvements

To ensure a fair, reproducible comparison:

* **Unified preprocessing**: all models use the same `max_df=0.9` stop-word filtering and tokenization.
* **Threshold tuning**: thresholds for 95% recall are selected **only** on validation data, then locked for test evaluation to avoid leakage.
* **Reranking option**: as Cohen 2006 and later work suggest, one can use a Logistic‐Regression filter plus cosine‐similarity reranking for further precision gains at high recall.
