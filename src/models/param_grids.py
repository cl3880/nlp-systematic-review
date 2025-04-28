# src/models/param_grids.py

from imblearn.over_sampling    import SMOTE
from imblearn.under_sampling   import RandomUnderSampler
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC

from src.models.baseline_classifier import baseline_param_grid, CosineSimilarityClassifier
from src.models.svm_classifier      import svm_param_grid

def full_param_grid():
    """
    Combines:
      - normalization: none, stemming, lemmatization
      - TF-IDF & logreg hyperparams (from baseline_param_grid)
      - TF-IDF & svm hyperparams (from svm_param_grid)
      - balancing: none, SMOTE, undersample
      - classifier choice: logreg, svm, cosine
    into one big grid for GridSearchCV.
    """
    tfidf_lr_grid = baseline_param_grid()
    tfidf_svm_grid = svm_param_grid()

    tfidf_keys = {k: v for k, v in tfidf_lr_grid.items() if k.startswith("tfidf__")}
    
    return {
        "normalizer__technique": [None, "stemming", "lemmatization"],
        **tfidf_keys,
        "sampler": [
            None,
            SMOTE(random_state=42),
            RandomUnderSampler(random_state=42)
        ],
        "clf": [
            LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced"),
            SVC(kernel="linear", probability=True, class_weight="balanced"),
            CosineSimilarityClassifier()
        ],
        **{
            f"clf__{param_name.split('clf__',1)[1]}": values
            for param_name, values in tfidf_svm_grid.items()
            if param_name.startswith("clf__")
        }
    }
