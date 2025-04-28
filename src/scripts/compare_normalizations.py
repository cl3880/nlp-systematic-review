#!/usr/bin/env python3
"""
Compare baseline model with different text normalization approaches.
"""
import os
import argparse
import logging
import joblib
import nltk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.config import PATHS
from src.utils.data_utils import load_data, make_splits
from src.utils.evaluate import evaluate, compare_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(PATHS["logs_dir"], "normalization_comparison.log")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def ensure_nltk_resources():
    checks = {
        "punkt": "tokenizers/punkt",
        "wordnet": "corpora/wordnet",
    }
    for pkg, path in checks.items():
        try:
            nltk.data.find(path)
        except LookupError:
            # quiet=True
            nltk.download(pkg, quiet=True)

ensure_nltk_resources()

class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, technique=None):
        self.technique = technique
        self.stemmer    = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.technique is None:
            return X
        return [self.normalize_doc(doc) for doc in X]

    def normalize_doc(self, doc):
        tokens = doc.split()
        if self.technique == 'stemming':
            tokens = [self.stemmer.stem(t) for t in tokens]
        elif self.technique == 'lemmatization':
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)


class NormalizingTextCombiner(BaseEstimator, TransformerMixin):
    """Combines text columns and applies normalization."""

    def __init__(self, text_columns=["title", "abstract"], technique=None):
        self.text_columns = text_columns
        self.technique = technique

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Combine text columns and apply normalization."""
        if hasattr(X, "iloc"):
            combined = X[self.text_columns[0]].fillna("")
            for col in self.text_columns[1:]:
                if col in X.columns:
                    combined = combined + " " + X[col].fillna("")
            combined = combined.values
        else:
            combined = X

        normalizer = TextNormalizer(technique=self.technique)
        return normalizer.transform(combined)


def make_normalized_pipeline(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=3,
    text_columns=["title", "abstract"],
    C=1.0,
    class_weight="balanced",
    technique=None,
):
    """Create a pipeline with text normalization."""

    logger.info(
        f"Creating TF-IDF + LogReg pipeline with {technique if technique else 'no'} normalization"
    )

    return Pipeline(
        [
            ("combiner", NormalizingTextCombiner(text_columns, technique=technique)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=5000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def make_normalized_svm_pipeline(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=3,
    text_columns=None,
    C=1.0,
    class_weight="balanced",
    technique=None,
    kernel="linear",
):
    """Create an SVM pipeline with text normalization."""
    from sklearn.svm import SVC

    if text_columns is None:
        text_columns = ["title", "abstract"]

    logger.info(
        f"Creating TF-IDF + SVM pipeline with {kernel} kernel and {technique if technique else 'no'} normalization"
    )

    return Pipeline(
        [
            ("combiner", NormalizingTextCombiner(text_columns, technique=technique)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                SVC(
                    C=C,
                    class_weight=class_weight,
                    kernel=kernel,
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )


def make_normalized_cosine_pipeline(
    max_features=10000, ngram_range=(1, 2), min_df=3, text_columns=None, technique=None
):
    """Create a cosine similarity pipeline with text normalization."""
    from src.models.baseline_classifier import CosineSimilarityClassifier

    if text_columns is None:
        text_columns = ["title", "abstract"]

    logger.info(
        f"Creating TF-IDF + Cosine Similarity pipeline with {technique if technique else 'no'} normalization"
    )

    return Pipeline(
        [
            ("combiner", NormalizingTextCombiner(text_columns, technique=technique)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            ("cosine", CosineSimilarityClassifier()),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare text normalization approaches"
    )
    parser.add_argument(
        "--data",
        default=os.path.join(PATHS["data_processed"], "data_final_cleaned.csv"),
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PATHS["results_dir"], "normalization"),
        help="Directory for outputs",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="Target recall level (default: 0.95)",
    )
    args = parser.parse_args()

    models_dir = os.path.join(args.output_dir, "models")
    metrics_dir = os.path.join(args.output_dir, "metrics")
    plots_dir = os.path.join(args.output_dir, "plots")
    analysis_dir = os.path.join(args.output_dir, "analysis")

    for directory in [
        args.output_dir,
        models_dir,
        metrics_dir,
        plots_dir,
        analysis_dir,
    ]:
        os.makedirs(directory, exist_ok=True)

    logger.info(f"Starting normalization comparison with data from {args.data}")

    df = load_data(args.data)
    train, val, test = make_splits(df, stratify=True, seed=42)

    logger.info(f"Dataset: {len(df)} total examples ({df['relevant'].sum()} relevant)")
    logger.info(f"Train: {len(train)} examples ({train['relevant'].sum()} relevant)")
    logger.info(f"Validation: {len(val)} examples ({val['relevant'].sum()} relevant)")
    logger.info(f"Test: {len(test)} examples ({test['relevant'].sum()} relevant)")

    model_params = {
        "max_features": 5000,
        "ngram_range": (1, 2),
        "min_df": 5,
        "C": 1.0,
        "class_weight": "balanced",
    }

    models = {
        "baseline": make_tfidf_logreg_pipeline(
            max_features=model_params["max_features"],
            ngram_range=model_params["ngram_range"],
            min_df=model_params["min_df"],
            C=model_params["C"],
            class_weight=model_params["class_weight"],
        ),
        "stemming": make_normalized_pipeline(
            max_features=model_params["max_features"],
            ngram_range=model_params["ngram_range"],
            min_df=model_params["min_df"],
            C=model_params["C"],
            class_weight=model_params["class_weight"],
            technique="stemming",
        ),
        "lemmatization": make_normalized_pipeline(
            max_features=model_params["max_features"],
            ngram_range=model_params["ngram_range"],
            min_df=model_params["min_df"],
            C=model_params["C"],
            class_weight=model_params["class_weight"],
            technique="lemmatization",
        ),
    }

    all_metrics = {}

    for model_name, model in models.items():
        logger.info(f"Training {model_name} model")
        model.fit(train, train["relevant"])

        joblib.dump(model, os.path.join(models_dir, f"{model_name}_model.joblib"))

        try:
            plot_top_features(
                model, os.path.join(plots_dir, f"{model_name}_top_features.png")
            )
        except Exception as e:
            logger.warning(f"Could not plot top features for {model_name}: {e}")

        logger.info(f"Evaluating {model_name} model on validation set")
        val_preds = model.predict(val)
        val_probs = model.predict_proba(val)[:, 1]

        val_metrics = evaluate(
            val["relevant"].values,
            val_preds,
            val_probs,
            metrics_dir,
            f"{model_name}_validation",
            target_recall=args.target_recall,
        )

        logger.info(f"Evaluating {model_name} model on test set")
        test_preds = model.predict(test)
        test_probs = model.predict_proba(test)[:, 1]

        test_metrics = evaluate(
            test["relevant"].values,
            test_preds,
            test_probs,
            metrics_dir,
            f"{model_name}_test",
            target_recall=args.target_recall,
        )

        all_metrics[model_name] = test_metrics

        test_with_preds = test.copy()
        test_with_preds["prediction"] = test_preds
        test_with_preds["probability"] = test_probs
        test_with_preds["correct"] = (
            test_with_preds["relevant"] == test_with_preds["prediction"]
        )
        test_with_preds.to_csv(
            os.path.join(analysis_dir, f"{model_name}_predictions.csv"), index=False
        )

    model_list = [(name, metrics) for name, metrics in all_metrics.items()]
    compare_models(model_list, plots_dir, "normalization_comparison")

    with open(os.path.join(args.output_dir, "normalization_summary.txt"), "w") as f:
        f.write("===== TEXT NORMALIZATION COMPARISON =====\n\n")

        f.write("Dataset Information:\n")
        f.write(f"- Total records: {len(df)}\n")
        f.write(
            f"- Relevant documents: {df['relevant'].sum()} ({df['relevant'].mean()*100:.1f}%)\n\n"
        )

        for model_name, metrics in all_metrics.items():
            f.write(f"{model_name.capitalize()} Model:\n")
            f.write(f"- Test AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"- Test Precision: {metrics['precision']:.4f}\n")
            f.write(f"- Test Recall: {metrics['recall']:.4f}\n")
            f.write(f"- Test F1: {metrics['f1']:.4f}\n")
            f.write(f"- Test WSS@95: {metrics['wss@95']:.4f}\n\n")

        best_model = max(all_metrics.items(), key=lambda x: x[1]["f1"])[0]
        f.write(f"The {best_model.capitalize()} model performs best on F1 score.\n")

        best_wss_model = max(all_metrics.items(), key=lambda x: x[1]["wss@95"])[0]
        f.write(
            f"The {best_wss_model.capitalize()} model saves most work at 95% recall.\n"
        )

    logger.info(
        f"Normalization comparison complete. Results saved to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
