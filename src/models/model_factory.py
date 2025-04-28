# src/models/model_factory.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from src.models.baseline_classifier import TextCombiner, CosineSimilarityClassifier
from src.scripts.compare_normalizations import NormalizingTextCombiner
from sklearn.naive_bayes import ComplementNB


def create_model(
    model_type="logreg",
    normalization=None,
    balancing=None,
    max_features=10000,
    ngram_range=(1, 2),
    min_df=3,
    text_columns=("title", "abstract"),
    C=1.0,
    class_weight="balanced",
):
    steps = []
    if normalization:
        steps.append(
            (
                "normalizer",
                NormalizingTextCombiner(
                    text_columns=text_columns, technique=normalization
                ),
            )
        )
    steps += [
        ("combiner", TextCombiner(text_columns)),
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
    ]

    if model_type == "logreg":
        steps.append(
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    C=C,
                    class_weight=class_weight,
                    solver="liblinear",
                    random_state=42,
                ),
            )
        )
    elif model_type == "svm":
        steps.append(
            (
                "clf",
                SVC(
                    kernel="linear",
                    probability=True,
                    C=C,
                    class_weight=class_weight,
                    random_state=42,
                ),
            )
        )
    elif model_type == "cosine":
        steps.append(("clf", CosineSimilarityClassifier()))
    elif model_type == "cnb":
        steps.append(("clf", ComplementNB(alpha=1.0)))

    pipeline = Pipeline(steps)

    if balancing:
        sampler = (
            SMOTE(random_state=42)
            if balancing == "smote"
            else RandomUnderSampler(random_state=42)
        )
        steps_with_sampler = (
            pipeline.steps[:-1] + [("sampler", sampler)] + [pipeline.steps[-1]]
        )
        pipeline = ImbPipeline(steps_with_sampler)

    return pipeline
