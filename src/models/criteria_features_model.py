from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src.models.classifiers import TextCombiner
from src.models.criteria_features import InclusionExclusionTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def create_criteria_pipeline(model_type="logreg", **kwargs):
    """Creates a pipeline combining TF-IDF features with criteria-based features."""
    text_columns = kwargs.get("text_columns", ["title", "abstract"])
    max_features = kwargs.get("max_features", 10000)
    ngram_range = kwargs.get("ngram_range", (1, 3))
    min_df = kwargs.get("min_df", 3)
    max_df = kwargs.get("max_df", 0.95)

    text_pipeline = Pipeline(
        [
            ("combiner", TextCombiner(text_columns=text_columns)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=max_df,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    criteria_pipeline = Pipeline(
        [
            ("combiner", TextCombiner(text_columns=text_columns)),
            ("rules", InclusionExclusionTransformer()),
        ]
    )

    features_list = [
        ("text_features", text_pipeline),
        ("criteria_features", criteria_pipeline),
    ]

    if kwargs.get("use_mesh", False) and "mesh_terms" in kwargs.get(
        "available_columns", []
    ):
        mesh_pipeline = Pipeline(
            [
                ("selector", MeshTermsTransformer(col="mesh_terms")),
                (
                    "vectorizer",
                    CountVectorizer(
                        max_features=kwargs.get("mesh_max_features", 1000),
                        min_df=1,
                        max_df=1.0,
                        binary=True,
                    ),
                ),
            ]
        )
        features_list.append(("mesh_features", mesh_pipeline))

    features = FeatureUnion(features_list)

    if model_type == "logreg":
        clf = LogisticRegression(
            C=kwargs.get("C", 1.0),
            class_weight=kwargs.get("class_weight", "balanced"),
            solver=kwargs.get("solver", "liblinear"),
            max_iter=5000,
            random_state=42,
        )
    elif model_type == "svm":
        clf = SVC(
            C=kwargs.get("C", 1.0),
            class_weight=kwargs.get("class_weight", "balanced"),
            kernel=kwargs.get("kernel", "linear"),
            probability=True,
            random_state=42,
        )
    elif model_type == "cosine":
        from src.models.classifiers import CosineSimilarityClassifier

        clf = CosineSimilarityClassifier(threshold=kwargs.get("threshold", 0.3))
    elif model_type == "cnb":
        from sklearn.naive_bayes import ComplementNB

        clf = ComplementNB(alpha=kwargs.get("alpha", 1.0))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if kwargs.get("balancing") == "smote":
        return ImbPipeline([
            ("features", features),
            ("sampler",  SMOTE(random_state=42)),
            ("clf",      clf),
        ])

    return Pipeline([
        ("features", features),
        ("clf",      clf),
    ])


class MeshTermsTransformer(BaseEstimator, TransformerMixin):
    """Transform mesh terms column while preserving sample dimensionality"""

    def __init__(self, col="mesh_terms"):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure all samples have consistent representation
        if self.col not in X.columns:
            # If mesh_terms column is missing, return empty strings with correct dimensionality
            return np.array([""] * X.shape[0])
        else:
            # Fill NA values with empty string to maintain dimensionality
            return X[self.col].fillna("").values
