from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
)


# --------------------------------------------------
# FILE LOCATIONS
# --------------------------------------------------

TEST_FILE = Path(
    "data/processed/cuad_test_multilabel.csv"
)

MODEL_DIR = Path("models")

OUTPUT_DIR = Path("outputs")

OUTPUT_FILE = OUTPUT_DIR / "baseline_per_label_metrics.csv"


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

test_df = pd.read_csv(TEST_FILE)

X_test_text = test_df["clause_text"].astype(str)

test_labels = [
    labels.split("|")
    for labels in test_df["labels"].astype(str)
]


# --------------------------------------------------
# LOAD SAVED MODEL COMPONENTS
# --------------------------------------------------

vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

model = joblib.load(
    MODEL_DIR / "baseline_logistic_model.joblib"
)

mlb = joblib.load(
    MODEL_DIR / "label_binarizer.joblib"
)


# --------------------------------------------------
# TRANSFORM TEST DATA
# --------------------------------------------------

X_test = vectorizer.transform(
    X_test_text
)

y_test = mlb.transform(
    test_labels
)


# --------------------------------------------------
# MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(
    X_test
)


# --------------------------------------------------
# CALCULATE PER-LABEL METRICS
# --------------------------------------------------

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average=None,
        zero_division=0,
    )
)


results = pd.DataFrame(
    {
        "category": mlb.classes_,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "test_support": support,
    }
)


# Sort by F1 score from lowest to highest
results = results.sort_values(
    by="f1_score",
    ascending=True,
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("=" * 90)
print("BASELINE PERFORMANCE BY CUAD CATEGORY")
print("=" * 90)

print(
    results.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1_score": "{:.4f}".format,
        },
    )
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print("\nLowest 5 F1 categories:")

print(
    results[
        [
            "category",
            "f1_score",
            "test_support",
        ]
    ].head(5).to_string(index=False)
)


print("\nHighest 5 F1 categories:")

print(
    results[
        [
            "category",
            "f1_score",
            "test_support",
        ]
    ].tail(5).sort_values(
        "f1_score",
        ascending=False,
    ).to_string(index=False)
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\nSaved:")
print(OUTPUT_FILE)