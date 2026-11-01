from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    hamming_loss,
)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

VALIDATION_FILE = Path(
    "data/processed/cuad_validation.csv"
)

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

OUTPUT_FILE = OUTPUT_DIR / "svm_threshold_tuning_results.csv"


# --------------------------------------------------
# LOAD VALIDATION DATA
# --------------------------------------------------

validation_df = pd.read_csv(
    VALIDATION_FILE
)

X_validation_text = (
    validation_df["clause_text"]
    .astype(str)
)

validation_labels = [
    value.split("|")
    for value in validation_df["labels"].astype(str)
]


# --------------------------------------------------
# LOAD SAVED COMPONENTS
# --------------------------------------------------

vectorizer = joblib.load(
    MODEL_DIR / "validation_tfidf_vectorizer.joblib"
)

model = joblib.load(
    MODEL_DIR / "validation_linear_svm.joblib"
)

mlb = joblib.load(
    MODEL_DIR / "validation_label_binarizer.joblib"
)


# --------------------------------------------------
# TRANSFORM DATA
# --------------------------------------------------

X_validation = vectorizer.transform(
    X_validation_text
)

y_validation = mlb.transform(
    validation_labels
)


# --------------------------------------------------
# GET SVM DECISION SCORES
# --------------------------------------------------

print("=" * 75)
print("LEXIRISK AI - LINEAR SVM THRESHOLD TUNING")
print("=" * 75)

decision_scores = model.decision_function(
    X_validation
)


print(
    "\nDecision score matrix shape:",
    decision_scores.shape
)

print(
    "Minimum decision score:",
    round(float(decision_scores.min()), 4)
)

print(
    "Maximum decision score:",
    round(float(decision_scores.max()), 4)
)


# --------------------------------------------------
# THRESHOLDS TO TEST
# --------------------------------------------------

thresholds = np.arange(
    -0.50,
    0.51,
    0.05,
)


results = []


# --------------------------------------------------
# EVALUATE THRESHOLDS
# --------------------------------------------------

for threshold in thresholds:

    y_pred = (
        decision_scores >= threshold
    ).astype(int)


    micro_precision = precision_score(
        y_validation,
        y_pred,
        average="micro",
        zero_division=0,
    )

    micro_recall = recall_score(
        y_validation,
        y_pred,
        average="micro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y_validation,
        y_pred,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_validation,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_validation,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    subset_accuracy = accuracy_score(
        y_validation,
        y_pred,
    )

    hamming = hamming_loss(
        y_validation,
        y_pred,
    )

    avg_predicted_labels = (
        y_pred.sum(axis=1).mean()
    )


    results.append(
        {
            "threshold": round(
                float(threshold),
                2,
            ),
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "subset_accuracy": subset_accuracy,
            "hamming_loss": hamming,
            "avg_predicted_labels":
                avg_predicted_labels,
        }
    )


# --------------------------------------------------
# RESULTS TABLE
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 115)
print("SVM THRESHOLD RESULTS")
print("=" * 115)


print(
    results_df.to_string(
        index=False,
        formatters={
            "micro_precision": "{:.4f}".format,
            "micro_recall": "{:.4f}".format,
            "micro_f1": "{:.4f}".format,
            "macro_f1": "{:.4f}".format,
            "weighted_f1": "{:.4f}".format,
            "subset_accuracy": "{:.4f}".format,
            "hamming_loss": "{:.4f}".format,
            "avg_predicted_labels": "{:.4f}".format,
        },
    )
)


# --------------------------------------------------
# BEST THRESHOLDS
# --------------------------------------------------

best_micro = results_df.loc[
    results_df["micro_f1"].idxmax()
]

best_macro = results_df.loc[
    results_df["macro_f1"].idxmax()
]

best_subset = results_df.loc[
    results_df["subset_accuracy"].idxmax()
]


print("\n" + "=" * 75)
print("BEST SVM VALIDATION THRESHOLDS")
print("=" * 75)


print(
    "\nBest threshold by Micro F1:",
    best_micro["threshold"]
)

print(
    "Micro Precision:",
    round(
        best_micro["micro_precision"],
        4,
    )
)

print(
    "Micro Recall:",
    round(
        best_micro["micro_recall"],
        4,
    )
)

print(
    "Micro F1:",
    round(
        best_micro["micro_f1"],
        4,
    )
)


print(
    "\nBest threshold by Macro F1:",
    best_macro["threshold"]
)

print(
    "Macro F1:",
    round(
        best_macro["macro_f1"],
        4,
    )
)


print(
    "\nBest threshold by Subset Accuracy:",
    best_subset["threshold"]
)

print(
    "Subset Accuracy:",
    round(
        best_subset["subset_accuracy"],
        4,
    )
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\nSaved:")
print(OUTPUT_FILE)