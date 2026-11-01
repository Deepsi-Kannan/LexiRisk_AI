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

OUTPUT_FILE = OUTPUT_DIR / "threshold_tuning_results.csv"


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
    MODEL_DIR / "validation_logistic_model.joblib"
)

mlb = joblib.load(
    MODEL_DIR / "validation_label_binarizer.joblib"
)


# --------------------------------------------------
# TRANSFORM VALIDATION DATA
# --------------------------------------------------

X_validation = vectorizer.transform(
    X_validation_text
)

y_validation = mlb.transform(
    validation_labels
)


# --------------------------------------------------
# GET PROBABILITIES
# --------------------------------------------------

print("=" * 75)
print("LEXIRISK AI - GLOBAL THRESHOLD TUNING")
print("=" * 75)

print("\nGenerating validation probabilities...")


y_prob = model.predict_proba(
    X_validation
)


print(
    "Probability matrix shape:",
    y_prob.shape
)


# --------------------------------------------------
# THRESHOLDS TO TEST
# --------------------------------------------------

thresholds = np.arange(
    0.30,
    0.81,
    0.05,
)


results = []


# --------------------------------------------------
# TEST EACH THRESHOLD
# --------------------------------------------------

for threshold in thresholds:

    y_pred = (
        y_prob >= threshold
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


    predicted_labels_per_sample = (
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
                predicted_labels_per_sample,
        }
    )


# --------------------------------------------------
# CREATE RESULTS TABLE
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 115)
print("THRESHOLD RESULTS")
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
# FIND BEST THRESHOLDS
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
print("BEST VALIDATION THRESHOLDS")
print("=" * 75)


print(
    "\nBest threshold by Micro F1:",
    best_micro["threshold"]
)

print(
    "Micro F1:",
    round(
        best_micro["micro_f1"],
        4,
    )
)

print(
    "Precision:",
    round(
        best_micro["micro_precision"],
        4,
    )
)

print(
    "Recall:",
    round(
        best_micro["micro_recall"],
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