from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    hamming_loss,
    precision_recall_fscore_support,
)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

TEST_FILE = Path(
    "data/processed/cuad_test_multilabel.csv"
)

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

SUMMARY_OUTPUT = (
    OUTPUT_DIR / "final_test_metrics.csv"
)

PER_LABEL_OUTPUT = (
    OUTPUT_DIR / "final_test_per_label_metrics.csv"
)


# --------------------------------------------------
# 1. LOAD TEST DATA
# --------------------------------------------------

test_df = pd.read_csv(TEST_FILE)

X_test_text = (
    test_df["clause_text"]
    .astype(str)
)

test_labels = [
    value.split("|")
    for value in test_df["labels"].astype(str)
]


print("=" * 75)
print("LEXIRISK AI - FINAL OFFICIAL TEST EVALUATION")
print("=" * 75)

print(
    "\nTest samples:",
    len(test_df)
)

print(
    "Test contracts:",
    test_df["contract_title"].nunique()
)


# --------------------------------------------------
# 2. LOAD FINAL MODEL
# --------------------------------------------------

vectorizer = joblib.load(
    MODEL_DIR / "final_tfidf_vectorizer.joblib"
)

model = joblib.load(
    MODEL_DIR / "final_linear_svm.joblib"
)

mlb = joblib.load(
    MODEL_DIR / "final_label_binarizer.joblib"
)


print(
    "Model categories:",
    len(mlb.classes_)
)


# --------------------------------------------------
# 3. TRANSFORM TEST DATA
# --------------------------------------------------

X_test = vectorizer.transform(
    X_test_text
)

y_test = mlb.transform(
    test_labels
)


print(
    "Test TF-IDF shape:",
    X_test.shape
)


# --------------------------------------------------
# 4. FINAL PREDICTION
# --------------------------------------------------

decision_scores = model.decision_function(
    X_test
)

# Selected validation threshold = 0.00
y_pred = (
    decision_scores >= 0.00
).astype(int)


# --------------------------------------------------
# 5. OVERALL METRICS
# --------------------------------------------------

micro_precision = precision_score(
    y_test,
    y_pred,
    average="micro",
    zero_division=0,
)

micro_recall = recall_score(
    y_test,
    y_pred,
    average="micro",
    zero_division=0,
)

micro_f1 = f1_score(
    y_test,
    y_pred,
    average="micro",
    zero_division=0,
)

macro_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

subset_accuracy = accuracy_score(
    y_test,
    y_pred,
)

hamming = hamming_loss(
    y_test,
    y_pred,
)

avg_predicted_labels = (
    y_pred.sum(axis=1).mean()
)


print("\n" + "=" * 75)
print("FINAL OFFICIAL TEST RESULTS")
print("=" * 75)

print(
    f"Micro Precision: {micro_precision:.4f}"
)

print(
    f"Micro Recall:    {micro_recall:.4f}"
)

print(
    f"Micro F1:        {micro_f1:.4f}"
)

print()

print(
    f"Macro Precision: {macro_precision:.4f}"
)

print(
    f"Macro Recall:    {macro_recall:.4f}"
)

print(
    f"Macro F1:        {macro_f1:.4f}"
)

print()

print(
    f"Weighted F1:     {weighted_f1:.4f}"
)

print(
    f"Subset Accuracy: {subset_accuracy:.4f}"
)

print(
    f"Hamming Loss:    {hamming:.4f}"
)

print(
    f"Avg Predicted Labels: "
    f"{avg_predicted_labels:.4f}"
)


# --------------------------------------------------
# 6. SAVE OVERALL METRICS
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

summary_df = pd.DataFrame(
    [
        {
            "model":
                "TF-IDF + One-vs-Rest Linear SVM",
            "decision_threshold": 0.00,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "subset_accuracy": subset_accuracy,
            "hamming_loss": hamming,
            "avg_predicted_labels":
                avg_predicted_labels,
            "test_samples": len(test_df),
            "test_contracts":
                test_df["contract_title"].nunique(),
        }
    ]
)

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# --------------------------------------------------
# 7. PER-CATEGORY METRICS
# --------------------------------------------------

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average=None,
        zero_division=0,
    )
)


per_label_df = pd.DataFrame(
    {
        "category": mlb.classes_,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "test_support": support,
    }
)

per_label_df = per_label_df.sort_values(
    by="f1_score",
    ascending=True,
)


per_label_df.to_csv(
    PER_LABEL_OUTPUT,
    index=False,
)


# --------------------------------------------------
# 8. SHOW LOWEST / HIGHEST CATEGORIES
# --------------------------------------------------

print("\n" + "=" * 75)
print("LOWEST 5 F1 CATEGORIES")
print("=" * 75)

print(
    per_label_df[
        [
            "category",
            "f1_score",
            "test_support",
        ]
    ]
    .head(5)
    .to_string(index=False)
)


print("\n" + "=" * 75)
print("HIGHEST 5 F1 CATEGORIES")
print("=" * 75)

print(
    per_label_df[
        [
            "category",
            "f1_score",
            "test_support",
        ]
    ]
    .tail(5)
    .sort_values(
        "f1_score",
        ascending=False,
    )
    .to_string(index=False)
)


print("\n" + "=" * 75)
print("SAVED RESULTS")
print("=" * 75)

print(SUMMARY_OUTPUT)
print(PER_LABEL_OUTPUT)