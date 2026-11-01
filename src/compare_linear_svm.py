from pathlib import Path

import joblib
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    hamming_loss,
)


TRAIN_FILE = Path(
    "data/processed/cuad_model_train.csv"
)

VALIDATION_FILE = Path(
    "data/processed/cuad_validation.csv"
)

MODEL_DIR = Path("models")


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

X_train_text = train_df["clause_text"].astype(str)
X_validation_text = validation_df["clause_text"].astype(str)


train_labels = [
    value.split("|")
    for value in train_df["labels"].astype(str)
]

validation_labels = [
    value.split("|")
    for value in validation_df["labels"].astype(str)
]


# --------------------------------------------------
# 2. LOAD THE SAME TF-IDF VECTORIZER
# --------------------------------------------------

vectorizer = joblib.load(
    MODEL_DIR / "validation_tfidf_vectorizer.joblib"
)

X_train = vectorizer.transform(
    X_train_text
)

X_validation = vectorizer.transform(
    X_validation_text
)


# --------------------------------------------------
# 3. PREPARE LABELS
# --------------------------------------------------

mlb = MultiLabelBinarizer()

y_train = mlb.fit_transform(
    train_labels
)

y_validation = mlb.transform(
    validation_labels
)


print("=" * 70)
print("LEXIRISK AI - LINEAR SVM VALIDATION")
print("=" * 70)

print("\nTraining samples:", len(train_df))
print("Validation samples:", len(validation_df))
print("Number of categories:", len(mlb.classes_))


# --------------------------------------------------
# 4. BUILD LINEAR SVM
# --------------------------------------------------

base_classifier = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000,
)


model = OneVsRestClassifier(
    base_classifier,
    n_jobs=1,
)


print("\nTraining Linear SVM...")


model.fit(
    X_train,
    y_train
)


print("Training complete.")


# --------------------------------------------------
# 5. PREDICT
# --------------------------------------------------

y_pred = model.predict(
    X_validation
)


# --------------------------------------------------
# 6. METRICS
# --------------------------------------------------

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


print("\n" + "=" * 70)
print("LINEAR SVM VALIDATION RESULTS")
print("=" * 70)

print(
    f"Micro Precision: {micro_precision:.4f}"
)

print(
    f"Micro Recall:    {micro_recall:.4f}"
)

print(
    f"Micro F1:        {micro_f1:.4f}"
)

print(
    f"Macro F1:        {macro_f1:.4f}"
)

print(
    f"Weighted F1:     {weighted_f1:.4f}"
)

print(
    f"Subset Accuracy: {subset_accuracy:.4f}"
)

print(
    f"Hamming Loss:    {hamming:.4f}"
)


# --------------------------------------------------
# 7. SAVE FOR POSSIBLE FINAL USE
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_DIR / "validation_linear_svm.joblib"
)


print("\nSaved:")
print(
    MODEL_DIR / "validation_linear_svm.joblib"
)