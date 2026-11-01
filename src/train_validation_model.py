from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

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


print("=" * 70)
print("LEXIRISK AI - VALIDATION BASELINE")
print("=" * 70)

print("\nTraining samples:", len(train_df))
print("Validation samples:", len(validation_df))

print(
    "Training contracts:",
    train_df["contract_title"].nunique()
)

print(
    "Validation contracts:",
    validation_df["contract_title"].nunique()
)


# --------------------------------------------------
# 2. PREPARE TEXT
# --------------------------------------------------

X_train_text = (
    train_df["clause_text"]
    .astype(str)
)

X_validation_text = (
    validation_df["clause_text"]
    .astype(str)
)


# --------------------------------------------------
# 3. PREPARE MULTI-LABEL TARGETS
# --------------------------------------------------

train_labels = [
    value.split("|")
    for value in train_df["labels"].astype(str)
]

validation_labels = [
    value.split("|")
    for value in validation_df["labels"].astype(str)
]


mlb = MultiLabelBinarizer()

y_train = mlb.fit_transform(
    train_labels
)

y_validation = mlb.transform(
    validation_labels
)


print(
    "\nNumber of categories:",
    len(mlb.classes_)
)


# --------------------------------------------------
# 4. TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=30000,
    sublinear_tf=True,
)


X_train = vectorizer.fit_transform(
    X_train_text
)

X_validation = vectorizer.transform(
    X_validation_text
)


print(
    "Training TF-IDF shape:",
    X_train.shape
)

print(
    "Validation TF-IDF shape:",
    X_validation.shape
)


# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

print("\nTraining model...")


base_classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="liblinear",
    random_state=42,
)


model = OneVsRestClassifier(
    base_classifier,
    n_jobs=1,
)


model.fit(
    X_train,
    y_train
)


print("Training complete.")


# --------------------------------------------------
# 6. DEFAULT 0.5 PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(
    X_validation
)


# --------------------------------------------------
# 7. VALIDATION METRICS
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
    y_pred
)

hamming = hamming_loss(
    y_validation,
    y_pred
)


print("\n" + "=" * 70)
print("VALIDATION RESULTS - DEFAULT THRESHOLD 0.50")
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
# 8. SAVE TUNING MODEL
# --------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


joblib.dump(
    vectorizer,
    MODEL_DIR / "validation_tfidf_vectorizer.joblib"
)

joblib.dump(
    model,
    MODEL_DIR / "validation_logistic_model.joblib"
)

joblib.dump(
    mlb,
    MODEL_DIR / "validation_label_binarizer.joblib"
)


print("\nSaved validation model components.")