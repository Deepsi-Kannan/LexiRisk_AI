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
    hamming_loss,
    accuracy_score,
)


TRAIN_FILE = Path(
    "data/processed/cuad_train_multilabel.csv"
)

TEST_FILE = Path(
    "data/processed/cuad_test_multilabel.csv"
)

MODEL_DIR = Path("models")


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


print("=" * 70)
print("LEXIRISK AI - BASELINE MODEL")
print("=" * 70)

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))


# --------------------------------------------------
# 2. PREPARE TEXT
# --------------------------------------------------

X_train_text = train_df["clause_text"].astype(str)
X_test_text = test_df["clause_text"].astype(str)


# --------------------------------------------------
# 3. PREPARE MULTI-LABEL TARGETS
# --------------------------------------------------

train_labels = [
    labels.split("|")
    for labels in train_df["labels"].astype(str)
]

test_labels = [
    labels.split("|")
    for labels in test_df["labels"].astype(str)
]


mlb = MultiLabelBinarizer()

y_train = mlb.fit_transform(train_labels)

y_test = mlb.transform(test_labels)


print("\nNumber of label categories:")
print(len(mlb.classes_))

print("\nLabel matrix shapes:")
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


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

X_test = vectorizer.transform(
    X_test_text
)


print("TF-IDF training shape:", X_train.shape)
print("TF-IDF testing shape:", X_test.shape)


# --------------------------------------------------
# 5. MODEL
# --------------------------------------------------

print("\nTraining Logistic Regression models...")


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
# 6. PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")


y_pred = model.predict(
    X_test
)


# --------------------------------------------------
# 7. EVALUATION
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
    y_pred
)


hamming = hamming_loss(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("BASELINE TEST RESULTS")
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


# --------------------------------------------------
# 8. SAVE MODEL COMPONENTS
# --------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

joblib.dump(
    model,
    MODEL_DIR / "baseline_logistic_model.joblib"
)

joblib.dump(
    mlb,
    MODEL_DIR / "label_binarizer.joblib"
)


print("\n" + "=" * 70)
print("SAVED MODEL FILES")
print("=" * 70)

print(
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

print(
    MODEL_DIR / "baseline_logistic_model.joblib"
)

print(
    MODEL_DIR / "label_binarizer.joblib"
)