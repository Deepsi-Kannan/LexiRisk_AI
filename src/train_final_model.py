from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

TRAIN_FILE = Path(
    "data/processed/cuad_train_multilabel.csv"
)

MODEL_DIR = Path("models")


# --------------------------------------------------
# 1. LOAD FULL ORIGINAL TRAINING DATA
# --------------------------------------------------

train_df = pd.read_csv(
    TRAIN_FILE
)


print("=" * 70)
print("LEXIRISK AI - FINAL MODEL TRAINING")
print("=" * 70)

print(
    "\nTraining samples:",
    len(train_df)
)

print(
    "Training contracts:",
    train_df["contract_title"].nunique()
)


# --------------------------------------------------
# 2. PREPARE TEXT
# --------------------------------------------------

X_train_text = (
    train_df["clause_text"]
    .astype(str)
)


# --------------------------------------------------
# 3. PREPARE LABELS
# --------------------------------------------------

train_labels = [
    value.split("|")
    for value in train_df["labels"].astype(str)
]


mlb = MultiLabelBinarizer()

y_train = mlb.fit_transform(
    train_labels
)


print(
    "Number of categories:",
    len(mlb.classes_)
)


# --------------------------------------------------
# 4. TF-IDF
# --------------------------------------------------

print("\nCreating final TF-IDF features...")


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


print(
    "TF-IDF shape:",
    X_train.shape
)


# --------------------------------------------------
# 5. FINAL LINEAR SVM
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


print("\nTraining final Linear SVM...")


model.fit(
    X_train,
    y_train
)


print("Final training complete.")


# --------------------------------------------------
# 6. SAVE FINAL MODEL COMPONENTS
# --------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


vectorizer_file = (
    MODEL_DIR / "final_tfidf_vectorizer.joblib"
)

model_file = (
    MODEL_DIR / "final_linear_svm.joblib"
)

label_file = (
    MODEL_DIR / "final_label_binarizer.joblib"
)


joblib.dump(
    vectorizer,
    vectorizer_file
)

joblib.dump(
    model,
    model_file
)

joblib.dump(
    mlb,
    label_file
)


print("\n" + "=" * 70)
print("FINAL MODEL SAVED")
print("=" * 70)

print(vectorizer_file)
print(model_file)
print(label_file)

print("\nSelected decision threshold: 0.00")