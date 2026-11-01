from pathlib import Path

import pandas as pd


TRAIN_FILE = Path(
    "data/processed/cuad_train_clauses.csv"
)

TEST_FILE = Path(
    "data/processed/cuad_test_clauses.csv"
)


train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


print("=" * 70)
print("BASIC SHAPE")
print("=" * 70)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print("\nTrain missing values:")
print(train_df.isna().sum())

print("\nTest missing values:")
print(test_df.isna().sum())


print("\n" + "=" * 70)
print("EMPTY CLAUSE TEXT")
print("=" * 70)

train_empty = (
    train_df["clause_text"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

test_empty = (
    test_df["clause_text"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

print("Empty train clause texts:", train_empty)
print("Empty test clause texts:", test_empty)


print("\n" + "=" * 70)
print("EXACT DUPLICATE ROWS")
print("=" * 70)

print(
    "Train exact duplicate rows:",
    train_df.duplicated().sum()
)

print(
    "Test exact duplicate rows:",
    test_df.duplicated().sum()
)


print("\n" + "=" * 70)
print("DUPLICATE CLAUSE TEXT")
print("=" * 70)

print(
    "Repeated train clause texts:",
    train_df["clause_text"].duplicated().sum()
)

print(
    "Repeated test clause texts:",
    test_df["clause_text"].duplicated().sum()
)


print("\n" + "=" * 70)
print("SAME TEXT WITH MULTIPLE CATEGORIES")
print("=" * 70)

train_multi_category = (
    train_df
    .groupby("clause_text")["category"]
    .nunique()
)

train_multi_category = train_multi_category[
    train_multi_category > 1
]

test_multi_category = (
    test_df
    .groupby("clause_text")["category"]
    .nunique()
)

test_multi_category = test_multi_category[
    test_multi_category > 1
]

print(
    "Train texts assigned to multiple categories:",
    len(train_multi_category)
)

print(
    "Test texts assigned to multiple categories:",
    len(test_multi_category)
)


print("\n" + "=" * 70)
print("TRAIN / TEST TEXT OVERLAP")
print("=" * 70)

train_texts = set(
    train_df["clause_text"]
    .dropna()
    .astype(str)
    .str.strip()
)

test_texts = set(
    test_df["clause_text"]
    .dropna()
    .astype(str)
    .str.strip()
)

overlap = train_texts.intersection(test_texts)

print(
    "Exact clause texts appearing in both train and test:",
    len(overlap)
)


print("\n" + "=" * 70)
print("CATEGORY COUNTS")
print("=" * 70)

print("\nTrain category counts:")
print(
    train_df["category"]
    .value_counts()
    .sort_index()
)

print("\nTest category counts:")
print(
    test_df["category"]
    .value_counts()
    .sort_index()
)