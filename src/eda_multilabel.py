from pathlib import Path
from collections import Counter

import pandas as pd


TRAIN_FILE = Path(
    "data/processed/cuad_train_multilabel.csv"
)

TEST_FILE = Path(
    "data/processed/cuad_test_multilabel.csv"
)


train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


def split_labels(label_string):
    return str(label_string).split("|")


# --------------------------------------------------
# 1. BASIC INFORMATION
# --------------------------------------------------

print("=" * 75)
print("MULTI-LABEL DATASET EDA")
print("=" * 75)

print("\nTrain shape:", train_df.shape)
print("Test shape:", test_df.shape)


# --------------------------------------------------
# 2. UNIQUE LABELS
# --------------------------------------------------

all_train_labels = []

for labels in train_df["labels"]:
    all_train_labels.extend(
        split_labels(labels)
    )

all_test_labels = []

for labels in test_df["labels"]:
    all_test_labels.extend(
        split_labels(labels)
    )


train_unique_labels = sorted(
    set(all_train_labels)
)

test_unique_labels = sorted(
    set(all_test_labels)
)


print("\nUnique train labels:", len(train_unique_labels))
print("Unique test labels:", len(test_unique_labels))


missing_from_test = sorted(
    set(train_unique_labels)
    - set(test_unique_labels)
)

print(
    "Labels present in train but absent from test:",
    len(missing_from_test)
)

for label in missing_from_test:
    print(" -", label)


# --------------------------------------------------
# 3. LABEL FREQUENCIES
# --------------------------------------------------

train_label_counts = Counter(
    all_train_labels
)

test_label_counts = Counter(
    all_test_labels
)


print("\n" + "=" * 75)
print("TRAIN LABEL FREQUENCIES")
print("=" * 75)

for label, count in sorted(
    train_label_counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{label:40} {count}")


print("\n" + "=" * 75)
print("TEST LABEL FREQUENCIES")
print("=" * 75)

for label, count in sorted(
    test_label_counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{label:40} {count}")


# --------------------------------------------------
# 4. NUMBER OF LABELS PER SAMPLE
# --------------------------------------------------

print("\n" + "=" * 75)
print("LABEL COUNT PER SAMPLE")
print("=" * 75)

print("\nTrain:")
print(
    train_df["label_count"]
    .value_counts()
    .sort_index()
)

print("\nTest:")
print(
    test_df["label_count"]
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# 5. LABEL CARDINALITY
# --------------------------------------------------

train_cardinality = (
    train_df["label_count"].mean()
)

test_cardinality = (
    test_df["label_count"].mean()
)


print("\nAverage labels per train sample:")
print(round(train_cardinality, 4))

print("Average labels per test sample:")
print(round(test_cardinality, 4))


# --------------------------------------------------
# 6. TEXT LENGTH ANALYSIS
# --------------------------------------------------

train_df["character_length"] = (
    train_df["clause_text"]
    .astype(str)
    .str.len()
)

train_df["word_length"] = (
    train_df["clause_text"]
    .astype(str)
    .str.split()
    .str.len()
)


print("\n" + "=" * 75)
print("TRAIN CLAUSE LENGTH")
print("=" * 75)

print("\nCharacter length:")
print(
    train_df["character_length"]
    .describe()
)

print("\nWord length:")
print(
    train_df["word_length"]
    .describe()
)


# --------------------------------------------------
# 7. SHORTEST AND LONGEST CLAUSES
# --------------------------------------------------

shortest = train_df.loc[
    train_df["word_length"].idxmin()
]

longest = train_df.loc[
    train_df["word_length"].idxmax()
]


print("\n" + "=" * 75)
print("SHORTEST TRAIN CLAUSE")
print("=" * 75)

print("Labels:", shortest["labels"])
print("Word count:", shortest["word_length"])
print("Text:")
print(shortest["clause_text"])


print("\n" + "=" * 75)
print("LONGEST TRAIN CLAUSE")
print("=" * 75)

print("Labels:", longest["labels"])
print("Word count:", longest["word_length"])
print("Text preview:")
print(
    str(longest["clause_text"])[:1000]
)