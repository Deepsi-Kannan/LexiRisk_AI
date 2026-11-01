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


# Normalise only surrounding whitespace for comparison
train_df["text_clean"] = (
    train_df["clause_text"]
    .astype(str)
    .str.strip()
)

test_df["text_clean"] = (
    test_df["clause_text"]
    .astype(str)
    .str.strip()
)


train_texts = set(train_df["text_clean"])
test_texts = set(test_df["text_clean"])

overlap_texts = train_texts.intersection(test_texts)


print("=" * 80)
print("TRAIN / TEST EXACT TEXT OVERLAP")
print("=" * 80)

print("Number of overlapping texts:", len(overlap_texts))


same_label_count = 0
different_label_count = 0


for text in overlap_texts:

    train_labels = set(
        train_df.loc[
            train_df["text_clean"] == text,
            "category"
        ]
    )

    test_labels = set(
        test_df.loc[
            test_df["text_clean"] == text,
            "category"
        ]
    )

    if train_labels == test_labels:
        same_label_count += 1
    else:
        different_label_count += 1


print(
    "Overlap texts with same label set:",
    same_label_count
)

print(
    "Overlap texts with different label sets:",
    different_label_count
)


print("\n" + "=" * 80)
print("FIRST 20 OVERLAPPING EXAMPLES")
print("=" * 80)


for number, text in enumerate(
    sorted(overlap_texts),
    start=1
):

    train_labels = sorted(
        set(
            train_df.loc[
                train_df["text_clean"] == text,
                "category"
            ]
        )
    )

    test_labels = sorted(
        set(
            test_df.loc[
                test_df["text_clean"] == text,
                "category"
            ]
        )
    )

    print(f"\nExample {number}")
    print("-" * 80)

    print("Text:")
    print(text[:500])

    print("\nTrain labels:")
    print(train_labels)

    print("Test labels:")
    print(test_labels)

    if number == 20:
        break