from pathlib import Path

import pandas as pd


TRAIN_FILE = Path(
    "data/processed/cuad_train_clauses.csv"
)

TEST_FILE = Path(
    "data/processed/cuad_test_clauses.csv"
)

OUTPUT_DIR = Path("data/processed")

TRAIN_OUTPUT = OUTPUT_DIR / "cuad_train_multilabel.csv"
TEST_OUTPUT = OUTPUT_DIR / "cuad_test_multilabel.csv"


def build_multilabel_dataset(df):
    """
    Combine annotations that refer to the exact same
    clause span inside the same contract.

    One row will contain:
    - contract title
    - answer position
    - clause text
    - one or more CUAD labels
    """

    grouped = (
        df.groupby(
            [
                "contract_title",
                "answer_start",
                "clause_text",
            ],
            as_index=False,
        )["category"]
        .agg(
            lambda values: "|".join(
                sorted(set(values))
            )
        )
    )

    grouped = grouped.rename(
        columns={
            "category": "labels"
        }
    )

    grouped["label_count"] = (
        grouped["labels"]
        .str.split("|", regex=False)
        .str.len()
    )

    return grouped


train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

train_multilabel = build_multilabel_dataset(train_df)
test_multilabel = build_multilabel_dataset(test_df)

train_multilabel["split"] = "train"
test_multilabel["split"] = "test"

train_multilabel.to_csv(
    TRAIN_OUTPUT,
    index=False,
    encoding="utf-8"
)

test_multilabel.to_csv(
    TEST_OUTPUT,
    index=False,
    encoding="utf-8"
)


print("=" * 70)
print("MULTI-LABEL DATASET CREATED")
print("=" * 70)

print("\nTraining rows:")
print(len(train_multilabel))

print("Testing rows:")
print(len(test_multilabel))

print("\nTraining multi-label rows:")
print(
    (
        train_multilabel["label_count"] > 1
    ).sum()
)

print("Testing multi-label rows:")
print(
    (
        test_multilabel["label_count"] > 1
    ).sum()
)

print("\nMaximum labels in one training row:")
print(
    train_multilabel["label_count"].max()
)

print("\nMaximum labels in one testing row:")
print(
    test_multilabel["label_count"].max()
)

print("\nFirst 5 rows:")
print(
    train_multilabel[
        [
            "labels",
            "label_count",
            "clause_text",
        ]
    ].head()
)

print("\nSaved files:")
print(TRAIN_OUTPUT)
print(TEST_OUTPUT)