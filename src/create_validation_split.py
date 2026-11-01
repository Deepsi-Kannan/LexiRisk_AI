from pathlib import Path

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


INPUT_FILE = Path(
    "data/processed/cuad_train_multilabel.csv"
)

OUTPUT_DIR = Path("data/processed")

TRAIN_OUTPUT = OUTPUT_DIR / "cuad_model_train.csv"
VALIDATION_OUTPUT = OUTPUT_DIR / "cuad_validation.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)


print("=" * 70)
print("CREATE CONTRACT-LEVEL TRAIN / VALIDATION SPLIT")
print("=" * 70)

print("\nTotal rows:", len(df))
print(
    "Unique contracts:",
    df["contract_title"].nunique()
)


# --------------------------------------------------
# GROUP-BASED SPLIT
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)


train_indices, validation_indices = next(
    splitter.split(
        X=df,
        groups=df["contract_title"],
    )
)


train_df = df.iloc[
    train_indices
].reset_index(drop=True)

validation_df = df.iloc[
    validation_indices
].reset_index(drop=True)


# --------------------------------------------------
# CONTRACT COUNTS
# --------------------------------------------------

train_contracts = set(
    train_df["contract_title"]
)

validation_contracts = set(
    validation_df["contract_title"]
)

overlap = train_contracts.intersection(
    validation_contracts
)


print("\nTraining rows:", len(train_df))
print(
    "Training contracts:",
    len(train_contracts)
)

print("\nValidation rows:", len(validation_df))
print(
    "Validation contracts:",
    len(validation_contracts)
)

print(
    "\nContracts appearing in both:",
    len(overlap)
)


# --------------------------------------------------
# LABEL CHECK
# --------------------------------------------------

def collect_labels(dataframe):
    labels = set()

    for value in dataframe["labels"]:
        labels.update(
            str(value).split("|")
        )

    return labels


train_labels = collect_labels(
    train_df
)

validation_labels = collect_labels(
    validation_df
)


missing_from_validation = sorted(
    train_labels - validation_labels
)


print(
    "\nUnique training labels:",
    len(train_labels)
)

print(
    "Unique validation labels:",
    len(validation_labels)
)

print(
    "Labels missing from validation:",
    len(missing_from_validation)
)


for label in missing_from_validation:
    print(" -", label)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

train_df.to_csv(
    TRAIN_OUTPUT,
    index=False,
    encoding="utf-8",
)

validation_df.to_csv(
    VALIDATION_OUTPUT,
    index=False,
    encoding="utf-8",
)


print("\nSaved:")
print(TRAIN_OUTPUT)
print(VALIDATION_OUTPUT)