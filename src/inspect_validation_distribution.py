from pathlib import Path
from collections import Counter

import pandas as pd


TRAIN_FILE = Path(
    "data/processed/cuad_model_train.csv"
)

VALIDATION_FILE = Path(
    "data/processed/cuad_validation.csv"
)


train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)


def count_labels(df):
    counter = Counter()

    for label_string in df["labels"]:
        labels = str(label_string).split("|")

        for label in labels:
            counter[label] += 1

    return counter


train_counts = count_labels(train_df)
validation_counts = count_labels(validation_df)

all_labels = sorted(
    set(train_counts.keys())
    | set(validation_counts.keys())
)


print("=" * 85)
print("MODEL TRAIN / VALIDATION LABEL DISTRIBUTION")
print("=" * 85)

print(
    f"{'Category':40}"
    f"{'Train':>12}"
    f"{'Validation':>15}"
)

print("-" * 85)


for label in all_labels:
    print(
        f"{label:40}"
        f"{train_counts[label]:>12}"
        f"{validation_counts[label]:>15}"
    )


print("\n" + "=" * 85)
print("VALIDATION SUPPORT CHECK")
print("=" * 85)


very_rare = [
    label
    for label in all_labels
    if validation_counts[label] < 5
]


print(
    "Labels with fewer than 5 validation examples:",
    len(very_rare)
)

for label in very_rare:
    print(
        f" - {label}: "
        f"{validation_counts[label]}"
    )


print(
    "\nMinimum validation support:",
    min(validation_counts.values())
)

print(
    "Maximum validation support:",
    max(validation_counts.values())
)