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


def inspect_multilabel_spans(df, split_name):

    grouped = (
        df.groupby(
            [
                "contract_title",
                "answer_start",
                "clause_text",
            ]
        )["category"]
        .agg(lambda values: sorted(set(values)))
        .reset_index()
    )

    grouped["label_count"] = grouped["category"].apply(len)

    multilabel = grouped[
        grouped["label_count"] > 1
    ]

    print("\n" + "=" * 80)
    print(split_name.upper())
    print("=" * 80)

    print(
        "Original annotation rows:",
        len(df)
    )

    print(
        "Unique contract-position-text spans:",
        len(grouped)
    )

    print(
        "Genuine multi-label spans:",
        len(multilabel)
    )

    if len(multilabel) > 0:

        print(
            "Maximum labels on one span:",
            multilabel["label_count"].max()
        )

        print("\nFirst 10 genuine multi-label examples:")

        for number, (_, row) in enumerate(
            multilabel.head(10).iterrows(),
            start=1
        ):

            print("\n" + "-" * 80)
            print("Example:", number)

            print(
                "Contract:",
                row["contract_title"]
            )

            print(
                "Answer start:",
                row["answer_start"]
            )

            print(
                "Categories:",
                row["category"]
            )

            print("Text:")
            print(
                str(row["clause_text"])[:600]
            )

    return grouped


train_grouped = inspect_multilabel_spans(
    train_df,
    "train"
)

test_grouped = inspect_multilabel_spans(
    test_df,
    "test"
)