import json
import re
from pathlib import Path

import pandas as pd


TRAIN_FILE = Path(
    "data/raw/cuad/train_separate_questions.json"
)

TEST_FILE = Path(
    "data/raw/cuad/test.json"
)

OUTPUT_DIR = Path("data/processed")

TRAIN_OUTPUT = OUTPUT_DIR / "cuad_train_clauses.csv"
TEST_OUTPUT = OUTPUT_DIR / "cuad_test_clauses.csv"


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_category(question):
    """
    Extract the CUAD category written inside quotation marks.

    Example:
    Highlight the parts related to "Governing Law"...
    becomes:
    Governing Law
    """

    match = re.search(r'"([^"]+)"', question)

    if match:
        return match.group(1)

    return question.strip()


def extract_clause_rows(dataset, split_name):
    rows = []

    for contract in dataset.get("data", []):

        contract_title = contract.get("title", "")

        for paragraph in contract.get("paragraphs", []):

            for qa in paragraph.get("qas", []):

                question = qa.get("question", "")
                category = extract_category(question)

                answers = qa.get("answers", [])

                for answer in answers:

                    clause_text = answer.get("text", "").strip()

                    answer_start = answer.get(
                        "answer_start"
                    )

                    rows.append(
                        {
                            "contract_title": contract_title,
                            "category": category,
                            "clause_text": clause_text,
                            "answer_start": answer_start,
                            "split": split_name,
                        }
                    )

    return rows


# Load original CUAD split files
train_data = load_json(TRAIN_FILE)
test_data = load_json(TEST_FILE)

# Convert annotations into rows
train_rows = extract_clause_rows(
    train_data,
    "train"
)

test_rows = extract_clause_rows(
    test_data,
    "test"
)

# Create pandas DataFrames
train_df = pd.DataFrame(train_rows)
test_df = pd.DataFrame(test_rows)

# Make sure processed folder exists
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Save without modifying or cleaning the annotations
train_df.to_csv(
    TRAIN_OUTPUT,
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    TEST_OUTPUT,
    index=False,
    encoding="utf-8"
)


print("=" * 65)
print("CUAD CLAUSE DATASET CREATED")
print("=" * 65)

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Total rows:", len(train_df) + len(test_df))

print("\nTraining columns:")
print(train_df.columns.tolist())

print("\nFirst 5 training rows:")
print(
    train_df[
        [
            "category",
            "clause_text",
            "split",
        ]
    ].head()
)

print("\nSaved files:")
print(TRAIN_OUTPUT)
print(TEST_OUTPUT)