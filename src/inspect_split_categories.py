import json
import re
from pathlib import Path
from collections import defaultdict


TRAIN_FILE = Path(
    "data/raw/cuad/train_separate_questions.json"
)

TEST_FILE = Path(
    "data/raw/cuad/test.json"
)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_category(question):
    match = re.search(r'"([^"]+)"', question)

    if match:
        return match.group(1)

    return question


def get_category_counts(dataset):
    category_counts = defaultdict(
        lambda: {
            "question_records": 0,
            "contracts_with_answer": 0,
            "answer_spans": 0,
        }
    )

    for contract in dataset.get("data", []):

        for paragraph in contract.get("paragraphs", []):

            for qa in paragraph.get("qas", []):

                category = extract_category(
                    qa.get("question", "")
                )

                answers = qa.get("answers", [])

                category_counts[category][
                    "question_records"
                ] += 1

                if answers:
                    category_counts[category][
                        "contracts_with_answer"
                    ] += 1

                category_counts[category][
                    "answer_spans"
                ] += len(answers)

    return category_counts


train_data = load_json(TRAIN_FILE)
test_data = load_json(TEST_FILE)

train_counts = get_category_counts(train_data)
test_counts = get_category_counts(test_data)

all_categories = sorted(
    set(train_counts.keys()) | set(test_counts.keys())
)


print("=" * 95)
print("CUAD TRAIN / TEST CATEGORY DISTRIBUTION")
print("=" * 95)

print(
    f"{'Category':40}"
    f"{'Train Docs':>12}"
    f"{'Test Docs':>12}"
    f"{'Train Spans':>14}"
    f"{'Test Spans':>13}"
)

print("-" * 95)


for category in all_categories:

    train_docs = train_counts[category][
        "contracts_with_answer"
    ]

    test_docs = test_counts[category][
        "contracts_with_answer"
    ]

    train_spans = train_counts[category][
        "answer_spans"
    ]

    test_spans = test_counts[category][
        "answer_spans"
    ]

    print(
        f"{category:40}"
        f"{train_docs:>12}"
        f"{test_docs:>12}"
        f"{train_spans:>14}"
        f"{test_spans:>13}"
    )


print("\n" + "=" * 95)
print("IMPORTANT CHECKS")
print("=" * 95)


missing_in_train = [
    category
    for category in all_categories
    if train_counts[category]["contracts_with_answer"] == 0
]


missing_in_test = [
    category
    for category in all_categories
    if test_counts[category]["contracts_with_answer"] == 0
]


print(
    "Categories with no positive example in train:",
    len(missing_in_train)
)

for category in missing_in_train:
    print(" -", category)


print(
    "\nCategories with no positive example in test:",
    len(missing_in_test)
)

for category in missing_in_test:
    print(" -", category)