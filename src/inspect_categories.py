import json
import re
from pathlib import Path
from collections import defaultdict

DATA_FILE = Path("data/raw/cuad/CUADv1.json")

with open(DATA_FILE, "r", encoding="utf-8") as file:
    cuad = json.load(file)

contracts = cuad.get("data", [])

category_stats = defaultdict(
    lambda: {
        "question_records": 0,
        "contracts_with_answer": 0,
        "total_answer_spans": 0,
    }
)

question_counts_per_contract = []


def extract_category(question):
    """
    Extract the category name inside quotation marks.

    Example:
    Highlight the parts ... related to "Document Name" ...
    becomes:
    Document Name
    """

    match = re.search(r'"([^"]+)"', question)

    if match:
        return match.group(1)

    return question


for contract in contracts:

    paragraphs = contract.get("paragraphs", [])

    contract_question_count = 0

    for paragraph in paragraphs:

        questions = paragraph.get("qas", [])

        contract_question_count += len(questions)

        for question_data in questions:

            question = question_data.get("question", "")
            category = extract_category(question)

            answers = question_data.get("answers", [])

            category_stats[category]["question_records"] += 1

            if len(answers) > 0:
                category_stats[category]["contracts_with_answer"] += 1

            category_stats[category]["total_answer_spans"] += len(answers)

    question_counts_per_contract.append(contract_question_count)


print("=" * 70)
print("CUAD CATEGORY INSPECTION")
print("=" * 70)

print("\nTotal contracts:", len(contracts))

print(
    "Minimum questions in a contract:",
    min(question_counts_per_contract)
)

print(
    "Maximum questions in a contract:",
    max(question_counts_per_contract)
)

print(
    "Number of unique categories:",
    len(category_stats)
)

print("\n" + "=" * 70)
print("CATEGORY STATISTICS")
print("=" * 70)

for number, category in enumerate(sorted(category_stats), start=1):

    stats = category_stats[category]

    print(f"\n{number}. {category}")
    print("   Question records:", stats["question_records"])
    print("   Contracts with answer:", stats["contracts_with_answer"])
    print("   Total answer spans:", stats["total_answer_spans"])


total_question_records = sum(
    stats["question_records"]
    for stats in category_stats.values()
)

total_answer_spans = sum(
    stats["total_answer_spans"]
    for stats in category_stats.values()
)

print("\n" + "=" * 70)
print("OVERALL TOTALS")
print("=" * 70)

print("Total question records:", total_question_records)
print("Total answer spans:", total_answer_spans)