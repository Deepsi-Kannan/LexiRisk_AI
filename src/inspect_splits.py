import json
from pathlib import Path


TRAIN_FILE = Path(
    "data/raw/cuad/train_separate_questions.json"
)

TEST_FILE = Path(
    "data/raw/cuad/test.json"
)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def collect_titles(dataset):
    titles = []

    for item in dataset.get("data", []):
        title = item.get("title")

        if title:
            titles.append(title)

    return titles


def count_questions_and_answers(dataset):
    question_count = 0
    answer_span_count = 0

    for item in dataset.get("data", []):

        for paragraph in item.get("paragraphs", []):

            for question in paragraph.get("qas", []):

                question_count += 1

                answers = question.get("answers", [])
                answer_span_count += len(answers)

    return question_count, answer_span_count


train_data = load_json(TRAIN_FILE)
test_data = load_json(TEST_FILE)


print("=" * 70)
print("TRAIN FILE")
print("=" * 70)

print("Top-level keys:", train_data.keys())
print("Version:", train_data.get("version"))
print("Number of data entries:", len(train_data.get("data", [])))


train_questions, train_answers = (
    count_questions_and_answers(train_data)
)

print("Question records:", train_questions)
print("Answer spans:", train_answers)


print("\n" + "=" * 70)
print("TEST FILE")
print("=" * 70)

print("Top-level keys:", test_data.keys())
print("Version:", test_data.get("version"))
print("Number of data entries:", len(test_data.get("data", [])))


test_questions, test_answers = (
    count_questions_and_answers(test_data)
)

print("Question records:", test_questions)
print("Answer spans:", test_answers)


train_titles = set(collect_titles(train_data))
test_titles = set(collect_titles(test_data))

overlap = train_titles.intersection(test_titles)


print("\n" + "=" * 70)
print("CONTRACT TITLE CHECK")
print("=" * 70)

print("Unique train titles:", len(train_titles))
print("Unique test titles:", len(test_titles))
print("Titles appearing in both:", len(overlap))


if overlap:
    print("\nWARNING: Overlapping titles found:")

    for title in sorted(overlap):
        print("-", title)
else:
    print("\nNo exact contract-title overlap detected.")