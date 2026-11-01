import json
from pathlib import Path

DATA_FILE = Path("data/raw/cuad/CUADv1.json")

with open(DATA_FILE, "r", encoding="utf-8") as file:
    cuad = json.load(file)

print("Top-level type:", type(cuad))
print("Top-level keys:", cuad.keys())

print("\nDataset version:", cuad.get("version"))

contracts = cuad.get("data", [])
print("Number of contracts:", len(contracts))

first_contract = contracts[0]

print("\nFirst contract keys:", first_contract.keys())
print("First contract title:", first_contract.get("title"))

paragraphs = first_contract.get("paragraphs", [])
print("Number of paragraphs in first contract:", len(paragraphs))

if paragraphs:
    first_paragraph = paragraphs[0]

    print("\nFirst paragraph keys:", first_paragraph.keys())

    context = first_paragraph.get("context", "")
    questions = first_paragraph.get("qas", [])

    print("Context length:", len(context))
    print("Number of questions:", len(questions))

    if questions:
        first_question = questions[0]

        print("\nFirst question:")
        print(first_question.get("question"))

        answers = first_question.get("answers", [])

        print("\nNumber of answers:", len(answers))

        if answers:
            print("\nFirst answer:")
            print(answers[0].get("text"))