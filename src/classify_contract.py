from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pdf_extractor import extract_text_from_pdf
from clause_segmenter import segment_contract


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PDF_FILE = Path(
    "data/raw/sample_contract.pdf"
)

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

OUTPUT_FILE = (
    OUTPUT_DIR / "sample_contract_predictions.csv"
)


# Final threshold selected during validation
DECISION_THRESHOLD = 0.00


# --------------------------------------------------
# LOAD MODEL COMPONENTS
# --------------------------------------------------

def load_model():

    vectorizer = joblib.load(
        MODEL_DIR / "final_tfidf_vectorizer.joblib"
    )

    model = joblib.load(
        MODEL_DIR / "final_linear_svm.joblib"
    )

    label_binarizer = joblib.load(
        MODEL_DIR / "final_label_binarizer.joblib"
    )

    return (
        vectorizer,
        model,
        label_binarizer,
    )


# --------------------------------------------------
# CLASSIFY CLAUSES
# --------------------------------------------------

def classify_clauses(
    clauses,
    vectorizer,
    model,
    label_binarizer,
):
    """
    Predict zero or more CUAD categories
    for every candidate clause.

    Linear SVM decision scores are margins,
    NOT probabilities.
    """

    if not clauses:
        return []

    clause_texts = [
        clause["text"]
        for clause in clauses
    ]

    X = vectorizer.transform(
        clause_texts
    )

    decision_scores = model.decision_function(
        X
    )

    results = []


    for index, clause in enumerate(
        clauses,
        start=1,
    ):

        scores = decision_scores[
            index - 1
        ]

        predicted_indices = np.where(
            scores >= DECISION_THRESHOLD
        )[0]


        predicted_categories = [
            label_binarizer.classes_[i]
            for i in predicted_indices
        ]


        predicted_scores = [
            float(scores[i])
            for i in predicted_indices
        ]


        # Sort predicted labels from strongest
        # SVM margin to weakest.
        ranked_predictions = sorted(
            zip(
                predicted_categories,
                predicted_scores,
            ),
            key=lambda item: item[1],
            reverse=True,
        )


        # Highest score across ALL 41 classes.
        best_index = int(
            np.argmax(scores)
        )

        top_category = (
            label_binarizer.classes_[
                best_index
            ]
        )

        top_score = float(
            scores[best_index]
        )


        results.append(
            {
                "clause_number":
                    index,

                "main_section":
                    clause["main_section"],

                "subsection":
                    clause["subsection"],

                "heading":
                    clause["heading"],

                "clause_text":
                    clause["text"],

                "predicted_categories":
                    [
                        category
                        for category, score
                        in ranked_predictions
                    ],

                "prediction_scores":
                    [
                        score
                        for category, score
                        in ranked_predictions
                    ],

                "prediction_count":
                    len(ranked_predictions),

                "top_category":
                    top_category,

                "top_decision_score":
                    top_score,
            }
        )


    return results


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 80)
    print("LEXIRISK AI - CONTRACT CLAUSE CLASSIFICATION")
    print("=" * 80)


    # ----------------------------------------------
    # 1. EXTRACT PDF
    # ----------------------------------------------

    pdf_result = extract_text_from_pdf(
        PDF_FILE
    )


    print(
        "\nDocument:",
        pdf_result["file_name"]
    )

    print(
        "Pages:",
        pdf_result["page_count"]
    )


    # ----------------------------------------------
    # 2. SEGMENT CONTRACT
    # ----------------------------------------------

    clauses = segment_contract(
        pdf_result["text"]
    )


    print(
        "Candidate clauses:",
        len(clauses)
    )


    # ----------------------------------------------
    # 3. LOAD FINAL MODEL
    # ----------------------------------------------

    (
        vectorizer,
        model,
        label_binarizer,
    ) = load_model()


    print(
        "CUAD categories:",
        len(label_binarizer.classes_)
    )


    # ----------------------------------------------
    # 4. CLASSIFY
    # ----------------------------------------------

    results = classify_clauses(
        clauses,
        vectorizer,
        model,
        label_binarizer,
    )


    # ----------------------------------------------
    # 5. SUMMARY
    # ----------------------------------------------

    predicted_count = sum(
        1
        for result in results
        if result["prediction_count"] > 0
    )

    no_prediction_count = (
        len(results) - predicted_count
    )


    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)

    print(
        "\nClauses analysed:",
        len(results)
    )

    print(
        "Clauses with >= 1 predicted category:",
        predicted_count
    )

    print(
        "Clauses with no category above threshold:",
        no_prediction_count
    )


    # ----------------------------------------------
    # 6. PRINT FIRST 20 RESULTS
    # ----------------------------------------------

    print("\n" + "=" * 80)
    print("FIRST 20 CLAUSE PREDICTIONS")
    print("=" * 80)


    for result in results[:20]:

        print(
            f"\nCLAUSE "
            f"{result['clause_number']}"
        )

        print("-" * 80)

        print(
            "Section:",
            result["main_section"]
        )

        print(
            "Subsection:",
            result["subsection"]
        )

        print(
            "Text:",
            result["clause_text"][:350]
        )


        if result["prediction_count"] > 0:

            print(
                "Predicted CUAD categories:"
            )

            for category, score in zip(
                result[
                    "predicted_categories"
                ],
                result[
                    "prediction_scores"
                ],
            ):

                print(
                    f"  - {category} "
                    f"(decision score: "
                    f"{score:.4f})"
                )

        else:

            print(
                "Predicted CUAD categories: "
                "None"
            )

            print(
                "Highest-scoring category:",
                result["top_category"]
            )

            print(
                "Highest decision score:",
                round(
                    result[
                        "top_decision_score"
                    ],
                    4,
                )
            )


    # ----------------------------------------------
    # 7. SAVE CSV
    # ----------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    rows = []

    for result in results:

        categories_string = " | ".join(
            result[
                "predicted_categories"
            ]
        )

        scores_string = " | ".join(
            f"{score:.4f}"
            for score in result[
                "prediction_scores"
            ]
        )


        rows.append(
            {
                "clause_number":
                    result["clause_number"],

                "main_section":
                    result["main_section"],

                "subsection":
                    result["subsection"],

                "heading":
                    result["heading"],

                "clause_text":
                    result["clause_text"],

                "predicted_categories":
                    categories_string,

                "decision_scores":
                    scores_string,

                "prediction_count":
                    result["prediction_count"],

                "top_category":
                    result["top_category"],

                "top_decision_score":
                    result["top_decision_score"],
            }
        )


    output_df = pd.DataFrame(
        rows
    )


    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    print("\n" + "=" * 80)
    print("RESULT SAVED")
    print("=" * 80)

    print(OUTPUT_FILE)