from pathlib import Path

import pandas as pd

from pdf_extractor import extract_text_from_pdf
from clause_segmenter import segment_contract
from classify_contract import (
    load_model,
    classify_clauses,
)
from risk_engine import analyze_clause_risk


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PDF_FILE = Path(
    "data/raw/sample_contract.pdf"
)

OUTPUT_DIR = Path(
    "outputs"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "sample_contract_full_analysis.csv"
)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 80)
    print(
        "LEXIRISK AI - FULL CONTRACT ANALYSIS"
    )
    print("=" * 80)


    # --------------------------------------------------
    # 1. PDF EXTRACTION
    # --------------------------------------------------

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


    # --------------------------------------------------
    # 2. CLAUSE SEGMENTATION
    # --------------------------------------------------

    clauses = segment_contract(
        pdf_result["text"]
    )


    print(
        "Candidate clauses:",
        len(clauses)
    )


    # --------------------------------------------------
    # 3. LOAD FINAL ML MODEL
    # --------------------------------------------------

    (
        vectorizer,
        model,
        label_binarizer,
    ) = load_model()


    # --------------------------------------------------
    # 4. CUAD CATEGORY CLASSIFICATION
    # --------------------------------------------------

    classification_results = (
        classify_clauses(
            clauses,
            vectorizer,
            model,
            label_binarizer,
        )
    )


    # --------------------------------------------------
    # 5. RISK ANALYSIS
    # --------------------------------------------------

    combined_results = []


    for result in classification_results:

        risk_result = (
            analyze_clause_risk(
                result["clause_text"]
            )
        )


        matched_rule_ids = [
            rule["rule_id"]
            for rule in risk_result[
                "matched_rules"
            ]
        ]


        matched_rule_names = [
            rule["rule_name"]
            for rule in risk_result[
                "matched_rules"
            ]
        ]


        matched_explanations = [
            rule["explanation"]
            for rule in risk_result[
                "matched_rules"
            ]
        ]


        matched_texts = [
            rule["matched_text"]
            for rule in risk_result[
                "matched_rules"
            ]
        ]


        combined_results.append(
            {
                "clause_number":
                    result[
                        "clause_number"
                    ],

                "main_section":
                    result[
                        "main_section"
                    ],

                "subsection":
                    result[
                        "subsection"
                    ],

                "heading":
                    result[
                        "heading"
                    ],

                "clause_text":
                    result[
                        "clause_text"
                    ],

                # ------------------------------
                # CUAD ML RESULT
                # ------------------------------

                "cuad_categories":
                    result[
                        "predicted_categories"
                    ],

                "cuad_scores":
                    result[
                        "prediction_scores"
                    ],

                "cuad_prediction_count":
                    result[
                        "prediction_count"
                    ],

                "top_cuad_category":
                    result[
                        "top_category"
                    ],

                "top_decision_score":
                    result[
                        "top_decision_score"
                    ],

                # ------------------------------
                # CUSTOM RISK RESULT
                # ------------------------------

                "risk_level":
                    risk_result[
                        "risk_level"
                    ],

                "risk_rule_count":
                    risk_result[
                        "rule_count"
                    ],

                "risk_rule_ids":
                    matched_rule_ids,

                "risk_rule_names":
                    matched_rule_names,

                "risk_matches":
                    matched_texts,

                "risk_explanations":
                    matched_explanations,
            }
        )


    # --------------------------------------------------
    # 6. SUMMARY COUNTS
    # --------------------------------------------------

    total_clauses = len(
        combined_results
    )


    high_count = sum(
        1
        for result
        in combined_results
        if result["risk_level"]
        == "HIGH"
    )


    medium_count = sum(
        1
        for result
        in combined_results
        if result["risk_level"]
        == "MEDIUM"
    )


    low_count = sum(
        1
        for result
        in combined_results
        if result["risk_level"]
        == "LOW"
    )


    none_count = sum(
        1
        for result
        in combined_results
        if result["risk_level"]
        == "NONE"
    )


    ml_prediction_count = sum(
        1
        for result
        in combined_results
        if result[
            "cuad_prediction_count"
        ] > 0
    )


    # --------------------------------------------------
    # 7. PRINT SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)


    print(
        "\nTotal clauses:",
        total_clauses
    )

    print(
        "Clauses with CUAD prediction:",
        ml_prediction_count
    )


    print("\nRisk review indicators:")

    print(
        "HIGH:",
        high_count
    )

    print(
        "MEDIUM:",
        medium_count
    )

    print(
        "LOW:",
        low_count
    )

    print(
        "NONE:",
        none_count
    )


    # --------------------------------------------------
    # 8. PRINT FLAGGED CLAUSES
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("FLAGGED CLAUSES")
    print("=" * 80)


    for result in combined_results:

        if result["risk_level"] == "NONE":
            continue


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
            "Risk level:",
            result["risk_level"]
        )


        print(
            "Text:",
            result[
                "clause_text"
            ][:400]
        )


        if result[
            "cuad_prediction_count"
        ] > 0:

            print(
                "CUAD categories:",
                ", ".join(
                    result[
                        "cuad_categories"
                    ]
                )
            )

        else:

            print(
                "CUAD categories: None"
            )


        print(
            "Matched risk rules:",
            ", ".join(
                result[
                    "risk_rule_names"
                ]
            )
        )


        for explanation in result[
            "risk_explanations"
        ]:

            print(
                "Reason:",
                explanation
            )


    # --------------------------------------------------
    # 9. SAVE CSV
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    csv_rows = []


    for result in combined_results:

        csv_rows.append(
            {
                "clause_number":
                    result[
                        "clause_number"
                    ],

                "main_section":
                    result[
                        "main_section"
                    ],

                "subsection":
                    result[
                        "subsection"
                    ],

                "heading":
                    result[
                        "heading"
                    ],

                "clause_text":
                    result[
                        "clause_text"
                    ],

                "cuad_categories":
                    " | ".join(
                        result[
                            "cuad_categories"
                        ]
                    ),

                "cuad_scores":
                    " | ".join(
                        f"{score:.4f}"
                        for score
                        in result[
                            "cuad_scores"
                        ]
                    ),

                "top_cuad_category":
                    result[
                        "top_cuad_category"
                    ],

                "top_decision_score":
                    result[
                        "top_decision_score"
                    ],

                "risk_level":
                    result[
                        "risk_level"
                    ],

                "risk_rule_count":
                    result[
                        "risk_rule_count"
                    ],

                "risk_rule_ids":
                    " | ".join(
                        result[
                            "risk_rule_ids"
                        ]
                    ),

                "risk_rule_names":
                    " | ".join(
                        result[
                            "risk_rule_names"
                        ]
                    ),

                "risk_matches":
                    " | ".join(
                        result[
                            "risk_matches"
                        ]
                    ),

                "risk_explanations":
                    " | ".join(
                        result[
                            "risk_explanations"
                        ]
                    ),
            }
        )


    output_df = pd.DataFrame(
        csv_rows
    )


    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    print("\n" + "=" * 80)
    print("RESULT SAVED")
    print("=" * 80)

    print(
        OUTPUT_FILE
    )


    print(
        "\nNOTE:"
    )

    print(
        "Risk levels are transparent review "
        "indicators, not legal advice."
    )

    print(
        "CUAD predictions and custom risk "
        "rules are separate components."
    )