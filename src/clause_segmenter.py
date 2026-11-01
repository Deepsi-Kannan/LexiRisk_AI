import re
from pathlib import Path

from pdf_extractor import extract_text_from_pdf


# --------------------------------------------------
# BASIC PATTERNS
# --------------------------------------------------

MAIN_SECTION_PATTERN = re.compile(
    r"^\s*(\d{1,3})\.\s+(.+)$"
)

ALPHA_SECTION_PATTERN = re.compile(
    r"^\s*([A-Z])\.\s+(.+)$"
)

NUMBER_ITEM_PATTERN = re.compile(
    r"^\s*(\d+)\)\s+(.+)$"
)

PAREN_ITEM_PATTERN = re.compile(
    r"^\s*\(([A-Za-z0-9]+)\)\s+(.+)$"
)

PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*Page\s+\d+\s*$",
    re.IGNORECASE,
)


# --------------------------------------------------
# MAIN SECTION DETECTION ANYWHERE IN FULL TEXT
# --------------------------------------------------

MAIN_HEADING_ANYWHERE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(\d{1,3})\."
    r"\s*"
    r"("
    r"[A-Z]"
    r"[A-Z0-9 /&,\-'’()\s]{0,100}?"
    r")"
    r"\."
    r"(?=\s|$)"
)


# --------------------------------------------------
# CLEAN LINE
# --------------------------------------------------

def clean_line(line):

    line = line.strip()

    line = re.sub(
        r"\s+",
        " ",
        line,
    )

    return line


# --------------------------------------------------
# NORMALIZE MAIN SECTION BOUNDARIES
# --------------------------------------------------

def normalize_main_sections(text):
    """
    Detect main section headings even if the PDF
    breaks them across lines.

    Examples handled:

    3. TERM.

    3.
    TERM.

    4. EARLY
    TERMINATION.
    """

    def replace_heading(match):

        section_number = (
            match.group(1)
        )

        heading_text = (
            match.group(2)
        )

        heading_text = re.sub(
            r"\s+",
            " ",
            heading_text,
        ).strip()

        return (
            "\n"
            f"{section_number}. "
            f"{heading_text}. "
        )


    normalized = (
        MAIN_HEADING_ANYWHERE_PATTERN.sub(
            replace_heading,
            text,
        )
    )

    return normalized


# --------------------------------------------------
# HEADING DETECTION
# --------------------------------------------------

def looks_like_heading(text):

    text = text.strip()

    if not text:
        return True

    words = text.split()

    if len(words) > 14:
        return False

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if not letters:
        return False

    uppercase_count = sum(
        1
        for char in letters
        if char.isupper()
    )

    uppercase_ratio = (
        uppercase_count
        / len(letters)
    )

    return uppercase_ratio >= 0.80


# --------------------------------------------------
# SPLIT MAIN SECTION HEADING + BODY
# --------------------------------------------------

def split_main_section(
    section_number,
    text,
):

    text = text.strip()

    match = re.match(
        r"^("
        r"[A-Z]"
        r"[A-Z0-9\s/&,\-'’()]+?"
        r"\."
        r")"
        r"\s*"
        r"(.*)$",
        text,
    )


    if match:

        heading_text = (
            match.group(1)
            .strip()
        )

        body_text = (
            match.group(2)
            .strip()
        )

        heading = (
            f"{section_number}. "
            f"{heading_text}"
        )

        return (
            heading,
            body_text,
        )


    return (
        f"{section_number}. {text}",
        "",
    )


# --------------------------------------------------
# SEGMENT CONTRACT
# --------------------------------------------------

def segment_contract(text):

    # ----------------------------------------------
    # STEP 1:
    # Normalize main section headings globally
    # ----------------------------------------------

    text = normalize_main_sections(
        text
    )


    # ----------------------------------------------
    # STEP 2:
    # Convert into cleaned logical lines
    # ----------------------------------------------

    raw_lines = text.splitlines()

    lines = []


    for raw_line in raw_lines:

        line = clean_line(
            raw_line
        )

        if not line:
            continue

        if PAGE_NUMBER_PATTERN.match(
            line
        ):
            continue

        lines.append(
            line
        )


    # ----------------------------------------------
    # SEGMENTATION STATE
    # ----------------------------------------------

    clauses = []

    current_lines = []

    current_main_section = None
    current_subsection = None
    current_heading = None


    # ----------------------------------------------
    # SAVE CURRENT CLAUSE
    # ----------------------------------------------

    def save_current_clause():

        nonlocal current_lines
        nonlocal current_heading

        if not current_lines:
            return

        clause_text = " ".join(
            current_lines
        ).strip()


        if len(clause_text) >= 10:

            clauses.append(
                {
                    "main_section":
                        current_main_section,

                    "subsection":
                        current_subsection,

                    "heading":
                        current_heading,

                    "text":
                        clause_text,
                }
            )


        current_lines = []

        current_heading = None


    # ----------------------------------------------
    # STEP 3:
    # PROCESS EACH LOGICAL LINE
    # ----------------------------------------------

    for line in lines:

        # ------------------------------------------
        # MAIN SECTION
        # ------------------------------------------

        main_match = (
            MAIN_SECTION_PATTERN.match(
                line
            )
        )


        if main_match:

            save_current_clause()


            section_number = (
                main_match.group(1)
            )

            section_content = (
                main_match.group(2)
            )


            heading, body = (
                split_main_section(
                    section_number,
                    section_content,
                )
            )


            current_main_section = (
                heading
            )

            current_subsection = None

            current_heading = (
                heading
            )


            if body:

                current_lines = [
                    body
                ]

            else:

                current_lines = []


            continue


        # ------------------------------------------
        # LETTERED SUBSECTION
        # ------------------------------------------

        alpha_match = (
            ALPHA_SECTION_PATTERN.match(
                line
            )
        )


        if alpha_match:

            save_current_clause()


            letter = (
                alpha_match.group(1)
            )

            subsection_text = (
                alpha_match.group(2)
                .strip()
            )


            complete_subsection = (
                f"{letter}. "
                f"{subsection_text}"
            )


            if looks_like_heading(
                subsection_text
            ):

                current_subsection = (
                    complete_subsection
                )

                current_heading = (
                    f"{letter}."
                )

                current_lines = []

            else:

                current_subsection = (
                    f"{letter}."
                )

                current_heading = (
                    f"{letter}."
                )

                current_lines = [
                    complete_subsection
                ]


            continue


        # ------------------------------------------
        # NUMBERED ITEM
        # ------------------------------------------

        number_match = (
            NUMBER_ITEM_PATTERN.match(
                line
            )
        )


        if number_match:

            save_current_clause()


            number = (
                number_match.group(1)
            )

            item_text = (
                number_match.group(2)
            )


            current_heading = (
                f"{number})"
            )

            current_lines = [
                f"{number}) "
                f"{item_text}"
            ]


            continue


        # ------------------------------------------
        # PAREN ITEM
        # ------------------------------------------

        paren_match = (
            PAREN_ITEM_PATTERN.match(
                line
            )
        )


        if paren_match:

            save_current_clause()


            item_id = (
                paren_match.group(1)
            )

            item_text = (
                paren_match.group(2)
            )


            current_heading = (
                f"({item_id})"
            )

            current_lines = [
                f"({item_id}) "
                f"{item_text}"
            ]


            continue


        # ------------------------------------------
        # NORMAL CONTINUATION
        # ------------------------------------------

        if current_lines:

            current_lines.append(
                line
            )

        else:

            current_lines = [
                line
            ]


    # ----------------------------------------------
    # SAVE FINAL CLAUSE
    # ----------------------------------------------

    save_current_clause()

    return clauses


# --------------------------------------------------
# TEST MODE
# --------------------------------------------------

if __name__ == "__main__":

    PDF_FILE = Path(
        "data/raw/sample_contract.pdf"
    )


    result = extract_text_from_pdf(
        PDF_FILE
    )


    clauses = segment_contract(
        result["text"]
    )


    print("=" * 75)

    print(
        "LEXIRISK CLAUSE SEGMENTATION "
        "TEST - VERSION 7"
    )

    print("=" * 75)


    print(
        "\nDocument:",
        result["file_name"]
    )

    print(
        "Extracted characters:",
        result[
            "character_count"
        ]
    )

    print(
        "Candidate clauses:",
        len(clauses)
    )


    print(
        "\n"
        + "=" * 75
    )

    print(
        "FIRST 30 CANDIDATE CLAUSES"
    )

    print(
        "=" * 75
    )


    for index, clause in enumerate(
        clauses[:30],
        start=1,
    ):

        print(
            f"\nCLAUSE {index}"
        )

        print(
            "-" * 75
        )


        print(
            "Main section:",
            clause[
                "main_section"
            ]
        )

        print(
            "Subsection:",
            clause[
                "subsection"
            ]
        )

        print(
            "Heading:",
            clause[
                "heading"
            ]
        )

        print(
            "Text:"
        )

        print(
            clause[
                "text"
            ][:800]
        )