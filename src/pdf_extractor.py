from pathlib import Path

import pymupdf


def extract_text_from_pdf(pdf_path):
    """
    Extract readable text from a PDF file.

    Parameters
    ----------
    pdf_path : str or Path
        Location of the PDF document.

    Returns
    -------
    dict
        Information about the PDF and extracted text.
    """

    pdf_path = Path(pdf_path)

    # Check that the file exists
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    # Check that it is actually a PDF filename
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "The selected file is not a PDF."
        )

    try:
        document = pymupdf.open(pdf_path)

    except Exception as error:
        raise ValueError(
            f"Unable to open PDF: {error}"
        )

    page_texts = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):

        text = page.get_text("text").strip()

        page_texts.append(
            {
                "page_number": page_number,
                "text": text,
            }
        )

    document.close()

    full_text = "\n\n".join(
        page["text"]
        for page in page_texts
        if page["text"]
    )

    empty_pages = sum(
        1
        for page in page_texts
        if not page["text"]
    )

    return {
        "file_name": pdf_path.name,
        "page_count": len(page_texts),
        "empty_pages": empty_pages,
        "character_count": len(full_text),
        "text": full_text,
        "pages": page_texts,
    }


# --------------------------------------------------
# TEST MODE
# --------------------------------------------------

if __name__ == "__main__":

    TEST_FILE = Path(
        "data/raw/sample_contract.pdf"
    )

    try:
        result = extract_text_from_pdf(
            TEST_FILE
        )

        print("=" * 70)
        print("LEXIRISK PDF EXTRACTION TEST")
        print("=" * 70)

        print(
            "\nFile:",
            result["file_name"]
        )

        print(
            "Pages:",
            result["page_count"]
        )

        print(
            "Empty pages:",
            result["empty_pages"]
        )

        print(
            "Extracted characters:",
            result["character_count"]
        )

        print("\n" + "=" * 70)
        print("FIRST 1500 CHARACTERS")
        print("=" * 70)

        print(
            result["text"][:1500]
        )

    except Exception as error:

        print("\nPDF extraction failed:")
        print(error)