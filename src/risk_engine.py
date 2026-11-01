import re


# --------------------------------------------------
# CUSTOM RISK REVIEW RULES - VERSION 2
# --------------------------------------------------
#
# IMPORTANT:
# These are project-specific transparent heuristics.
# They are NOT CUAD labels.
# They do NOT provide legal advice.
# --------------------------------------------------


RISK_RULES = [

    # ==================================================
    # HIGH REVIEW PRIORITY
    # ==================================================

    {
        "rule_id": "R001",
        "name": "Potential Unlimited Liability",
        "risk_level": "HIGH",
        "patterns": [
            r"\bunlimited liability\b",
            r"\bliability shall not be limited\b",
            r"\bliability is not limited\b",
            r"\bwithout limitation\b.{0,80}\bliability\b",
        ],
        "explanation":
            "The wording may expose a party to liability without a clear contractual limit.",
    },

    {
        "rule_id": "R002",
        "name": "Broad Indemnification",
        "risk_level": "HIGH",
        "patterns": [
            r"\bindemnify\b",
            r"\bhold harmless\b",
            r"\bdefend and hold harmless\b",
            r"\bexonerate\b.{0,80}\bindemnify\b",
        ],
        "explanation":
            "The clause contains indemnification language that may transfer significant responsibility between parties.",
    },

    {
        "rule_id": "R003",
        "name": "Non-Compete Restriction",
        "risk_level": "HIGH",
        "patterns": [
            r"\bnon[- ]compete\b",
            r"\bshall not compete\b",
            r"\bnot engage in\b.{0,80}\bcompeting\b",
            r"\bcompetitive business\b",
        ],
        "explanation":
            "The clause may restrict a party from competing or conducting certain business activities.",
    },

    {
        "rule_id": "R004",
        "name": "Exclusive Relationship",
        "risk_level": "HIGH",
        "patterns": [
            r"\bexclusive agreement\b",
            r"\bexclusive right\b",
            r"\bexclusively\b",
            r"\bshall deal exclusively\b",
        ],
        "explanation":
            "The clause may create an exclusive commercial relationship.",
    },


    # ==================================================
    # MEDIUM REVIEW PRIORITY
    # ==================================================

    {
        "rule_id": "R005",
        "name": "Termination for Convenience",
        "risk_level": "MEDIUM",
        "patterns": [
            r"\bterminate\b.{0,80}\bfor (?:its|their) convenience\b",
            r"\btermination for convenience\b",
            r"\bterminate this agreement\b.{0,80}\bany time\b",
        ],
        "explanation":
            "The agreement may allow termination without requiring a contractual breach.",
    },

    {
        "rule_id": "R006",
        "name": "Termination Notice Requirement",
        "risk_level": "MEDIUM",
        "patterns": [
            r"\b\d+[- ]day\b.{0,80}\bwritten notice\b",
            r"\b\d+\s+days?\b.{0,80}\badvance written notice\b",
            r"\bwritten notice\b.{0,80}\bterminate\b",
        ],
        "explanation":
            "The clause contains a notice requirement related to termination.",
    },

    {
        "rule_id": "R007",
        "name": "Assignment Restriction",
        "risk_level": "MEDIUM",
        "patterns": [
            r"\bshall not assign\b",
            r"\bmay not assign\b",
            r"\bassignment\b.{0,100}\b(?:prior )?written consent\b",
            r"\bassign\b.{0,100}\b(?:prior )?written consent\b",
        ],
        "explanation":
            "The clause may restrict assignment or transfer of contractual rights.",
    },

    {
        "rule_id": "R008",
        "name": "Automatic Renewal",
        "risk_level": "MEDIUM",
        "patterns": [
            r"\bautomatically renew\b",
            r"\bautomatic renewal\b",
            r"\bautomatically extended\b",
            r"\brenewal term\b",
        ],
        "explanation":
            "The agreement may renew automatically unless action is taken.",
    },

    {
        "rule_id": "R009",
        "name": "Audit or Inspection Rights",
        "risk_level": "MEDIUM",
        "patterns": [
            r"\bpermit\b.{0,100}\b(?:review|inspect|audit|examine)\b",
            r"\ballow\b.{0,100}\b(?:review|inspect|audit|examine)\b",
            r"\b(?:review|inspect|audit|examine)\b.{0,100}\b(?:records|books|documents|files|accounts|activities)\b",
            r"\baccess to\b.{0,80}\b(?:records|books|documents|files)\b",
        ],
        "explanation":
            "The clause may provide another party with audit, inspection, or document-review rights.",
    },

    {
        "rule_id": "R013",
        "name": "Broad Cost Liability",
        "risk_level": "MEDIUM",
        "patterns": [
            r"\bliable for any and all reasonable costs\b",
            r"\bliable for all reasonable costs\b",
        ],
        "explanation":
            "The clause may impose broad responsibility for costs arising from default or breach.",
    },


    # ==================================================
    # LOW REVIEW PRIORITY
    # ==================================================

    {
        "rule_id": "R010",
        "name": "Payment Obligation",
        "risk_level": "LOW",
        "patterns": [
            r"\bpayment\b",
            r"\bshall pay\b",
            r"\breimburs",
            r"\binvoice\b",
        ],
        "explanation":
            "The clause contains payment or reimbursement obligations that should be reviewed.",
    },

    {
        "rule_id": "R011",
        "name": "Insurance Requirement",
        "risk_level": "LOW",
        "patterns": [
            r"\b(?:obtain|maintain|provide|carry)\b.{0,100}\binsurance\b",
            r"\binsurance coverage\b",
            r"\binsurance policy\b",
            r"\binsurance policies\b",
            r"\bcertificate(?:s)? of insurance\b",
            r"\badditional insured\b",
        ],
        "explanation":
            "The clause contains an insurance or coverage requirement.",
    },

    {
        "rule_id": "R012",
        "name": "Contract Term or Expiration",
        "risk_level": "LOW",
        "patterns": [
            r"\bagreement shall end\b",
            r"\bagreement shall take effect\b",
            r"\bexpiration date\b",
            r"\beffective date\b",
        ],
        "explanation":
            "The clause defines an important effective-date, term, or expiration condition.",
    },
]


# --------------------------------------------------
# RISK PRIORITY
# --------------------------------------------------

RISK_PRIORITY = {
    "NONE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


# --------------------------------------------------
# ANALYSE ONE CLAUSE
# --------------------------------------------------

def analyze_clause_risk(clause_text):

    matched_rules = []

    for rule in RISK_RULES:

        matched_pattern = None

        for pattern in rule["patterns"]:

            match = re.search(
                pattern,
                clause_text,
                flags=re.IGNORECASE,
            )

            if match:

                matched_pattern = (
                    match.group(0)
                )

                break


        if matched_pattern:

            matched_rules.append(
                {
                    "rule_id":
                        rule["rule_id"],

                    "rule_name":
                        rule["name"],

                    "risk_level":
                        rule["risk_level"],

                    "matched_text":
                        matched_pattern,

                    "explanation":
                        rule["explanation"],
                }
            )


    if not matched_rules:

        overall_level = "NONE"

    else:

        overall_level = max(
            matched_rules,
            key=lambda item:
                RISK_PRIORITY[
                    item["risk_level"]
                ],
        )["risk_level"]


    return {
        "risk_level":
            overall_level,

        "rule_count":
            len(matched_rules),

        "matched_rules":
            matched_rules,
    }


# --------------------------------------------------
# SIMPLE TEST
# --------------------------------------------------

if __name__ == "__main__":

    sample_clauses = [

        (
            "COMMISSION may terminate this Agreement "
            "for its convenience at any time by giving "
            "thirty-day written notice."
        ),

        (
            "CONSULTANT shall indemnify, defend, and "
            "hold harmless the COMMISSION from claims."
        ),

        (
            "The CONSULTANT shall submit invoices "
            "monthly for payment."
        ),

        (
            "The parties will meet when necessary "
            "to discuss project progress."
        ),

        (
            "The CONSULTANT shall not assign the "
            "Agreement without the prior written "
            "consent of the COMMISSION."
        ),

        (
            "The CONSULTANT shall permit the COMMISSION "
            "to review and inspect project files."
        ),
    ]


    print("=" * 75)
    print("LEXIRISK AI - RISK ENGINE TEST VERSION 2")
    print("=" * 75)


    for index, clause in enumerate(
        sample_clauses,
        start=1,
    ):

        result = analyze_clause_risk(
            clause
        )

        print(
            f"\nTEST CLAUSE {index}"
        )

        print("-" * 75)

        print(
            "Text:",
            clause
        )

        print(
            "Risk level:",
            result["risk_level"]
        )

        print(
            "Matched rules:",
            result["rule_count"]
        )


        for rule in result[
            "matched_rules"
        ]:

            print(
                f"  - {rule['rule_id']} | "
                f"{rule['rule_name']} | "
                f"{rule['risk_level']}"
            )

            print(
                "    Matched text:",
                rule["matched_text"]
            )

            print(
                "    Reason:",
                rule["explanation"]
            )