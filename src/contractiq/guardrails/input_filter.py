"""
guardrails/input_filter.py
--------------------------
Basic checks on the input query to detect PII and prompt injection.
"""

import re

PII_PATTERNS = {
    "email": re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"),
    "phone": re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
}

INJECTION_PATTERNS = [
    re.compile(r"ignore previous instructions", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"you are (now )?a", re.IGNORECASE),
    re.compile(r"disregard", re.IGNORECASE)
]

def scan_input(query: str) -> dict:
    """
    Scans the query for PII and potential injection attacks.
    Returns a dictionary of flags.
    """
    flags = {
        "has_pii": False,
        "pii_types": [],
        "has_injection": False,
        "injection_matches": []
    }

    for name, pattern in PII_PATTERNS.items():
        if pattern.search(query):
            flags["has_pii"] = True
            flags["pii_types"].append(name)

    for pattern in INJECTION_PATTERNS:
        match = pattern.search(query)
        if match:
            flags["has_injection"] = True
            flags["injection_matches"].append(match.group(0))

    return flags
