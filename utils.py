"""Helper functions for memory parsing and response selection."""

from __future__ import annotations

import random
import re
from typing import Optional


def normalize_name(name: str) -> str:
    """Normalize a detected name for display and storage."""

    cleaned = re.sub(r"\s+", " ", name.strip().strip(".?!,;:"))
    if not cleaned:
        return cleaned
    return " ".join(part[:1].upper() + part[1:].lower() if part else part for part in cleaned.split(" "))


def extract_name(message: str) -> Optional[str]:
    """Extract a name from common self-introduction phrases."""

    if not message:
        return None

    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*)",
        r"\bmy name's\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*)",
        r"\bi am\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*)",
        r"\bi'm\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*)",
        r"\bcall me\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return normalize_name(match.group(1))
    return None


def is_name_query(message: str) -> bool:
    """Detect whether the message asks for the stored name."""

    if not message:
        return False

    lowered = message.lower()
    triggers = [
        "what is my name",
        "what's my name",
        "do you know my name",
        "can you tell me my name",
        "who am i",
        "remember my name",
    ]
    return any(trigger in lowered for trigger in triggers)


def choose_response(responses: list[str]) -> str:
    """Return a random response from a non-empty list."""

    if not responses:
        return "I'm not sure how to respond to that."
    return random.choice(responses)
