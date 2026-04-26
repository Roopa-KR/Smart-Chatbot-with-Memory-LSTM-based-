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


def normalize_phrase(text: str) -> str:
    """Normalize a short remembered phrase for display and storage."""

    cleaned = re.sub(r"\s+", " ", text.strip().strip(".?!,;:"))
    return cleaned.lower()


def normalize_possession(text: str) -> str:
    """Normalize a possession phrase for display and storage."""

    cleaned = re.sub(r"\s+", " ", text.strip().strip(".?!,;:"))
    return cleaned.lower()


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


def extract_preference(message: str) -> Optional[str]:
    """Extract a simple liked thing from messages like 'I like eating'."""

    if not message:
        return None

    patterns = [
        r"\bi like\s+(.+)$",
        r"\bi love\s+(.+)$",
        r"\bi enjoy\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            value = normalize_phrase(match.group(1))
            return value if value else None
    return None


def extract_possession(message: str) -> Optional[str]:
    """Extract a simple possession statement from messages like 'I have GitHub account'."""

    if not message:
        return None

    patterns = [
        r"\bi have\s+(.+)$",
        r"\bi've got\s+(.+)$",
        r"\bi own\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            value = normalize_possession(match.group(1))
            return value if value else None
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


def is_preference_query(message: str) -> bool:
    """Detect whether the message asks what the user likes."""

    if not message:
        return False

    lowered = message.lower()
    triggers = [
        "what do i like",
        "what do i love",
        "what do i enjoy",
        "what do i like?",
        "what do i love?",
        "what do i enjoy?",
    ]
    return any(trigger in lowered for trigger in triggers)


def is_possession_query(message: str) -> bool:
    """Detect whether the message asks what the user has."""

    if not message:
        return False

    lowered = message.lower()
    triggers = [
        "what do i have",
        "what do i own",
        "do i have",
        "what have i got",
        "what accounts do i have",
    ]
    return any(trigger in lowered for trigger in triggers)


def choose_response(responses: list[str]) -> str:
    """Return a random response from a non-empty list."""

    if not responses:
        return "I'm not sure how to respond to that."
    return random.choice(responses)
