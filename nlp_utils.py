"""NLP preprocessing helpers for the chatbot."""

from __future__ import annotations

import re
from typing import List

import nltk
from nltk.stem import PorterStemmer

_STEMMER = PorterStemmer()
_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    """Tokenize text with a safe fallback when punkt is unavailable."""

    try:
        return nltk.word_tokenize(text)
    except (LookupError, ValueError):
        return _WORD_PATTERN.findall(text)


def preprocess_text(text: str) -> List[str]:
    """Lowercase, tokenize, and stem a piece of text."""

    if not text:
        return []

    tokens = _tokenize(text.lower())
    processed_tokens: List[str] = []
    for token in tokens:
        cleaned = re.sub(r"[^a-z0-9']+", "", token)
        if cleaned:
            processed_tokens.append(_STEMMER.stem(cleaned))
    return processed_tokens


def preprocess_as_string(text: str) -> str:
    """Return the stemmed tokens as a single string for Keras Tokenizer."""

    return " ".join(preprocess_text(text))
