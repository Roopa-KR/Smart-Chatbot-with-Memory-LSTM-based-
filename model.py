"""Model training and inference for intent classification."""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import numpy as np
    from tensorflow.keras.layers import Dense, Embedding, LSTM
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    _ML_IMPORT_ERROR = None
except Exception as import_error:  # pragma: no cover - exercised only when deps are missing
    np = None  # type: ignore[assignment]
    Dense = Embedding = LSTM = Sequential = load_model = pad_sequences = Tokenizer = None  # type: ignore[assignment]
    _ML_IMPORT_ERROR = import_error

from nlp_utils import preprocess_as_string, preprocess_text

BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "intents.json"
MODEL_PATH = BASE_DIR / "chatbot_lstm_model.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

_MODEL = None
_TOKENIZER: Tokenizer | None = None
_LABELS: List[str] = []
_LABEL_TO_INDEX: Dict[str, int] = {}
_INDEX_TO_LABEL: Dict[int, str] = {}
_MAX_SEQUENCE_LENGTH = 0


def _require_ml_stack() -> None:
    if _ML_IMPORT_ERROR is not None:
        raise RuntimeError(
            "TensorFlow/Keras and NumPy are required for model training and prediction. "
            "Install the project dependencies from requirements.txt in a Python environment "
            "that supports TensorFlow."
        ) from _ML_IMPORT_ERROR


def load_intents() -> dict:
    with open(INTENTS_PATH, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _build_training_corpus(intents: dict) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    for intent in intents.get("intents", []):
        tag = intent["tag"]
        for pattern in intent.get("patterns", []):
            processed = preprocess_as_string(pattern)
            if processed:
                texts.append(processed)
                labels.append(tag)
    return texts, labels


def _score_pattern_match(message_tokens: List[str], pattern_tokens: List[str]) -> float:
    """Score overlap between a message and a training pattern."""

    if not message_tokens or not pattern_tokens:
        return 0.0

    message_token_set = set(message_tokens)
    pattern_token_set = set(pattern_tokens)
    overlap = len(message_token_set & pattern_token_set)
    return overlap / max(len(pattern_token_set), 1)


def _heuristic_predict_intent(message: str) -> Tuple[str, float]:
    """Predict an intent without TensorFlow by matching against training patterns."""

    intents = load_intents()
    message_tokens = preprocess_text(message)
    if not message_tokens:
        return "fallback", 0.0

    best_intent = "fallback"
    best_score = 0.0

    for intent in intents.get("intents", []):
        tag = intent.get("tag", "fallback")
        if tag == "fallback":
            continue

        for pattern in intent.get("patterns", []):
            pattern_tokens = preprocess_text(pattern)
            score = _score_pattern_match(message_tokens, pattern_tokens)
            if score > best_score:
                best_score = score
                best_intent = tag

    if best_score < 0.25:
        return "fallback", best_score
    return best_intent, best_score


def _encode_labels(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    unique_labels = sorted(set(labels))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    y = np.eye(len(unique_labels), dtype=np.float32)[[label_to_index[label] for label in labels]]
    return y, label_to_index, index_to_label


def _create_model(vocab_size: int, max_sequence_length: int, num_classes: int) -> Sequential:
    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=64, input_length=max_sequence_length),
            LSTM(32),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(force_retrain: bool = False) -> Sequential:
    """Train the intent classifier and persist all required artifacts."""

    _require_ml_stack()

    global _MODEL, _TOKENIZER, _LABELS, _LABEL_TO_INDEX, _INDEX_TO_LABEL, _MAX_SEQUENCE_LENGTH

    if MODEL_PATH.exists() and TOKENIZER_PATH.exists() and METADATA_PATH.exists() and not force_retrain:
        return load_trained_artifacts()

    intents = load_intents()
    texts, labels = _build_training_corpus(intents)
    if not texts:
        raise ValueError("No training data found in intents.json")

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_sequence_length = max(len(sequence) for sequence in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="post")

    y, label_to_index, index_to_label = _encode_labels(labels)
    model = _create_model(len(tokenizer.word_index) + 1, max_sequence_length, len(label_to_index))

    model.fit(
        padded_sequences,
        y,
        epochs=200,
        verbose=0,
        shuffle=True,
    )

    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as file_handle:
        pickle.dump(tokenizer, file_handle)
    with open(METADATA_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(
            {
                "labels": sorted(label_to_index.keys()),
                "label_to_index": label_to_index,
                "index_to_label": {str(index): label for index, label in index_to_label.items()},
                "max_sequence_length": max_sequence_length,
            },
            file_handle,
            indent=2,
        )

    _MODEL = model
    _TOKENIZER = tokenizer
    _LABELS = sorted(label_to_index.keys())
    _LABEL_TO_INDEX = label_to_index
    _INDEX_TO_LABEL = index_to_label
    _MAX_SEQUENCE_LENGTH = max_sequence_length
    return model


def load_trained_artifacts() -> Sequential:
    """Load the model, tokenizer, and metadata from disk."""

    _require_ml_stack()

    global _MODEL, _TOKENIZER, _LABELS, _LABEL_TO_INDEX, _INDEX_TO_LABEL, _MAX_SEQUENCE_LENGTH

    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL

    if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists() or not METADATA_PATH.exists():
        return train_model(force_retrain=True)

    _MODEL = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as file_handle:
        _TOKENIZER = pickle.load(file_handle)
    with open(METADATA_PATH, "r", encoding="utf-8") as file_handle:
        metadata = json.load(file_handle)

    _LABELS = metadata["labels"]
    _LABEL_TO_INDEX = {label: int(index) for label, index in metadata["label_to_index"].items()}
    _INDEX_TO_LABEL = {int(index): label for index, label in metadata["index_to_label"].items()}
    _MAX_SEQUENCE_LENGTH = int(metadata["max_sequence_length"])
    return _MODEL


def _ensure_artifacts_loaded() -> None:
    if _MODEL is None or _TOKENIZER is None or not _LABEL_TO_INDEX:
        load_trained_artifacts()


def predict_intent(message: str, threshold: float = 0.7) -> Tuple[str, float]:
    """Predict the most likely intent and confidence for a message."""

    if _ML_IMPORT_ERROR is not None:
        return _heuristic_predict_intent(message)

    _ensure_artifacts_loaded()
    assert _MODEL is not None
    assert _TOKENIZER is not None

    processed_message = preprocess_as_string(message)
    if not processed_message:
        return "fallback", 0.0

    sequence = _TOKENIZER.texts_to_sequences([processed_message])
    padded_sequence = pad_sequences(sequence, maxlen=_MAX_SEQUENCE_LENGTH, padding="post")
    probabilities = _MODEL.predict(padded_sequence, verbose=0)[0]
    top_index = int(np.argmax(probabilities))
    confidence = float(probabilities[top_index])
    intent = _INDEX_TO_LABEL.get(top_index, "fallback")

    if confidence < threshold:
        return "fallback", confidence
    return intent, confidence


def get_random_response(intent_tag: str) -> str:
    """Return a random response for a predicted intent tag."""

    intents = load_intents()
    for intent in intents.get("intents", []):
        if intent.get("tag") == intent_tag:
            responses = intent.get("responses", [])
            if responses:
                return random.choice(responses)
    for intent in intents.get("intents", []):
        if intent.get("tag") == "fallback":
            return random.choice(intent.get("responses", ["I am not sure how to respond."]))
    return "I am not sure how to respond."


def is_model_ready() -> bool:
    """Return True when the ML stack is available and artifacts can be loaded."""

    if _ML_IMPORT_ERROR is not None:
        return False
    try:
        _ensure_artifacts_loaded()
        return True
    except Exception:
        return False
