"""Flask application exposing the chatbot API."""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, request

from db import get_recent_messages, initialize_database, log_conversation, retrieve_memory, store_memory
from model import get_random_response, is_model_ready, predict_intent
from utils import extract_name, is_name_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
initialize_database()


def _json_error(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def generate_chat_response(user_id: str, message: str) -> str:
    """Generate a response using memory first, then the intent model."""

    recent_messages = get_recent_messages(user_id, limit=2)
    logger.info("Recent context for %s: %s", user_id, recent_messages)

    extracted_name = extract_name(message)
    if extracted_name:
        store_memory(user_id, "name", extracted_name)
        return f"Nice to meet you, {extracted_name}!"

    if is_name_query(message):
        stored_name = retrieve_memory(user_id, "name")
        if stored_name:
            return f"Your name is {stored_name}."
        return "I do not know your name yet. Tell me by saying 'My name is ...'."

    intent, confidence = predict_intent(message, threshold=0.7)
    logger.info("Predicted intent for %s: %s (%.3f)", user_id, intent, confidence)

    if intent == "fallback":
        return get_random_response("fallback")

    return get_random_response(intent)


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id", "")).strip()
    message = str(payload.get("message", "")).strip()

    if not user_id:
        return _json_error("'user_id' is required.")
    if not message:
        return _json_error("'message' is required.")

    try:
        log_conversation(user_id, "user", message)
        response_text = generate_chat_response(user_id, message)
        log_conversation(user_id, "assistant", response_text)
        return jsonify({"response": response_text})
    except RuntimeError as exc:
        logger.exception("Model stack unavailable")
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception("Chat request failed")
        return jsonify({"error": "An unexpected error occurred.", "details": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_ready": is_model_ready()})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
