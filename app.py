"""Terminal-based chatbot application."""

from __future__ import annotations

import argparse
import logging
from typing import Optional

from db import get_recent_messages, initialize_database, log_conversation, retrieve_memory, store_memory
from model import get_random_response, is_model_ready, predict_intent
from utils import extract_name, is_name_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

initialize_database()


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


def _print_header(user_id: str) -> None:
    print("Smart Chatbot with Memory")
    print(f"User ID: {user_id}")
    print("Type 'quit' or 'exit' to stop.\n")


def chat_once(user_id: str, message: str) -> str:
    """Process one chat turn and persist the exchange."""

    if not user_id:
        raise ValueError("user_id is required")
    if not message:
        raise ValueError("message is required")

    log_conversation(user_id, "user", message)
    response_text = generate_chat_response(user_id, message)
    log_conversation(user_id, "assistant", response_text)
    return response_text


def run_interactive_chat(user_id: str) -> None:
    """Run an interactive terminal session."""

    _print_header(user_id)
    if not is_model_ready():
        print("Warning: TensorFlow model is not ready. Memory and regex replies will still work.")

    while True:
        try:
            message = input("You: ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not message:
            continue

        if message.lower() in {"quit", "exit"}:
            print("Bye.")
            break

        try:
            response_text = chat_once(user_id, message)
            print(f"Bot: {response_text}")
        except Exception as exc:
            logger.exception("Chat request failed")
            print(f"Error: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Chatbot with Memory")
    parser.add_argument("--user-id", default="terminal_user", help="Persistent user identifier for memory storage")
    parser.add_argument("--message", help="Send one message and print the response, then exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.message:
        try:
            print(chat_once(args.user_id, args.message))
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
    else:
        run_interactive_chat(args.user_id)
