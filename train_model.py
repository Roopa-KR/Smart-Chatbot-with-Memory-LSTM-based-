"""Command-line entrypoint for training the LSTM intent model."""

from __future__ import annotations

import argparse

from model import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the chatbot intent classifier.")
    parser.add_argument("--force", action="store_true", help="Retrain even if saved artifacts already exist.")
    args = parser.parse_args()

    train_model(force_retrain=args.force)
    print("Training complete. Saved model and tokenizer artifacts.")


if __name__ == "__main__":
    main()
