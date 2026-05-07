import json
import re
import os
from dataclasses import dataclass
from typing import List

# Force CPU-only execution
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


START_TOKEN = "<start>"
END_TOKEN = "<end>"
OOV_TOKEN = "<unk>"


def clean_text(text: str) -> str:
    """Lowercase and keep only basic characters."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\?\.\!,]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_dataset(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return records


@dataclass
class PreparedData:
    tokenizer: Tokenizer
    encoder_input_data: np.ndarray
    decoder_input_data: np.ndarray
    decoder_target_data: np.ndarray
    max_encoder_len: int
    max_decoder_len: int
    num_tokens: int


def build_training_matrices(dataset: List[dict], num_words: int = 5000) -> PreparedData:
    """Convert raw question-answer pairs to padded encoder/decoder matrices."""
    cleaned_inputs = [clean_text(item["input"]) for item in dataset]
    cleaned_outputs = [clean_text(item["output"]) for item in dataset]
    decoder_texts = [f"{START_TOKEN} {txt} {END_TOKEN}" for txt in cleaned_outputs]

    tokenizer = Tokenizer(num_words=num_words, filters="", oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(cleaned_inputs + decoder_texts)

    encoder_sequences = tokenizer.texts_to_sequences(cleaned_inputs)
    decoder_sequences = tokenizer.texts_to_sequences(decoder_texts)

    max_encoder_len = max(len(seq) for seq in encoder_sequences)
    max_decoder_len = max(len(seq) for seq in decoder_sequences)

    encoder_input_data = pad_sequences(
        encoder_sequences, maxlen=max_encoder_len, padding="post"
    )
    decoder_input_data = pad_sequences(
        decoder_sequences, maxlen=max_decoder_len, padding="post"
    )

    decoder_target_data = np.zeros_like(decoder_input_data)
    decoder_target_data[:, :-1] = decoder_input_data[:, 1:]

    num_tokens = min(num_words, len(tokenizer.word_index) + 1)
    return PreparedData(
        tokenizer=tokenizer,
        encoder_input_data=encoder_input_data,
        decoder_input_data=decoder_input_data,
        decoder_target_data=decoder_target_data,
        max_encoder_len=max_encoder_len,
        max_decoder_len=max_decoder_len,
        num_tokens=num_tokens,
    )
