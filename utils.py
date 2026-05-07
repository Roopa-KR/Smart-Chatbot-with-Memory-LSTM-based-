import re
import json
import os
from datetime import datetime
from typing import Optional, Dict, List

# Force CPU-only execution
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import numpy as np
from sqlalchemy import create_engine, text
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import clean_text, START_TOKEN, END_TOKEN


# ---------------- DB ----------------
def get_engine(mysql_url):
    return create_engine(mysql_url)


# ---------------- METADATA ----------------
def load_metadata(path):
    with open(path, "r") as f:
        return json.load(f)


# ---------------- INTENT ----------------
def classify_intent(text):
    text = clean_text(text)

    if "track" in text or "status" in text:
        return "track"
    if "refund" in text or "return" in text or "cancel" in text:
        return "refund"
    if "payment" in text or "upi" in text or "card" in text:
        return "payment"
    if "product" in text or "price" in text:
        return "product"
    if "hi" in text or "hello" in text:
        return "greeting"

    return "other"


# ---------------- ORDER ID ----------------
def extract_order_id(text):
    match = re.search(r"\b\d{3,6}\b", text)
    return match.group(0) if match else None


# ---------------- USERS ----------------
def get_user_by_email(mysql_url, email):
    engine = get_engine(mysql_url)

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT user_id, name, email FROM users WHERE email=:email"),
            {"email": email},
        ).fetchone()

    if result:
        return {
            "user_id": result[0],
            "name": result[1],
            "email": result[2],
        }

    return None


# ---------------- ORDERS ----------------
def get_orders_by_user(mysql_url, user_id):
    engine = get_engine(mysql_url)

    with engine.connect() as conn:
        results = conn.execute(
            text("SELECT * FROM orders WHERE user_id=:uid"),
            {"uid": user_id},
        ).fetchall()

    orders = []
    for r in results:
        orders.append(
            {
                "order_id": r[0],
                "user_id": r[1],
                "product_name": r[2],
                "status": r[3],
            }
        )

    return orders


# ---------------- INTENT HANDLER ----------------
def handle_intent(intent, user_input, session, mysql_url):
    engine = get_engine(mysql_url)

    order_id = session.get("order_id")

    # ---- TRACK ----
    if intent == "track":
        if not order_id:
            return {"handled": True, "response": "Please provide your order ID.", "action": "ask_order_id"}

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT product_name, status FROM orders WHERE order_id=:oid"),
                {"oid": order_id},
            ).fetchone()

        if result:
            return {
                "handled": True,
                "response": f"Your order {order_id} ({result[0]}) is currently {result[1]}.",
                "action": "track_order",
            }

        return {"handled": True, "response": "Order not found.", "action": "error"}

    # ---- REFUND ----
    if intent == "refund":
        if not order_id:
            return {"handled": True, "response": "Please provide order ID for refund.", "action": "ask_order_id"}

        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO refunds (order_id, status, reason, created_at)
                    VALUES (:oid, 'pending', 'user_requested', :time)
                    """
                ),
                {"oid": order_id, "time": datetime.now()},
            )

        return {
            "handled": True,
            "response": f"Refund initiated for order {order_id}. It will be processed soon.",
            "action": "refund_created",
        }

    return {"handled": False}


# ---------------- FALLBACK ----------------
def keyword_intent_fallback(user_input, order_id=None):
    text = clean_text(user_input)

    if "hi" in text or "hello" in text:
        return "Hello! How can I assist you today?"

    if "how are you" in text:
        return "I'm here and ready to help you with your orders 😊"

    if "name" in text:
        return "You can call me your shopping assistant!"

    if "track" in text and order_id:
        return f"I can see you're asking about order {order_id}. Let me check that for you."

    if "refund" in text and order_id:
        return f"I can help you with refund for order {order_id}. Please confirm the reason."

    return "I'm here to help with orders, tracking, refunds, and payments. What do you need?"

# ---------------- CONTEXT ----------------
def build_context_input(context, current_input, max_len=3):
    context = context[-max_len:]
    return " ".join(context + [current_input])


# ---------------- CHAT LOG ----------------
def append_chat_log_mysql(
    mysql_url,
    user_input,
    bot_response,
    user_id=None,
    intent=None,
    action_taken=None,
):
    engine = get_engine(mysql_url)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO chat_logs (timestamp, user_id, user_input, bot_response, intent, action_taken)
                VALUES (:t, :uid, :ui, :br, :i, :a)
                """
            ),
            {
                "t": datetime.now(),
                "uid": user_id,
                "ui": user_input,
                "br": bot_response,
                "i": intent,
                "a": action_taken,
            },
        )


# ---------------- LSTM DECODE ----------------
def decode_sequence(encoder, decoder, tokenizer, input_text, max_encoder_len, max_decoder_len):
    seq = tokenizer.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen=max_encoder_len, padding="post")

    enc_out, state_h, state_c = encoder.predict(seq, verbose=0)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get("<start>", 1)

    decoded = []

    for _ in range(max_decoder_len):
        output_tokens, h, c = decoder.predict(
            [target_seq, enc_out, state_h, state_c],
            verbose=0
        )

        probs = output_tokens[0, -1, :]
        sampled_token_index = np.random.choice(len(probs), p=probs)

        word = None
        for w, i in tokenizer.word_index.items():
            if i == sampled_token_index:
                word = w
                break

        if word == "<end>" or word is None:
            break

        decoded.append(word)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        state_h, state_c = h, c

    sentence = " ".join(decoded).strip()

    # 🚨 FILTER BAD OUTPUT
    if len(sentence.split()) < 3:
        return ""

    return sentence
# ---------------- SAVE METADATA ----------------
def save_metadata(path, data):
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=4)