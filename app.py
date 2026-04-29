import os
import pickle
from dotenv import load_dotenv
import streamlit as st
from tensorflow.keras.models import load_model

from utils import (
    classify_intent,
    extract_order_id,
    get_orders_by_user,
    get_user_by_email,
    handle_intent,
    keyword_intent_fallback,
    append_chat_log_mysql,
    build_context_input,
    load_metadata,
)

load_dotenv()

st.set_page_config(page_title="E-commerce Chatbot", page_icon="🛍️")
st.title("🛍️ E-commerce Support Chatbot")

MODEL_DIR = "models"
ENCODER_MODEL_PATH = os.path.join(MODEL_DIR, "encoder_model.keras")
DECODER_MODEL_PATH = os.path.join(MODEL_DIR, "decoder_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

MYSQL_URL = os.getenv("MYSQL_URL")


# ---------- FIX FOR STREAMLIT RERUN ----------
def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    encoder = load_model(ENCODER_MODEL_PATH)
    decoder = load_model(DECODER_MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    metadata = load_metadata(METADATA_PATH)

    return encoder, decoder, tokenizer, metadata


# ---------- SESSION ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "context" not in st.session_state:
    st.session_state.context = []

if "user" not in st.session_state:
    st.session_state.user = None

if "order_id" not in st.session_state:
    st.session_state.order_id = None


# ---------- LOGIN ----------
if not st.session_state.user:
    st.subheader("Login")

    email = st.text_input("Enter Email")

    if st.button("Login"):
        user = get_user_by_email(MYSQL_URL, email)

        if user:
            st.session_state.user = user
            st.success(f"Welcome {user['name']}")
            rerun()
        else:
            st.error("User not found")

    st.stop()


# ---------- DISPLAY CHAT ----------
for msg in st.session_state.messages:
    st.write(f"**{msg['role']}**: {msg['content']}")


# ---------- INPUT ----------
user_input = st.text_input("Ask something...")

if st.button("Send") and user_input:

    st.session_state.messages.append({"role": "You", "content": user_input})

    # extract order id
    order_id = extract_order_id(user_input)
    if order_id:
        st.session_state.order_id = order_id

    # intent
    intent = classify_intent(user_input)

    # backend logic
    result = handle_intent(
        intent,
        user_input,
        st.session_state,
        MYSQL_URL
    )

    # ---------- LSTM ----------
    encoder, decoder, tokenizer, metadata = load_models()

    context_input = build_context_input(
        st.session_state.context,
        user_input
    )

    from utils import decode_sequence

    with st.spinner("Thinking..."):
        model_output = decode_sequence(
            encoder,
            decoder,
            tokenizer,
            context_input,
            metadata["max_encoder_len"],
            metadata["max_decoder_len"]
        )

    bot_response = ""

    if result["handled"]:
        bot_response = result["response"]
        action = result["action"]
    else:
        bot_response = model_output
        action = "lstm"

    if not bot_response:
        bot_response = keyword_intent_fallback(
            user_input,
            st.session_state.order_id
        )

    # save context
    st.session_state.context.append(user_input)
    st.session_state.context = st.session_state.context[-3:]

    st.session_state.messages.append({"role": "Bot", "content": bot_response})

    # save logs
    append_chat_log_mysql(
        MYSQL_URL,
        user_input,
        bot_response,
        user_id=st.session_state.user["user_id"],
        intent=intent,
        action_taken=action,
    )

    rerun()


# ---------- SIDEBAR ----------
with st.sidebar:
    st.write("User:", st.session_state.user["name"])

    orders = get_orders_by_user(
        MYSQL_URL,
        st.session_state.user["user_id"]
    )

    st.subheader("Orders")

    for o in orders:
        st.write(f"{o['order_id']} - {o['product_name']} ({o['status']})")