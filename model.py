from typing import Tuple
import os

# Force CPU-only execution
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.layers import AdditiveAttention, Concatenate, LSTM, Dense, Embedding, Input
from tensorflow.keras.models import Model


def build_seq2seq_model(
    num_tokens: int, embedding_dim: int, latent_dim: int
) -> Tuple[Model, Model, Model]:
    """Build training model + inference encoder and decoder models."""
    encoder_inputs = Input(shape=(None,), name="encoder_inputs")
    encoder_embedding = Embedding(
        input_dim=num_tokens, output_dim=embedding_dim, mask_zero=True, name="encoder_emb"
    )(encoder_inputs)
    encoder_lstm = LSTM(
        latent_dim, return_sequences=True, return_state=True, name="encoder_lstm"
    )
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    decoder_embedding_layer = Embedding(
        input_dim=num_tokens, output_dim=embedding_dim, mask_zero=True, name="decoder_emb"
    )
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    decoder_lstm = LSTM(
        latent_dim, return_sequences=True, return_state=True, name="decoder_lstm"
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    attention_layer = AdditiveAttention(name="attention_layer")
    context_vector = attention_layer([decoder_outputs, encoder_outputs])
    decoder_context = Concatenate(axis=-1, name="decoder_context_concat")(
        [decoder_outputs, context_vector]
    )
    decoder_dense = Dense(num_tokens, activation="softmax", name="decoder_output")
    decoder_outputs = decoder_dense(decoder_context)

    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

    encoder_outputs_input = Input(shape=(None, latent_dim), name="encoder_outputs_input")
    decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_h")
    decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_emb_inf = decoder_embedding_layer(decoder_inputs)
    decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
        decoder_emb_inf, initial_state=decoder_states_inputs
    )
    context_vector_inf = attention_layer([decoder_outputs_inf, encoder_outputs_input])
    decoder_context_inf = Concatenate(axis=-1)([decoder_outputs_inf, context_vector_inf])
    decoder_states_inf = [state_h_inf, state_c_inf]
    decoder_outputs_inf = decoder_dense(decoder_context_inf)
    decoder_model = Model(
        [decoder_inputs, encoder_outputs_input] + decoder_states_inputs,
        [decoder_outputs_inf] + decoder_states_inf,
    )
    return training_model, encoder_model, decoder_model
