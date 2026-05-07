import os
import pickle

# Force CPU-only execution
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from model import build_seq2seq_model
from preprocess import build_training_matrices, load_dataset
from utils import save_metadata


DATASET_PATH = "data/dataset.json"
MODEL_DIR = "models"

TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
TRAINING_MODEL_PATH = os.path.join(MODEL_DIR, "seq2seq_training.keras")
ENCODER_MODEL_PATH = os.path.join(MODEL_DIR, "encoder_model.keras")
DECODER_MODEL_PATH = os.path.join(MODEL_DIR, "decoder_model.keras")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)

    print("Preprocessing...")
    prepared = build_training_matrices(dataset)

    # ---------------- MODEL CONFIG ---------------- #
    latent_dim = 256          # ↑ stronger memory
    embedding_dim = 128       # ↑ better word understanding
    batch_size = 32
    epochs = 100

    print("Building model...")
    training_model, encoder_model, decoder_model = build_seq2seq_model(
        num_tokens=prepared.num_tokens,
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
    )

    decoder_target_one_hot = to_categorical(
        prepared.decoder_target_data,
        num_classes=prepared.num_tokens
    )

    training_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ---------------- CALLBACK ---------------- #
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    print("Training started...\n")

    history = training_model.fit(
        [prepared.encoder_input_data, prepared.decoder_input_data],
        decoder_target_one_hot,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    print("\nSaving models...")

    training_model.save(TRAINING_MODEL_PATH)
    encoder_model.save(ENCODER_MODEL_PATH)
    decoder_model.save(DECODER_MODEL_PATH)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(prepared.tokenizer, f)

    save_metadata(
        METADATA_PATH,
        {
            "model_type": "lstm_seq2seq",
            "dataset_size": len(dataset),
            "max_encoder_len": prepared.max_encoder_len,
            "max_decoder_len": prepared.max_decoder_len,
            "num_tokens": prepared.num_tokens,
            "embedding_dim": embedding_dim,
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "epochs": len(history.history["loss"]),
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
        },
    )

    print("\n✅ Training complete")
    print(f"Final Train Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Val Loss: {history.history['val_loss'][-1]:.4f}")
    print("Models saved in /models folder")


if __name__ == "__main__":
    main()