# Smart Chatbot with Memory

Terminal-based memory-aware chatbot built with TensorFlow/Keras, SQLite, and NLTK.

## Files

- `app.py` - terminal chatbot entrypoint
- `model.py` - LSTM training and prediction logic
- `nlp_utils.py` - text preprocessing helpers
- `db.py` - SQLite memory and conversation logging
- `utils.py` - regex and helper functions
- `intents.json` - training data and responses
- `train_model.py` - standalone model training script

## Setup

1. Create and activate a Python environment.
2. Install the runtime dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. If you want to train or use the LSTM intent model, install TensorFlow in the Python 3.10 `.venv` environment:

   ```bash
   pip install -r requirements-ml.txt
   ```

4. Train the model artifacts if needed:

   ```bash
   python train_model.py
   ```

5. Start the chatbot in the terminal:

   ```bash
   python app.py
   ```

## Example Usage

### Interactive mode

```bash
python app.py --user-id 123
```

Then type:

```json
My name is Nisha
```

Expected reply:

```text
Bot: Nice to meet you, Nisha!
```

Then ask:

```text
What is my name?
```

Expected reply:

```text
Bot: Your name is Nisha.
```

### One-shot mode

```bash
python app.py --user-id 123 --message "Hi"
```

## Notes

- The API uses SQLite and creates the database automatically.
- Saved artifacts include `chatbot_lstm_model.keras`, `tokenizer.pkl`, and `model_metadata.json`.
- Use Python 3.10 in `.venv` for the ML-backed model path.
