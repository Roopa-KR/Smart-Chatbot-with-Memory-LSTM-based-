# Smart Chatbot with Memory

Backend for a simple memory-aware chatbot built with Flask, TensorFlow/Keras, SQLite, and NLTK.

## Files

- `app.py` - Flask API with `POST /chat`
- `model.py` - LSTM training and prediction logic
- `nlp_utils.py` - text preprocessing helpers
- `db.py` - SQLite memory and conversation logging
- `utils.py` - regex and helper functions
- `intents.json` - training data and responses
- `train_model.py` - standalone model training script

## Setup

1. Create and activate a Python environment that supports TensorFlow.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model artifacts if needed:

   ```bash
   python train_model.py
   ```

4. Start the API:

   ```bash
   python app.py
   ```

## Example Requests

### Store a name

```bash
curl -X POST http://127.0.0.1:5000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"123\",\"message\":\"My name is Nisha\"}"
```

Expected response:

```json
{ "response": "Nice to meet you, Nisha!" }
```

### Query a name

```bash
curl -X POST http://127.0.0.1:5000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"123\",\"message\":\"What is my name?\"}"
```

Expected response:

```json
{ "response": "Your name is Nisha." }
```

## Notes

- The API uses SQLite and creates the database automatically.
- Saved artifacts include `chatbot_lstm_model.keras`, `tokenizer.pkl`, and `model_metadata.json`.
- The health endpoint returns whether the ML stack is ready: `GET /health`.
