from data_loader import load_stock_data
from preprocess import preprocess_data
from model import build_model

from transformers import BertTokenizer
import tensorflow as tf

# Load data
data = load_stock_data("AAPL")
X, y, scaler = preprocess_data(data)

# Dummy text (replace with real news later)
texts = ["stock going up"] * len(X)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
enc = tokenizer(
    texts,
    padding='max_length',
    truncation=True,
    max_length=32,
    return_tensors="tf"
)

input_ids = enc['input_ids']
attention_mask = enc['attention_mask']

# Build model
model = build_model()

# Train
model.fit([X, input_ids, attention_mask], y, epochs=3, batch_size=32)

# Save model
model.save("model.h5")