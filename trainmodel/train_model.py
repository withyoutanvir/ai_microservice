import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import clean_text
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch

# Example placeholder dataset loading
def load_data():
    # Replace with your real dataset loading logic
    return [
        {"text": "Patient has high blood sugar and frequent urination", "labels": ["diabetes"]},
        {"text": "Patient suffers from shortness of breath and wheezing", "labels": ["asthma"]},
        # Add more samples here
    ]

def preprocess_data(tokenizer, data, mlb):
    texts = [d['text'] for d in data]
    labels = mlb.transform([d['labels'] for d in data])
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    encodings["labels"] = labels.tolist()
    return encodings

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

def main():
    model_name = "bert-base-uncased"

    data = load_data()
    all_labels = sorted({label for d in data for label in d['labels']})
    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb.fit([all_labels])  # Fit to known labels

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(all_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"  # Enables sigmoid + BCE loss internally
    )

    encodings = preprocess_data(tokenizer, data, mlb)
    train_dataset = Dataset(encodings)

    training_args = TrainingArguments(
        output_dir='./saved_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model('./saved_model')

if __name__ == "__main__":
    main()
