# ✅ 1. Imports
!pip install evaluate -q
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch
import evaluate

# ✅ 2. Load CSV
df = pd.read_csv("/kaggle/input/datasetss/emotion_dataset_expanded.csv")
df = df[["text", "emotion"]].dropna()

# ✅ 3. Label encoding
labels = df["emotion"].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["emotion"].map(label2id)

# ✅ 4. Train-test split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# ✅ 5. Tokenizer (DistilBERT)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ✅ 6. Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# ✅ 7. Define metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# ✅ 8. Training arguments
training_args = TrainingArguments(
    output_dir="/kaggle/working/model_output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_dir="/kaggle/working/logs",
    logging_steps=10,
    report_to="none"  # Disable WandB
)

# ✅ 9. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# ✅ 10. Start training
print("🚀 Training Started...")
trainer.train()
print("✅ Training Completed.")
