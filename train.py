
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import BertTokenizer, BertConfig, TrainingArguments, Trainer
from model import BertRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv("priority_data.csv")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class PriorityDataset(TorchDataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        encoding = tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["priority"], dtype=torch.float)
        }

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

train_dataset = PriorityDataset(train_df.reset_index(drop=True))
test_dataset = PriorityDataset(test_df.reset_index(drop=True))

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertRegressor.from_pretrained("bert-base-uncased", config=config)

training_args = TrainingArguments(
    output_dir="./priority_model",
    evaluation_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    save_total_limit=1,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"mse": mean_squared_error(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained("./priority_model")
tokenizer.save_pretrained("./priority_model")
