from datasets import load_dataset, DatasetDict
from transformers import BertConfig, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import src.config as cfg

from src.tokenize import tokenizer, tokenize_and_label
from src.multiheadBERT import MultiHeadBertForSequenceClassification 

def compute_metrics(p):
    if isinstance(p.predictions, dict):
        logits = p.predictions.get('logits_occasion')
    else:
        logits = p.predictions
    
    if not isinstance(logits, np.ndarray):
        logits = np.array(logits)
    # print("Logits :", logits)
    preds = np.argmax(logits, axis=-1)
    # print("Preds: ", preds)
    labels = p.label_ids
    # print("Labels: ", labels)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average = "macro")

    return {"Accuracy: ": acc, "F1_Macro:": f1,}

def train_multi(raw_jsonl: str, output_dir: str):
    #Load raw JSONL dataset
    #shuffling so that results stay same each time training occurs
    ds = load_dataset("json", data_files=raw_jsonl, split="train").shuffle(42)
    # print("Raw Example: ", ds[0])

    #Train/val split
    # tr= 0.8, tmp = 0.2
    #val , _ = tmp/2 = .1, .1
    tr, tmp = ds.train_test_split(0.80, seed=42).values()
    data = DatasetDict(train=tr, validation=tmp)
    # print(data)

    #Tokenize + add labels_* 
    data = data.map(tokenize_and_label, batched=False, load_from_cache_file = False,)
    # print("RAW Example: ", ds[0])
    # print("Tokenized Example: ", data["train"][0])
    
    
    config = BertConfig.from_pretrained("bert-base-uncased")
    model = MultiHeadBertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        config=config,
    )

    # print(model)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,#how many text inputs are process in parallel in training phase
        gradient_accumulation_steps = 4,
        per_device_eval_batch_size=2,
        learning_rate=3e-5, # step size to adjust model weights
        num_train_epochs=8, # num of full passes training will make over entire dataset
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy ="steps",
        logging_steps=500,
    )

   
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=tokenizer,
        compute_metrics = compute_metrics,
    )

    trainer.train()
    print("Validation:", trainer.evaluate())
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_multi("data/seed_slots.jsonl", "models/multihead_bert")
