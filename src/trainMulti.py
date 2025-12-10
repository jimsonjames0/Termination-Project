from datasets import load_dataset, DatasetDict
from transformers import BertConfig, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import src.config as cfg

from src.tokenize import tokenizer, tokenize_and_label
from src.multiheadBERT import MultiHeadBertForSequenceClassification 

# def compute_metrics(p):
#     # Unpack predictions (from model forward tuple)
#     logits_occasion, logits_size, logits_due_date = p.predictions

#     # Unpack labels (in the same order as label_names)
#     labels_occasion, labels_size, labels_due_date = p.label_ids

#     # Argmax over class dimension for each head
#     preds_occasion = np.argmax(logits_occasion, axis=-1)
#     preds_size     = np.argmax(logits_size, axis=-1)
#     preds_due_date = np.argmax(logits_due_date, axis=-1)

#     # Occasion metrics
#     acc_occ = accuracy_score(labels_occasion, preds_occasion)
#     f1_occ  = f1_score(labels_occasion, preds_occasion, average="macro")

#     # Size metrics
#     acc_size = accuracy_score(labels_size, preds_size)
#     f1_size  = f1_score(labels_size, preds_size, average="macro")

#     # Due date metrics
#     acc_date = accuracy_score(labels_due_date, preds_due_date)
#     f1_date  = f1_score(labels_due_date, preds_due_date, average="macro")

#    #overall accuracy and 
#     acc_mean = (acc_occ + acc_size + acc_date) / 3.0
#     f1_mean  = (f1_occ + f1_size + f1_date)  / 3.0

#     return {
#         "Acc_Occasion": acc_occ,
#         "F1_Macro_Occasion": f1_occ,
#         "Acc_Size": acc_size,
#         "F1_Macro_Size": f1_size,
#         "Acc_DueDate": acc_date,
#         "F1_Macro_DueDate": f1_date,
#         "Acc_Mean": acc_mean,
#         "F1_Mean": f1_mean,
#     }

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
        # label_names = ["labels_occasion", "labels_size", "labels_due_date"],
    )

   
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=tokenizer,
        # compute_metrics = compute_metrics,
    )

    trainer.train()
    print("Validation:", trainer.evaluate())
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_multi("data/small.jsonl", "models/multihead_bert")
