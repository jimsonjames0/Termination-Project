from datasets import load_dataset, DatasetDict
from transformers import BertConfig, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import src.config as cfg

from src.tokenize import tokenizer, tokenize_and_label
from src.multiheadBERT import MultiHeadBertForSequenceClassification 

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_metrics(p):
    # Unpack predictions
    (
        logits_occasion,
        logits_size,
        logits_due_date,
        logits_flavor,
        logits_filling,
        logits_icing,
    ) = p.predictions

    # Unpack labels 
    (labels_occasion, labels_size, labels_due_date,labels_flavor, labels_filling,labels_icing,) = p.label_ids

    preds_occasion = np.argmax(logits_occasion, axis=-1)
    preds_size     = np.argmax(logits_size, axis=-1)
    preds_due_date = np.argmax(logits_due_date, axis=-1)

    acc_occ = accuracy_score(labels_occasion, preds_occasion)
    f1_occ  = f1_score(labels_occasion, preds_occasion, average="macro")

    acc_size = accuracy_score(labels_size, preds_size)
    f1_size  = f1_score(labels_size, preds_size, average="macro")

    acc_date = accuracy_score(labels_due_date, preds_due_date)
    f1_date  = f1_score(labels_due_date, preds_due_date, average="macro")

   
    thr_flavor = 0.6
    thr_filling = 0.655
    thr_icing = 0.66

    probs_flavor  = sigmoid(logits_flavor)
    probs_filling = sigmoid(logits_filling)
    probs_icing   = sigmoid(logits_icing)

    preds_flavor  = (probs_flavor  > thr_flavor).astype(int)
    preds_filling = (probs_filling > thr_filling).astype(int)
    preds_icing   = (probs_icing   > thr_icing).astype(int)

    print("Flavor label mean:", labels_flavor.mean(), "pred mean:", preds_flavor.mean())
    print("Filling label mean:", labels_filling.mean(), "pred mean:", preds_filling.mean())
    print("Icing label mean:", labels_icing.mean(), "pred mean:", preds_icing.mean())

    # labels_flavor/filling/icing are already 0/1 multi-hot matrices
    f1_flavor  = f1_score(labels_flavor,  preds_flavor,  average="macro", zero_division=0)
    f1_filling = f1_score(labels_filling, preds_filling, average="macro", zero_division=0)
    f1_icing   = f1_score(labels_icing,   preds_icing,   average="macro", zero_division=0)

    # overall averages (optional)
    acc_mean = (acc_occ + acc_size + acc_date) / 3.0
    f1_mean_single = (f1_occ + f1_size + f1_date) / 3.0
    f1_mean_multi  = (f1_flavor + f1_filling + f1_icing) / 3.0

    return {
        "Acc_Occasion": acc_occ,
        "F1_Occasion":  f1_occ,

        "Acc_Size":     acc_size,
        "F1_Size":      f1_size,

        "Acc_DueDate":  acc_date,
        "F1_DueDate":   f1_date,

        "F1_Flavor":    f1_flavor,
        "F1_Filling":   f1_filling,
        "F1_Icing":     f1_icing,

        "Acc_Mean_Single": acc_mean,
        "F1_Mean_Single":  f1_mean_single,
        "F1_Mean_Multi":   f1_mean_multi,
    }

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
    
    # "models/multihead_bert_s2" for evlauation
    config = BertConfig.from_pretrained("bert-base-uncased")
    model = MultiHeadBertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        #"models/multihead_bert_s2",
        config=config,
    )

    # print(model)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,#how many text inputs are process in parallel in training phase
        #gradient_accumulation_steps = 4,
        per_device_eval_batch_size=16,
        learning_rate=3e-5, # step size to adjust model weights
        num_train_epochs=15, # num of full passes training will make over entire dataset
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy ="steps",
        logging_steps=500,
        label_names = ["labels_occasion", "labels_size", "labels_due_date", "labels_flavor", "labels_filling", "labels_icing"],
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
    train_multi("data/seed_slots.jsonl", "models/multihead_bert_s2")

