WalterBot: Multi-Head BERT Slot Extraction 🍰

This repository contains the slot extraction module for WalterBot, a cake-ordering assistant that turns free-form customer messages into a structured JSON order.

The model reads messages like:

“I need a 10-inch marble and strawberry birthday cake for my son next week.”

and predicts:

Occasion → birthday

Size → 10-inch

Due_Date → too_soon

Flavor → ["marble", "strawberry"]

Filling → e.g. ["oreo filling"]

Icing → e.g. ["vanilla buttercream"]

All of this is done with one BERT encoder and six classification heads.

1. Project Overview
Goal

Help a small bakery (Walter’s Cake Art) automate custom cake orders by:

Accepting natural-language chat input from customers.

Extracting structured “slots” required to fulfill an order:

Occasion

Size

Due date (explicit / too soon / unspecified)

Flavor(s)

Filling(s)

Icing(s)

Returning a JSON object that downstream systems / staff can use directly.

Key Design Choice

Instead of:

Traditional BIO NER tagging, or

Separate models per slot,

we use a single Multi-Head BERT sequence classifier:

Shared encoder: bert-base-uncased

Six heads on top of the pooled [CLS] embedding:

3 × single-label (softmax + CrossEntropyLoss)

3 × multi-label (sigmoid + BCEWithLogitsLoss with class weights)

This makes training and inference simpler and faster while still modeling all slots jointly.

2. Architecture
2.1 Model

Defined in src/multiheadBERT.py:

Base: BertModel(config)

Shared representation: pooled_output = outputs.pooler_output
Shape: [batch_size, hidden_size] (e.g. [B, 768])

Heads (each a nn.Linear(hidden_size, num_labels)):

occasion → shape [B, num_occasions]

size → [B, num_sizes]

due_date → [B, num_due_dates]

flavor → [B, num_flavors]

filling → [B, num_fillings]

icing → [B, num_icings]

2.2 Losses

In forward(...):

Single-label:

loss_o = CrossEntropyLoss(logits_occasion, labels_occasion)

Same pattern for size & due_date.

Multi-label:

labels_* are multi-hot vectors (0/1 per class).

BCEWithLogitsLoss with pos_weight to handle class imbalance:

flavor_weights ~ 8–9

filling_weights ~ 15+

icing_weights ~ 10+

Total loss:

loss = loss_o + loss_s + loss_d + loss_fl + loss_fi + loss_i


(Only terms with provided labels are included.)

2.3 Outputs

The model returns a tuple:

(loss, logits_occasion, logits_size, logits_due_date,
 logits_flavor, logits_filling, logits_icing)

3. Data & Labeling
3.1 Input Format

All training data is JSONL of this form (e.g. data/seed_slots.jsonl, data/augmented.jsonl):

{
  "Text": "Hi, this is Sophia. I’d like a 7-inch marble cake ...",
  "Occasion": "anniversary",
  "Size": "7-inch",
  "Due_Date": "explicit",
  "Flavor": ["marble"],
  "Filling": ["vanilla custard", "cookies & cream buttercream"],
  "Icing": ["strawberry buttercream"]
}

3.2 Tokenization + Label Encoding

Handled in src/tokenize.py:

Uses BertTokenizerFast to produce:

input_ids, attention_mask, token_type_ids

Converts string labels → integer IDs using mappings in src/config.py, e.g.:

occasion_label2id, size_label2id, date_label2id

Converts multi-label lists into multi-hot vectors of shape:

labels_flavor → [num_flavors]

labels_filling → [num_fillings]

labels_icing → [num_icings]

4. Training

Training is orchestrated by src/trainMulti.py using Hugging Face Trainer.

4.1 Dependencies

Install typical dependencies:

pip install torch transformers datasets scikit-learn


(Optional: wandb for logging.)

4.2 Running Training (Local)

From the project root:

python3 -m src.trainMulti


Key pieces inside trainMulti.py:

Load & shuffle dataset:

ds = load_dataset("json", data_files=raw_jsonl, split="train").shuffle(42)
tr, tmp = ds.train_test_split(0.80, seed=42).values()
data = DatasetDict(train=tr, validation=tmp)
data = data.map(tokenize_and_label, batched=False, load_from_cache_file=False)


Model & config:

config = BertConfig.from_pretrained("bert-base-uncased")
model = MultiHeadBertForSequenceClassification.from_pretrained(
    "bert-base-uncased", config=config
)


TrainingArguments (typical values you used):

args = TrainingArguments(
    output_dir="models/multihead_bert",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    num_train_epochs=15,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=500,
)


Trainer with custom metrics:

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

4.3 Metrics

compute_metrics(p):

Unpacks logits for each head.

Decodes predictions:

Single-label: argmax over class dimension.

Multi-label: sigmoid → threshold → 0/1.

Computes:

Accuracy + macro-F1 for Occasion, Size, Due_Date.

Macro-F1 for Flavor, Filling, Icing.

Also prints mean label vs mean prediction for each multi-label head to debug over/under-prediction.

5. Inference

Inference is handled in src/predict.py (or src/predict_multi.py depending on version).

5.1 Loading the Trained Model
from transformers import BertTokenizerFast, BertConfig
from src.multiheadBERT import MultiHeadBertForSequenceClassification
import src.config as cfg

MODEL_DIR = "models/multihead_bert"  # or your Google Drive path on Colab

tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
config = BertConfig.from_pretrained(MODEL_DIR)
model = MultiHeadBertForSequenceClassification.from_pretrained(MODEL_DIR, config=config)
model.eval()

5.2 Prediction Function
def predict_all_slots(text: str):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        loss, logits_occasion, logits_size, logits_due_date, \
            logits_flavor, logits_filling, logits_icing = model(**inputs)

    # Single-label predictions
    occ_id  = logits_occasion.argmax(-1).item()
    size_id = logits_size.argmax(-1).item()
    date_id = logits_due_date.argmax(-1).item()

    # Multi-label: sigmoid + thresholds
    thr_flavor  = 0.60
    thr_filling = 0.655
    thr_icing   = 0.66

    probs_flavor  = logits_flavor.sigmoid()[0]
    probs_filling = logits_filling.sigmoid()[0]
    probs_icing   = logits_icing.sigmoid()[0]

    chosen_fl = (probs_flavor  > thr_flavor).nonzero(as_tuple=True)[0].tolist()
    chosen_fi = (probs_filling > thr_filling).nonzero(as_tuple=True)[0].tolist()
    chosen_i  = (probs_icing   > thr_icing).nonzero(as_tuple=True)[0].tolist()

    pred_flavor  = [cfg.FLAVOR_LABELS[i]  for i in chosen_fl]
    pred_filling = [cfg.FILLING_LABELS[i] for i in chosen_fi]
    pred_icing   = [cfg.ICING_LABELS[i]   for i in chosen_i]

    return {
        "Text: ": text,
        "Occasion: ": cfg.occasion_id2label[occ_id],
        "Size: ":     cfg.size_id2label[size_id],
        "Due_Date: ": cfg.date_id2label[date_id],
        "Flavor: ":   pred_flavor,
        "Filling: ":  pred_filling,
        "Icing: ":    pred_icing,
    }


Run with:

python3 -m src.predict


Example outputs:

I need a 10-inch marble and strawberry birthday cake for my son next week
→ {
  "Occasion": "birthday",
  "Size": "10-inch",
  "Due_Date": "too_soon",
  "Flavor": ["marble", "strawberry"],
  "Filling": ["oreo filling"],
  "Icing": []
}

6. Results (Current Best)

On a held-out validation set (with ~11k total examples and balanced flavor/filling/icing usage), the best model achieves roughly:

Occasion

Accuracy ≈ 99.9%

F1 macro ≈ 0.999

Size

Accuracy ≈ 99.8%

F1 macro ≈ 0.998

Due_Date

Accuracy ≈ 99.9%

F1 macro ≈ 0.999

Flavor (multi-label)

F1 macro ≈ 0.97–0.98

Icing (multi-label)

F1 macro ≈ 0.74–0.75

Filling (multi-label)

F1 macro ≈ 0.53–0.55

Flavor is essentially solved; Filling and Icing are still the hardest, due to semantic overlap between flavor/filling/icing terms (e.g., “white chocolate ganache” can be filling or icing depending on phrasing).

7. Future Work

Planned improvements:

Better Filling/Icing separation

Refine annotation guidelines for ambiguous phrases (e.g., “chocolate cream”).

Possibly collapse some categories or add a combined “filling_or_icing” slot.

Threshold tuning per label

Instead of global thresholds, learn / search per-label thresholds to optimize F1.

Confidence & fallback logic

When low confidence, ask the user clarifying questions in the chat flow.

8. Repository Structure

Rough structure (yours may vary slightly):

.
├── data/
│   ├── seed_slots.jsonl
│   ├── augmented.jsonl
│   └── ...
├── models/
│   └── multihead_bert/
│       ├── config.json
│       ├── tokenizer.json / tokenizer_config.json / vocab.txt
│       └── model.safetensors (or pytorch_model.bin)
├── src/
│   ├── config.py              # label lists, id ↔ label mappings
│   ├── tokenize.py            # tokenizer + tokenize_and_label
│   ├── multiheadBERT.py       # model definition
│   ├── trainMulti.py          # training script
│   ├── predict.py             # inference script
│   └── ...
├── augmentData.py             # data augmentation / balancing
└── README.md                  # (this file)


If you tell me your exact current file names/paths, I can tweak this README to match them perfectly (and add a short “How to run on Colab” section too).

You said:
can you give me something I can just drop straight into the githubreadme when I press edit that looks nice
ChatGPT said:
Thought for 7s

Sure — here’s a clean, copy-paste-ready README.md you can drop straight into GitHub.

WalterBot: Multi-Head BERT Slot Extraction 🍰

WalterBot is a cake-ordering assistant that turns messy, natural-language messages into a structured JSON order the bakery can actually bake from.

Example:

“I need a 10-inch marble and strawberry birthday cake for my son next week.”

Becomes something like:

{
  "Occasion": "birthday",
  "Size": "10-inch",
  "Due_Date": "too_soon",
  "Flavor": ["marble", "strawberry"],
  "Filling": ["oreo filling"],
  "Icing": ["vanilla buttercream"]
}


This repo contains the BERT-based slot extraction model that powers that behavior.

🌟 Highlights

Single BERT encoder (bert-base-uncased)

Six heads on top of the pooled [CLS] embedding:

3 × single-label:

Occasion

Size

Due_Date

3 × multi-label:

Flavor

Filling

Icing

Handles multiple flavors/fillings/icings per order (multi-label classification)

Uses class-weighted BCE for rare fillings & icings

Trained on augmented ~11K+ examples of realistic cake orders

🧱 Architecture

Everything lives in src/multiheadBERT.py.

Encoder

BertModel(config) from Hugging Face.

We take pooled_output = outputs.pooler_output
Shape: [batch_size, hidden_size] (e.g. [B, 768]).

Heads

Each head is a linear layer over the pooled output:

self.heads = nn.ModuleDict({
    "occasion": nn.Linear(hidden_size, num_occasions),
    "size":     nn.Linear(hidden_size, num_sizes),
    "due_date": nn.Linear(hidden_size, num_due_dates),
    "flavor":   nn.Linear(hidden_size, num_flavors),
    "filling":  nn.Linear(hidden_size, num_fillings),
    "icing":    nn.Linear(hidden_size, num_icings),
})


Output shapes:

Occasion / Size / Due_Date → [B, num_classes] (softmax)

Flavor / Filling / Icing → [B, num_labels] (sigmoid, multi-label)

Losses

Single-label slots:

CrossEntropyLoss on:

logits_occasion vs labels_occasion

logits_size vs labels_size

logits_due_date vs labels_due_date

Multi-label slots:

BCEWithLogitsLoss with pos_weight to handle imbalance:

Flavor → medium weight

Filling → higher weight

Icing → medium-high weight

Total loss:

loss = loss_occasion + loss_size + loss_due_date \
     + loss_flavor + loss_filling + loss_icing


(Only terms with labels provided are included.)

The model’s forward returns:

(
  loss,
  logits_occasion,
  logits_size,
  logits_due_date,
  logits_flavor,
  logits_filling,
  logits_icing,
)

📦 Data Format

Training data is in JSONL format (data/*.jsonl), one order per line:

{
  "Text": "Hi, this is Sophia. I'd like a 7-inch marble cake...",
  "Occasion": "anniversary",
  "Size": "7-inch",
  "Due_Date": "explicit",
  "Flavor": ["marble"],
  "Filling": ["vanilla custard", "cookies & cream buttercream"],
  "Icing": ["strawberry buttercream"]
}

Tokenization & Labels

Handled in src/tokenize.py and src/config.py:

Uses BertTokenizerFast to create:

input_ids, attention_mask, token_type_ids

Maps string labels → integer IDs:

occasion_label2id, size_label2id, date_label2id

Multi-label slots become multi-hot vectors:

labels_flavor → [num_flavors]

labels_filling → [num_fillings]

labels_icing → [num_icings]

🚀 Training

Training is done via Hugging Face’s Trainer in src/trainMulti.py.

Install Dependencies
pip install torch transformers datasets scikit-learn


(Optional) For logging with Weights & Biases:

pip install wandb

Run Training (Local)

From the project root:

python3 -m src.trainMulti


What trainMulti.py does (simplified):

from datasets import load_dataset, DatasetDict
from transformers import BertConfig, TrainingArguments, Trainer
from src.tokenize import tokenizer, tokenize_and_label
from src.multiheadBERT import MultiHeadBertForSequenceClassification

# 1) Load & split
ds = load_dataset("json", data_files="data/seed_slots.jsonl", split="train").shuffle(42)
train_split, val_split = ds.train_test_split(0.80, seed=42).values()
data = DatasetDict(train=train_split, validation=val_split)

# 2) Tokenize & add labels
data = data.map(tokenize_and_label, batched=False, load_from_cache_file=False)

# 3) Model
config = BertConfig.from_pretrained("bert-base-uncased")
model = MultiHeadBertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)

# 4) Training arguments
args = TrainingArguments(
    output_dir="models/multihead_bert",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    num_train_epochs=15,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=500,
)

# 5) Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # custom metrics
)

trainer.train()
trainer.save_model("models/multihead_bert")
tokenizer.save_pretrained("models/multihead_bert")

📊 Metrics

src/trainMulti.py defines compute_metrics(p) to evaluate each head:

For Occasion / Size / Due_Date (single-label):

Accuracy

Macro-F1

For Flavor / Filling / Icing (multi-label):

Macro-F1 (using sigmoid + thresholds per slot)

We also print label mean vs prediction mean per head (e.g., how often a label is actually 1 vs predicted 1) to debug thresholds and class weights.

Example (best-ish run)

On a large augmented dataset (~11K orders):

Occasion

Accuracy ≈ 99.8–99.9%

F1 (macro) ≈ 0.998+

Size

Accuracy ≈ 99.7–99.8%

F1 (macro) ≈ 0.997+

Due_Date

Accuracy ≈ 99.9%

F1 (macro) ≈ 0.999+

Flavor (multi-label)

F1 (macro) ≈ 0.97–0.98

Icing (multi-label)

F1 (macro) ≈ ~0.74–0.75

Filling (multi-label)

F1 (macro) ≈ ~0.53–0.55

Flavor is essentially “solved”; Filling and Icing are harder because real customer language blurs the line between flavor / filling / icing.

🔮 Inference

Inference happens in src/predict.py (or src/predict_multi.py).

Load Model
import torch
from transformers import BertTokenizerFast, BertConfig
from src.multiheadBERT import MultiHeadBertForSequenceClassification
import src.config as cfg

MODEL_DIR = "models/multihead_bert"

tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
config = BertConfig.from_pretrained(MODEL_DIR)
model = MultiHeadBertForSequenceClassification.from_pretrained(MODEL_DIR, config=config)
model.eval()

Predict Function
def predict_all_slots(text: str):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        loss, logits_occasion, logits_size, logits_due_date, \
            logits_flavor, logits_filling, logits_icing = model(**inputs)

    # Single-label
    occ_id  = logits_occasion.argmax(-1).item()
    size_id = logits_size.argmax(-1).item()
    date_id = logits_due_date.argmax(-1).item()

    # Multi-label thresholds
    thr_flavor  = 0.60
    thr_filling = 0.655
    thr_icing   = 0.66

    probs_flavor  = logits_flavor.sigmoid()[0]
    probs_filling = logits_filling.sigmoid()[0]
    probs_icing   = logits_icing.sigmoid()[0]

    chosen_fl = (probs_flavor  > thr_flavor).nonzero(as_tuple=True)[0].tolist()
    chosen_fi = (probs_filling > thr_filling).nonzero(as_tuple=True)[0].tolist()
    chosen_i  = (probs_icing   > thr_icing).nonzero(as_tuple=True)[0].tolist()

    pred_flavor  = [cfg.FLAVOR_LABELS[i]  for i in chosen_fl]
    pred_filling = [cfg.FILLING_LABELS[i] for i in chosen_fi]
    pred_icing   = [cfg.ICING_LABELS[i]   for i in chosen_i]

    return {
        "Text: ": text,
        "Occasion: ": cfg.occasion_id2label[occ_id],
        "Size: ":     cfg.size_id2label[size_id],
        "Due_Date: ": cfg.date_id2label[date_id],
        "Flavor: ":   pred_flavor,
        "Filling: ":  pred_filling,
        "Icing: ":    pred_icing,
    }


Run:

python3 -m src.predict

📁 Repo Structure

Rough layout:

.
├── data/
│   ├── seed_slots.jsonl          # original labeled orders
│   ├── augmented.jsonl           # synthetic / balanced orders
│   └── ...
├── models/
│   └── multihead_bert/
│       ├── config.json
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       └── model.safetensors     # or pytorch_model.bin
├── src/
│   ├── config.py                 # label lists & mappings
│   ├── tokenize.py               # tokenizer + tokenize_and_label
│   ├── multiheadBERT.py          # multi-head BERT model
│   ├── trainMulti.py             # training driver
│   ├── predict.py                # inference script
│   └── ...
├── augmentData.py                # data augmentation / balancing
└── README.md

🔧 Future Work

Improve Filling vs Icing distinction (they’re very semantically similar).

Per-label threshold tuning instead of one threshold per head.

Add confidence scores & clarification questions to the chat experience.

Integrate this model end-to-end into a production chat UI.
