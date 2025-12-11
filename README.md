##WalterBot: Multi-Head BERT Slot Extraction##

WalterBot is a cake-ordering assistant that turns messy, natural-language messages into a structured JSON order the bakery can actually bake from.

Example: “I need a 10-inch marble and strawberry birthday cake for my son next week.”

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

##Highlights##

-Single BERT encoder (bert-base-uncased)

-Six heads on top of the pooled [CLS] embedding:
   -3 × single-label: Occasion, Size, Due_Date
   -3 × multi-label: Flavor, Filling, Icing

-Handles multiple flavors/fillings/icings per order (multi-label classification)
-Uses class-weighted BCE for rare fillings & icings
-Trained on augmented ~11K+ examples of realistic cake orders

##Architecture##

Everything lives in src/multiheadBERT.py.
Encoder: BertModel(config) from Hugging Face.
    - We take pooled_output = outputs.pooler_output
    - Shape: [batch_size, hidden_size] (e.g. [B, 768]).

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
   - Occasion / Size / Due_Date → [B, num_classes] (softmax)
   - Flavor / Filling / Icing → [B, num_labels] (sigmoid, multi-label)

Losses
Single-label slots:
   CrossEntropyLoss on:
   - logits_occasion vs labels_occasion
   - logits_size vs labels_size
   - logits_due_date vs labels_due_date

Multi-label slots:
   BCEWithLogitsLoss with pos_weight to handle imbalance:
   - Flavor → medium weight
   - Filling → higher weight
   - Icing → medium-high weight

Total loss: loss = loss_occasion + loss_size + loss_due_date + loss_flavor + loss_filling + loss_icing
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

##Data Format##
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
   - Handled in src/tokenize.py and src/config.py:
   - Uses BertTokenizerFast to create: input_ids, attention_mask, token_type_ids

Maps string labels → integer IDs:occasion_label2id, size_label2id, date_label2id

Multi-label slots become multi-hot vectors: 
   - labels_flavor → [num_flavors]
   - labels_filling → [num_fillings]
   - labels_icing → [num_icings]

##Training##

Training is done via Hugging Face’s Trainer in src/trainMulti.py.

Install Dependencies
pip install torch transformers datasets scikit-learn
pip install wandb

Run Training (Local)
From the project root: python3 -m src.trainMulti


##Metrics##

src/trainMulti.py defines compute_metrics(p) to evaluate each head:

For Occasion / Size / Due_Date (single-label): 
   - Accuracy
   - Macro-F1

For Flavor / Filling / Icing (multi-label):
   - Macro-F1 (using sigmoid + thresholds per slot)
   - We also print label mean vs prediction mean per head (e.g., how often a label is actually 1 vs predicted 1) to debug thresholds and class weights.

Example (best-run)
On a large augmented dataset (~11K orders):

Occasion Accuracy ≈ 99.8–99.9%
F1 (macro) ≈ 0.998+

SizeAccuracy ≈ 99.7–99.8%
F1 (macro) ≈ 0.997+

Due_DateAccuracy ≈ 99.9%
F1 (macro) ≈ 0.999+

Flavor (multi-label) F1 (macro) ≈ 0.97–0.98

Icing (multi-label) F1 (macro) ≈ ~0.74–0.75

Filling (multi-label) F1 (macro) ≈ ~0.53–0.55

Flavor is essentially “solved”; Filling and Icing are harder because real customer language blurs the line between flavor / filling / icing.

## Inference ##
Inference happens in src/predict.py (or src/predict_multi.py).
Run:

python3 -m src.predict

##Repo Structure##

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
