# predict_multi.py
import torch
from transformers import BertTokenizerFast, BertConfig
from src.multiheadBERT import MultiHeadBertForSequenceClassification
import src.config as cfg

# Load tokenizer + model from the directory you saved in training
MODEL_DIR = "models/multihead_bert"

tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
config = BertConfig.from_pretrained(MODEL_DIR)
model = MultiHeadBertForSequenceClassification.from_pretrained(MODEL_DIR, config=config)
model.eval()

def predict_all_slots(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    logits_occasion   = outputs["logits_occasion"]
    # probs = logits_occasion.softmax(-1)[0].tolist()

    # for i, p in enumerate(probs):
    #     print(cfg.OCCASION_LABELS[i], round(p, 3))
    logits_size       = outputs["logits_size"]
    logits_due_date = outputs["logits_due_date"]
    # logits_flavor = outputs["logits_flavor"]
    # logits_filling = outputs["logits_filling"]
    # logits_icing = outputs["logits_icing"]

    occ_id = logits_occasion.argmax(-1).item()
    size_id = logits_size.argmax(-1).item()
    date_id = logits_due_date.argmax(-1).item()

    # threshold = 0.65
    # #extracting probabiliities for multi-class slots
    # probs_flavor = logits_flavor.sigmoid()[0]
    
    # probs_filling = logits_filling.sigmoid()[0]
    # probs_icing = logits_icing.sigmoid()[0]

    # chosen_fl = (probs_flavor > threshold).nonzero(as_tuple=True)[0].tolist()
    
    # chosen_fi = (probs_filling > threshold).nonzero(as_tuple=True)[0].tolist()
    # chosen_i = (probs_icing > threshold).nonzero(as_tuple=True)[0].tolist()

    # pred_flavor = [cfg.FLAVOR_LABELS[i] for i in chosen_fl]
    # pred_filling = [cfg.FILLING_LABELS[i] for i in chosen_fi]
    # pred_icing = [cfg.ICING_LABELS[i] for i in chosen_i]



    return {
        "Text: ": text,
        "Occasion: ":   cfg.occasion_id2label[occ_id],
        "Size: ":       cfg.size_id2label[size_id],
        "Due_Date: ": cfg.date_id2label[date_id],
        # "Flavor: ": pred_flavor,
        # "Filling: ": pred_filling,
        # "Icing: ": pred_icing,
    }

if __name__ == "__main__":
    examples = [
        "I need a 10-inch birthday cake for my son next week",
        "We want a huge three-tier wedding cake with lots of decor for Sunday March 25th",
        "Just a simple small 7-inch cake to celebrate my graduation"
    ]
    for text in examples:
        print(text, "→", predict_all_slots(text))
